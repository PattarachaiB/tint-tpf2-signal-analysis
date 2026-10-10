"""
pipeline.py
===========
Re-usable, fully parameterised implementation of the TPF-2 Rogowski-coil
signal-recovery chain originally implemented in ``filter.py``.

The chain is:

  (0) read raw di/dt record, split into shots, trim pre-trigger samples
  (1) *fixed* pre-smoothing stage (Savitzky-Golay, W = 51, p = 3) used ONLY to
      locate the negative pulse (the pinch) and to fit the local baseline
      (linear trend over 2-4 us)
  (2) hysteretic (two-threshold) state machine -> pulse start / end
  (3) multi-segment Savitzky-Golay filtering:
        flank segments (before / after the pulse)  -> W_flank, p
        pulse segment                              -> W_pulse, p
  (4) one-sided (positive) clipping of the filtered signal against a band
      parallel to the fitted baseline, at +CLIP_PCT % of |baseline|
  (5) cumulative trapezoidal integration -> discharge current, scaled to kA

Every stage that the reviewers asked us to justify (H3) is exposed as an
argument of :func:`process_shot`, so that a sensitivity sweep can be run
without touching the algorithm itself.

Author: TPF-2 team, KMITL / TINT
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid

# --------------------------------------------------------------------------
# Acquisition constants (identical to filter.py)
# --------------------------------------------------------------------------
TIME_INTERVAL_PS = 80                 # PXIe-5185 @ 12.5 GS/s
SAMPLES_PER_EXPERIMENT = 375_000
POINTS_TO_SKIP = 181_250
START_TIME_US_AFTER_SKIP = -0.5
HEADER_SKIP_KEYWORD = "di/dt"

#: Rogowski chain calibration:  V*us  ->  kA   (28846 A per V*us / 1000)
ROGOWSKI_SCALE_KA = 28846.0 / 1000.0

#: analysis window in which the pinch is searched / baseline fitted (us)
ANALYSIS_WINDOW_US = (2.0, 4.0)

# --------------------------------------------------------------------------
# Baseline (nominal) processing parameters -- exactly filter.py's values
# --------------------------------------------------------------------------
BASELINE = dict(
    sg_window_initial=51,        # pre-smoothing for pulse detection (fixed)
    sg_polyorder_initial=3,
    pct_threshold_inner=50,      # PERCENT_THRESHOLD_INNER
    pct_threshold_outer=70,      # PERCENT_THRESHOLD_OUTER
    sg_window_pulse=11,          # SG_WINDOW_PULSE_SEGMENT
    sg_window_flank=1001,        # SG_WINDOW_FLANK_SEGMENTS
    sg_polyorder=3,              # SG_POLYORDER_SEGMENT
    clip_pct=40,                 # MAX_POSITIVE_DEVIATION_PERCENT
)


# ==========================================================================
# 1. Data loading
# ==========================================================================
def load_all_shots(file_path: str, cache: bool = True) -> np.ndarray:
    """Read the raw ASCII record and return an array of shape (n_shots, N).

    A ``.npy`` cache is written next to the source file the first time, because
    parsing 7.5 x 10^6 ASCII floats takes ~30 s while the cache loads in <1 s.
    """
    cache_path = os.path.splitext(file_path)[0] + "_shots.npy"
    if cache and os.path.exists(cache_path):
        if os.path.getmtime(cache_path) >= os.path.getmtime(file_path):
            return np.load(cache_path)

    import pandas as pd

    # locate the end of the header
    n_header = 0
    with open(file_path, "r", errors="ignore") as fh:
        for line in fh:
            n_header += 1
            if HEADER_SKIP_KEYWORD in line:
                break

    values = pd.read_csv(
        file_path, skiprows=n_header, header=None, names=["v"],
        dtype=np.float64, engine="c", na_filter=False,
    )["v"].to_numpy()

    n_shots = values.size // SAMPLES_PER_EXPERIMENT
    if n_shots == 0:
        raise RuntimeError(f"not enough samples in {file_path}")
    shots = values[: n_shots * SAMPLES_PER_EXPERIMENT].reshape(
        n_shots, SAMPLES_PER_EXPERIMENT
    )
    if cache:
        np.save(cache_path, shots)
    return shots


def trim_and_time_axis(shot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Drop the pre-trigger samples and build the matching time axis (us)."""
    data = np.asarray(shot[POINTS_TO_SKIP:], dtype=np.float64)
    dt_us = TIME_INTERVAL_PS * 1e-12 * 1e6           # 8.0e-5 us
    t = START_TIME_US_AFTER_SKIP + np.arange(data.size) * dt_us
    return t, data


# ==========================================================================
# 2. Fixed pre-stage: baseline fit + hysteretic pulse detection
# ==========================================================================
@dataclass
class PreStage:
    """Everything that does NOT depend on the swept parameters."""
    mask: np.ndarray                      # boolean mask of the 2-4 us window
    t_win: np.ndarray                     # time inside the window
    slope: float
    intercept: float
    baseline_win: np.ndarray              # fitted baseline inside the window
    pulse_start: Optional[float]
    pulse_end: Optional[float]
    idx_start: Optional[int]
    idx_end: Optional[int]
    smoothed: np.ndarray = field(repr=False)


def pre_stage(t: np.ndarray, data: np.ndarray,
              sg_window_initial: int = BASELINE["sg_window_initial"],
              sg_polyorder_initial: int = BASELINE["sg_polyorder_initial"],
              pct_inner: float = BASELINE["pct_threshold_inner"],
              pct_outer: float = BASELINE["pct_threshold_outer"],
              window_us: Tuple[float, float] = ANALYSIS_WINDOW_US) -> PreStage:
    """Reproduce steps (1) and (2) of ``filter.py``."""
    dt_us = TIME_INTERVAL_PS * 1e-12 * 1e6
    smoothed = savgol_filter(data, sg_window_initial, sg_polyorder_initial,
                             deriv=0, delta=dt_us, mode="nearest")

    mask = (t >= window_us[0]) & (t <= window_us[1])
    t_win = t[mask]
    y_win = smoothed[mask]

    slope, intercept = np.polyfit(t_win, y_win, 1)
    baseline_win = slope * t_win + intercept
    base_mean = np.mean(baseline_win)

    off_in = (pct_inner / 100.0) * abs(base_mean)
    off_out = (pct_outer / 100.0) * abs(base_mean)
    th_h = baseline_win - off_in          # inner  (armed)
    th_l = baseline_win - off_out         # outer  (confirmed)

    state, tmp_start = "IDLE", None
    pulse_start = pulse_end = None
    for i, (tt, yy) in enumerate(zip(t_win, y_win)):
        if state == "IDLE":
            if yy < th_h[i]:
                tmp_start, state = tt, "ARMED_FALL"
        elif state == "ARMED_FALL":
            if yy < th_l[i]:
                pulse_start, state = tmp_start, "IN_PULSE"
            elif yy >= th_h[i]:
                state = "IDLE"
        elif state == "IN_PULSE":
            if yy > th_l[i]:
                state = "ARMED_RISE"
        elif state == "ARMED_RISE":
            if yy > th_h[i]:
                pulse_end = tt
                break
            elif yy <= th_l[i]:
                state = "IN_PULSE"

    idx_start = idx_end = None
    if pulse_start is not None and pulse_end is not None:
        idx_start = int(np.searchsorted(t, pulse_start))
        idx_end = int(np.searchsorted(t, pulse_end))
        idx_start = max(0, min(idx_start, data.size))
        idx_end = max(idx_start, min(idx_end, data.size))

    return PreStage(mask=mask, t_win=t_win, slope=float(slope),
                    intercept=float(intercept), baseline_win=baseline_win,
                    pulse_start=pulse_start, pulse_end=pulse_end,
                    idx_start=idx_start, idx_end=idx_end, smoothed=smoothed)


# ==========================================================================
# 3. Multi-segment filtering + clipping + integration
# ==========================================================================
def multi_segment_filter(data: np.ndarray, pre: PreStage,
                         w_pulse: int, w_flank: int, poly: int) -> np.ndarray:
    """Step (3): different SG window inside and outside the detected pulse."""
    def _sg(seg, win):
        if win > poly and win % 2 == 1 and seg.size >= win:
            try:
                return savgol_filter(seg, win, poly, mode="interp")
            except ValueError:
                return seg
        return seg

    if pre.idx_start is None or pre.idx_end is None:
        return _sg(data, w_flank).copy()

    parts = []
    if pre.idx_start > 0:
        parts.append(_sg(data[: pre.idx_start], w_flank))
    if pre.idx_end > pre.idx_start:
        parts.append(_sg(data[pre.idx_start: pre.idx_end], w_pulse))
    if pre.idx_end < data.size:
        parts.append(_sg(data[pre.idx_end:], w_flank))

    out = np.concatenate(parts) if parts else data.copy()
    if out.size != data.size:
        out = _sg(data, w_flank).copy()
    return np.asarray(out, dtype=np.float64)


def clip_positive(filtered: np.ndarray, pre: PreStage,
                  clip_pct: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step (4): one-sided clip against a band parallel to the baseline.

    Returns (clipped_signal, upper_bound_in_window, lower_bound_in_window).
    The lower bound is drawn for visualisation only -- it is never applied,
    which is precisely why the clipping cannot displace the negative pinch
    feature (see the reply to Reviewer 2, point 4).
    """
    out = filtered.copy()
    base = pre.baseline_win
    offset = (clip_pct / 100.0) * abs(np.mean(base))
    upper = base + offset
    lower = base - offset
    out[pre.mask] = np.minimum(out[pre.mask], upper)
    return out, upper, lower


def integrate_kA(t: np.ndarray, sig: np.ndarray) -> np.ndarray:
    """Step (5): cumulative trapezoid, scaled to kA."""
    return cumulative_trapezoid(sig, t, initial=0.0) * ROGOWSKI_SCALE_KA


# ==========================================================================
# 4. Feature extraction (definitions used for the paper)
# ==========================================================================
def _fwhm_of_dip(t_win: np.ndarray, depth: np.ndarray) -> Tuple[float, float, float]:
    """Full width at half minimum of the current dip.

    ``depth`` is (baseline - signal) inside the analysis window, so the pinch
    is a positive maximum.  Crossings are found by linear interpolation, which
    makes the result independent of the 80 ps sample grid.
    Returns (fwhm_ns, t_left, t_right); NaN if the dip is not resolved.
    """
    i_pk = int(np.argmax(depth))
    half = depth[i_pk] / 2.0
    if not np.isfinite(half) or half <= 0:
        return np.nan, np.nan, np.nan

    # walk left
    j = i_pk
    while j > 0 and depth[j] > half:
        j -= 1
    if depth[j] > half:
        return np.nan, np.nan, np.nan
    t_l = np.interp(half, [depth[j], depth[j + 1]], [t_win[j], t_win[j + 1]])

    # walk right
    k = i_pk
    n = depth.size
    while k < n - 1 and depth[k] > half:
        k += 1
    if depth[k] > half:
        return np.nan, np.nan, np.nan
    t_r = np.interp(half, [depth[k], depth[k - 1]], [t_win[k], t_win[k - 1]])

    return float((t_r - t_l) * 1e3), float(t_l), float(t_r)   # ns


def extract_features(t: np.ndarray, sig: np.ndarray, integ: np.ndarray,
                     pre: PreStage) -> Dict[str, float]:
    """Quantities that go into the manuscript, all with explicit definitions."""
    depth = pre.baseline_win - sig[pre.mask]        # positive at the pinch
    i_pk = int(np.argmax(depth))

    t_pinch = float(pre.t_win[i_pk])                # time of maximum current dip
    dip_depth_V = float(depth[i_pk])
    fwhm_ns, t_l, t_r = _fwhm_of_dip(pre.t_win, depth)

    i_peak = float(np.nanmax(integ))
    t_ipeak = float(t[int(np.nanargmax(integ))])
    i_at_pinch = float(np.interp(t_pinch, t, integ))

    return dict(
        t_pinch_us=t_pinch,
        pinch_fwhm_ns=fwhm_ns,
        pinch_half_left_us=t_l,
        pinch_half_right_us=t_r,
        dip_depth_V=dip_depth_V,
        I_peak_kA=i_peak,
        t_Ipeak_us=t_ipeak,
        I_at_pinch_kA=i_at_pinch,
        pulse_start_us=(np.nan if pre.pulse_start is None else float(pre.pulse_start)),
        pulse_end_us=(np.nan if pre.pulse_end is None else float(pre.pulse_end)),
        pulse_duration_ns=(
            np.nan if (pre.pulse_start is None or pre.pulse_end is None)
            else float((pre.pulse_end - pre.pulse_start) * 1e3)
        ),
    )


# ==========================================================================
# 5. Independent, filter-free reference (Reviewer 3 / plan item H3.3)
# ==========================================================================
def raw_reference(t: np.ndarray, data: np.ndarray, pre: PreStage,
                  median_k: int = 0) -> Dict[str, float]:
    """Pinch time obtained from the UNFILTERED di/dt record.

    ``median_k = 0`` -> strictly raw (argmin of the raw trace inside 2-4 us).
    ``median_k = k`` -> k-point running median first; this is *not* a
    smoothing of the physics, only a rejection of single-sample digitiser
    outliers, and is reported alongside the strictly-raw value.
    """
    y = data[pre.mask]
    if median_k and median_k > 1:
        from scipy.signal import medfilt
        kk = median_k if median_k % 2 == 1 else median_k + 1
        y = medfilt(y, kernel_size=kk)
    depth = pre.baseline_win - y
    i_pk = int(np.argmax(depth))
    fwhm_ns, _, _ = _fwhm_of_dip(pre.t_win, depth)
    return dict(t_pinch_raw_us=float(pre.t_win[i_pk]),
                raw_dip_depth_V=float(depth[i_pk]),
                raw_fwhm_ns=fwhm_ns)


def coil_resonance_MHz(t: np.ndarray, data: np.ndarray, pre: PreStage,
                       t_pinch: float, t_from: float = 0.05,
                       t_to: float = 0.45, f_min_MHz: float = 10.0) -> float:
    """Dominant self-resonance of the Rogowski coil, from the post-pinch ring.

    A Hann-windowed FFT of the raw record between ``t_pinch + t_from`` and
    ``t_pinch + t_to`` (us).  This frequency sets the scale of the artefact the
    multi-segment filter has to remove, and it explains the discrete +/- one
    period jumps occasionally seen in the filter-free pinch-time estimate.
    """
    m = (t > t_pinch + t_from) & (t < t_pinch + t_to)
    y = data[m]
    if y.size < 64:
        return np.nan
    y = (y - y.mean()) * np.hanning(y.size)
    dt_s = TIME_INTERVAL_PS * 1e-12
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(y.size, dt_s)
    # ignore residual baseline curvature below f_min_MHz
    band = freq >= f_min_MHz * 1e6
    if not np.any(band):
        return np.nan
    return float(freq[band][int(np.argmax(spec[band]))] / 1e6)


# ==========================================================================
# 6. One-call convenience wrapper
# ==========================================================================
def process_shot(t: np.ndarray, data: np.ndarray, pre: PreStage,
                 w_pulse: int, w_flank: int, poly: int, clip_pct: float,
                 return_traces: bool = False) -> Dict[str, Any]:
    """Run stages (3)-(5) for one parameter set and extract the features."""
    filt = multi_segment_filter(data, pre, w_pulse, w_flank, poly)
    clipped, upper, lower = clip_positive(filt, pre, clip_pct)
    integ = integrate_kA(t, clipped)
    res = extract_features(t, clipped, integ, pre)
    res.update(sg_polyorder=poly, sg_window_pulse=w_pulse,
               sg_window_flank=w_flank, clip_pct=clip_pct)
    if return_traces:
        res["_traces"] = dict(filtered=clipped, integrated=integ,
                              upper=upper, lower=lower)
    return res
