"""
verify.py
=========
Verification suite for the sensitivity analysis.  Run this before quoting any
number from `results/` in the manuscript.

Checks
------
 1. REGRESSION vs filter.py -- the refactored `pipeline.py`, run with the
    nominal parameters, must reproduce the arrays produced by the original
    `filter.py` code path bit-for-bit (or to within floating-point noise).
 2. The nominal row of grid A equals a fresh nominal computation.
 3. The clipping operation never touches negative excursions (this is the
    claim made to Reviewer 2, so it is asserted, not assumed).
 4. Integration is consistent with an independent Simpson integration.
 5. The sweep CSVs are complete: no missing combinations, no NaNs in the
    quantities that are quoted.
 6. Sanity ranges: I_peak within 100-140 kA, t_pinch within 2-4 us,
    FWHM within 1-100 ns.

Usage:  python verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid, simpson

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (                                       # noqa: E402
    BASELINE, load_all_shots, trim_and_time_axis, pre_stage,
    multi_segment_filter, clip_positive, integrate_kA, extract_features,
    ROGOWSKI_SCALE_KA, POINTS_TO_SKIP, SAMPLES_PER_EXPERIMENT,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PASS, FAIL = "  PASS", "  FAIL"
failures = []


def check(name, cond, detail=""):
    print(f"{PASS if cond else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


# ==========================================================================
# Reference implementation: a literal transcription of filter.py's algorithm
# ==========================================================================
def filter_py_reference(data, t):
    """Independent re-implementation copied from the original filter.py."""
    dt_us = 80 * 1e-12 * 1e6

    smoothed = savgol_filter(data, 51, 3, deriv=0, delta=dt_us, mode="nearest")
    mask = (t >= 2.0) & (t <= 4.0)
    t_win, y_win = t[mask], smoothed[mask]
    slope, intercept = np.polyfit(t_win, y_win, 1)
    base = slope * t_win + intercept
    avg = np.mean(base)
    th_h = base - 0.50 * abs(avg)
    th_l = base - 0.70 * abs(avg)

    state, tmp, ps, pe = "IDLE", None, None, None
    for i in range(len(y_win)):
        ct, cd = t_win[i], y_win[i]
        if state == "IDLE":
            if cd < th_h[i]:
                tmp, state = ct, "ARMED_FALL"
        elif state == "ARMED_FALL":
            if cd < th_l[i]:
                ps, state = tmp, "IN_PULSE"
            elif cd >= th_h[i]:
                state = "IDLE"
        elif state == "IN_PULSE":
            if cd > th_l[i]:
                state = "ARMED_RISE"
        elif state == "ARMED_RISE":
            if cd > th_h[i]:
                pe = ct
                break
            elif cd <= th_l[i]:
                state = "IN_PULSE"

    segs = []
    i0 = int(np.searchsorted(t, ps))
    i1 = int(np.searchsorted(t, pe))
    if i0 > 0:
        s = data[:i0]
        segs.append(savgol_filter(s, 1001, 3, mode="interp")
                    if len(s) >= 1001 else s)
    if i1 > i0:
        s = data[i0:i1]
        segs.append(savgol_filter(s, 11, 3, mode="interp")
                    if len(s) >= 11 else s)
    if i1 < len(data):
        s = data[i1:]
        segs.append(savgol_filter(s, 1001, 3, mode="interp")
                    if len(s) >= 1001 else s)
    filt = np.concatenate(segs)

    off = 0.40 * abs(np.mean(base))
    filt = filt.copy()
    filt[mask] = np.minimum(filt[mask], base + off)
    integ = cumulative_trapezoid(filt, t, initial=0) * 28846 / 1000
    return ps, pe, filt, integ


# ==========================================================================
def main():
    print("=" * 74)
    print("VERIFICATION SUITE  --  TPF-2 signal-processing sensitivity analysis")
    print("=" * 74)

    data_path = os.path.join(HERE, "..", "PD_Ex12.txt")
    shots = load_all_shots(os.path.abspath(data_path))
    print(f"\nloaded {shots.shape[0]} shots x {shots.shape[1]} samples\n")

    check("data splits into whole shots",
          shots.shape[1] == SAMPLES_PER_EXPERIMENT)
    check("trim length is positive",
          shots.shape[1] - POINTS_TO_SKIP > 0,
          f"{shots.shape[1] - POINTS_TO_SKIP} samples retained")

    # ---- 1. regression against filter.py on three shots ------------------
    print("\n[1] regression against the original filter.py algorithm")
    for s in (1, 12, 20):
        t, d = trim_and_time_axis(shots[s - 1])
        ps, pe, filt_ref, integ_ref = filter_py_reference(d, t)
        pre = pre_stage(t, d)
        filt = multi_segment_filter(d, pre, BASELINE["sg_window_pulse"],
                                    BASELINE["sg_window_flank"],
                                    BASELINE["sg_polyorder"])
        clipped, _, _ = clip_positive(filt, pre, BASELINE["clip_pct"])
        integ = integrate_kA(t, clipped)

        check(f"shot {s:>2d}: pulse start identical",
              np.isclose(ps, pre.pulse_start, atol=1e-12))
        check(f"shot {s:>2d}: pulse end identical",
              np.isclose(pe, pre.pulse_end, atol=1e-12))
        dmax = np.max(np.abs(clipped - filt_ref))
        check(f"shot {s:>2d}: filtered+clipped arrays identical",
              dmax < 1e-9, f"max |diff| = {dmax:.2e} V")
        imax = np.max(np.abs(integ - integ_ref))
        check(f"shot {s:>2d}: integrated current identical",
              imax < 1e-9, f"max |diff| = {imax:.2e} kA")

    # ---- 2. nominal row of grid A ---------------------------------------
    print("\n[2] grid A nominal row equals a fresh nominal computation")
    dfA = pd.read_csv(os.path.join(RESULTS, "sweep_gridA.csv"))
    for s in (1, 12, 20):
        t, d = trim_and_time_axis(shots[s - 1])
        pre = pre_stage(t, d)
        filt = multi_segment_filter(d, pre, BASELINE["sg_window_pulse"],
                                    BASELINE["sg_window_flank"],
                                    BASELINE["sg_polyorder"])
        clipped, _, _ = clip_positive(filt, pre, BASELINE["clip_pct"])
        f = extract_features(t, clipped, integrate_kA(t, clipped), pre)
        r = dfA[(dfA.shot == s) & (dfA.sg_polyorder == 3) &
                (dfA.sg_window_pulse == 11) & (dfA.clip_pct == 40)].iloc[0]
        check(f"shot {s:>2d}: t_pinch matches CSV",
              np.isclose(f["t_pinch_us"], r.t_pinch_us, atol=1e-9))
        check(f"shot {s:>2d}: I_peak matches CSV",
              np.isclose(f["I_peak_kA"], r.I_peak_kA, atol=1e-9))

    # ---- 3. clipping never touches negative excursions -------------------
    print("\n[3] clipping is strictly one-sided (the claim made to Reviewer 2)")
    t, d = trim_and_time_axis(shots[11])
    pre = pre_stage(t, d)
    filt = multi_segment_filter(d, pre, 11, 1001, 3)
    for clip in (20, 30, 40, 50, 60):
        clipped, upper, _ = clip_positive(filt, pre, clip)
        below = filt[pre.mask] < pre.baseline_win
        check(f"clip {clip:>2d}%: samples below baseline are unmodified",
              np.array_equal(clipped[pre.mask][below], filt[pre.mask][below]))
        check(f"clip {clip:>2d}%: clipped signal never exceeds the bound",
              np.all(clipped[pre.mask] <= upper + 1e-12))

    # ---- 4. integration cross-check --------------------------------------
    print("\n[4] trapezoidal integration cross-checked against Simpson")
    clipped, _, _ = clip_positive(filt, pre, 40)
    trap = integrate_kA(t, clipped)[-1]
    simp = simpson(clipped, x=t) * ROGOWSKI_SCALE_KA
    rel = abs(trap - simp) / abs(simp) * 100
    check("trapezoid vs Simpson agree to < 0.01 %", rel < 0.01,
          f"{rel:.2e} %")

    # ---- 5. CSV completeness --------------------------------------------
    print("\n[5] sweep result completeness")
    n_shots = dfA.shot.nunique()
    expected = n_shots * 3 * 5 * 5
    check("grid A has every (shot, p, W_pulse, clip) combination",
          len(dfA) == expected, f"{len(dfA)} rows, expected {expected}")
    for col in ("t_pinch_us", "I_peak_kA", "pinch_fwhm_ns"):
        check(f"grid A: no NaN in {col}", dfA[col].notna().all(),
              f"{int(dfA[col].isna().sum())} NaN")
    nom = pd.read_csv(os.path.join(RESULTS, "nominal_per_shot.csv"))
    check("nominal table has one row per shot", len(nom) == n_shots)

    # ---- 6. physical sanity ---------------------------------------------
    print("\n[6] physical sanity of the extracted quantities")
    check("I_peak in 100-140 kA on every shot and every parameter set",
          dfA.I_peak_kA.between(100, 140).all(),
          f"{dfA.I_peak_kA.min():.1f} - {dfA.I_peak_kA.max():.1f} kA")
    check("t_pinch inside the 2-4 us analysis window",
          dfA.t_pinch_us.between(2.0, 4.0).all(),
          f"{dfA.t_pinch_us.min():.3f} - {dfA.t_pinch_us.max():.3f} us")
    check("pinch FWHM in 1-100 ns",
          dfA.pinch_fwhm_ns.between(1, 100).all(),
          f"{dfA.pinch_fwhm_ns.min():.1f} - {dfA.pinch_fwhm_ns.max():.1f} ns")
    check("t_pinch precedes the current maximum on every shot",
          (nom.t_pinch_us < nom.t_Ipeak_us).all())
    # analytic short-circuit current, independent of any model or fit
    L0, C0, V0 = 212e-9, 25e-6, 11e3
    I0 = V0 * np.sqrt(C0 / L0) / 1e3
    dev = 100 * abs(nom.I_peak_kA.mean() - I0) / I0
    check("measured I_peak agrees with V0*sqrt(C0/L0) to < 5 %", dev < 5,
          f"analytic {I0:.1f} kA vs measured {nom.I_peak_kA.mean():.1f} kA "
          f"({dev:.2f} %)")

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 74)
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for f in failures:
            print(f"   - {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()
