"""
run_sweep.py
============
Parameter-sensitivity sweep of the TPF-2 signal-recovery chain
(revision plan, item **H3** -- "Signal processing may create artefacts:
a robustness check is required").

What is swept
-------------
Grid A  (main, reported in the manuscript)
    Savitzky-Golay polynomial order   p        = 2, 3, 4
    pulse-segment window              W_pulse  = 5, 7, 11, 15, 21 samples
    positive clipping threshold       C        = 20, 30, 40, 50, 60 %
    -> 3 x 5 x 5 = 75 combinations, on every shot

Grid B  (supplementary)
    flank-segment window              W_flank  = 501, 751, 1001, 1251, 1501
    (baseline +/- 50 %), all other parameters nominal

Grid C  (supplementary)
    pulse-detection hysteresis thresholds (inner / outer)
      = (40,60) (45,65) (50,70) (55,75) (60,80) %
    This one changes the *segmentation* itself, so the pre-stage is re-run.

Reference
---------
For every shot the pinch time is additionally obtained from the **unfiltered**
di/dt record (argmin inside 2-4 us) so that the processed value can be
compared against a filter-free estimate.

Outputs (written to ./results/)
-------------------------------
    sweep_gridA.csv          one row per (shot, p, W_pulse, C)
    sweep_gridB.csv          one row per (shot, W_flank)
    sweep_gridC.csv          one row per (shot, threshold pair)
    raw_reference.csv        one row per shot
    summary_stats.csv        aggregated spreads used in the manuscript text
    nominal_per_shot.csv     nominal-parameter result for each shot (H1 input)

Usage
-----
    python run_sweep.py [--data ../PD_Ex12.txt] [--jobs 4]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline import (                                    # noqa: E402
    BASELINE, ANALYSIS_WINDOW_US,
    load_all_shots, trim_and_time_axis, pre_stage,
    multi_segment_filter, clip_positive, integrate_kA,
    extract_features, raw_reference, coil_resonance_MHz,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# ----------------------------- sweep grids --------------------------------
POLY_VALUES = [2, 3, 4]
W_PULSE_VALUES = [5, 7, 11, 15, 21]          # nominal 11, ~ +/- 50 % and wider
CLIP_VALUES = [20, 30, 40, 50, 60]           # nominal 40 %
W_FLANK_VALUES = [501, 751, 1001, 1251, 1501]  # nominal 1001, +/- 50 %
THRESHOLD_PAIRS = [(40, 60), (45, 65), (50, 70), (55, 75), (60, 80)]


# ==========================================================================
def analyse_one_shot(args):
    """Worker: run every grid for a single shot. Returns three record lists."""
    shot_idx, shot = args
    t, data = trim_and_time_axis(shot)

    pre = pre_stage(t, data)
    if pre.pulse_start is None or pre.pulse_end is None:
        return shot_idx, [], [], [], dict(shot=shot_idx + 1, detected=False)

    rows_a, rows_b, rows_c = [], [], []

    # ---- Grid A ---------------------------------------------------------
    # Savitzky-Golay filtering is the expensive step and does not depend on
    # the clipping threshold, so it is computed once per (p, W_pulse) and
    # reused across the five clipping levels.
    for poly, w_pulse in itertools.product(POLY_VALUES, W_PULSE_VALUES):
        filt = multi_segment_filter(data, pre, w_pulse,
                                    BASELINE["sg_window_flank"], poly)
        for clip in CLIP_VALUES:
            clipped, _, _ = clip_positive(filt, pre, clip)
            integ = integrate_kA(t, clipped)
            rec = extract_features(t, clipped, integ, pre)
            rec.update(shot=shot_idx + 1, grid="A", sg_polyorder=poly,
                       sg_window_pulse=w_pulse,
                       sg_window_flank=BASELINE["sg_window_flank"],
                       clip_pct=clip,
                       pct_inner=BASELINE["pct_threshold_inner"],
                       pct_outer=BASELINE["pct_threshold_outer"])
            rows_a.append(rec)

    # ---- Grid B : flank window -----------------------------------------
    for w_flank in W_FLANK_VALUES:
        filt = multi_segment_filter(data, pre, BASELINE["sg_window_pulse"],
                                    w_flank, BASELINE["sg_polyorder"])
        clipped, _, _ = clip_positive(filt, pre, BASELINE["clip_pct"])
        integ = integrate_kA(t, clipped)
        rec = extract_features(t, clipped, integ, pre)
        rec.update(shot=shot_idx + 1, grid="B",
                   sg_polyorder=BASELINE["sg_polyorder"],
                   sg_window_pulse=BASELINE["sg_window_pulse"],
                   sg_window_flank=w_flank, clip_pct=BASELINE["clip_pct"],
                   pct_inner=BASELINE["pct_threshold_inner"],
                   pct_outer=BASELINE["pct_threshold_outer"])
        rows_b.append(rec)

    # ---- Grid C : detection hysteresis ---------------------------------
    for p_in, p_out in THRESHOLD_PAIRS:
        pre_c = pre_stage(t, data, pct_inner=p_in, pct_outer=p_out)
        if pre_c.pulse_start is None or pre_c.pulse_end is None:
            rows_c.append(dict(shot=shot_idx + 1, grid="C", pct_inner=p_in,
                               pct_outer=p_out, detected=False,
                               t_pinch_us=np.nan, I_peak_kA=np.nan,
                               pinch_fwhm_ns=np.nan))
            continue
        filt = multi_segment_filter(data, pre_c, BASELINE["sg_window_pulse"],
                                    BASELINE["sg_window_flank"],
                                    BASELINE["sg_polyorder"])
        clipped, _, _ = clip_positive(filt, pre_c, BASELINE["clip_pct"])
        integ = integrate_kA(t, clipped)
        rec = extract_features(t, clipped, integ, pre_c)
        rec.update(shot=shot_idx + 1, grid="C", detected=True,
                   sg_polyorder=BASELINE["sg_polyorder"],
                   sg_window_pulse=BASELINE["sg_window_pulse"],
                   sg_window_flank=BASELINE["sg_window_flank"],
                   clip_pct=BASELINE["clip_pct"],
                   pct_inner=p_in, pct_outer=p_out)
        rows_c.append(rec)

    # ---- filter-free reference -----------------------------------------
    ref = dict(shot=shot_idx + 1, detected=True)
    ref.update(raw_reference(t, data, pre, median_k=0))
    r5 = raw_reference(t, data, pre, median_k=5)
    ref["t_pinch_raw_med5_us"] = r5["t_pinch_raw_us"]
    ref["raw_fwhm_med5_ns"] = r5["raw_fwhm_ns"]
    ref["baseline_slope_V_per_us"] = pre.slope
    ref["baseline_intercept_V"] = pre.intercept
    # dominant Rogowski-coil self-resonance (sets the artefact time-scale)
    f_ring = coil_resonance_MHz(t, data, pre, ref["t_pinch_raw_us"])
    ref["coil_resonance_MHz"] = f_ring
    ref["coil_ring_period_ns"] = (np.nan if not np.isfinite(f_ring)
                                  else 1e3 / f_ring)

    return shot_idx, rows_a, rows_b, rows_c, ref


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "PD_Ex12.txt"))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--shots", type=int, default=0,
                    help="limit to first N shots (0 = all)")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)

    t0 = time.time()
    print(f"[1/4] loading {args.data} ...", flush=True)
    shots = load_all_shots(os.path.abspath(args.data))
    if args.shots:
        shots = shots[: args.shots]
    print(f"      {shots.shape[0]} shots x {shots.shape[1]} samples "
          f"({time.time()-t0:.1f} s)", flush=True)

    n_a = len(POLY_VALUES) * len(W_PULSE_VALUES) * len(CLIP_VALUES)
    print(f"[2/4] sweeping: grid A = {n_a} combos/shot, "
          f"grid B = {len(W_FLANK_VALUES)}, grid C = {len(THRESHOLD_PAIRS)}",
          flush=True)

    payload = [(i, shots[i]) for i in range(shots.shape[0])]
    A, B, C, REF = [], [], [], []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for k, (idx, ra, rb, rc, ref) in enumerate(
                    ex.map(analyse_one_shot, payload), 1):
                A += ra; B += rb; C += rc; REF.append(ref)
                print(f"      shot {idx+1:>2d} done ({k}/{len(payload)})",
                      flush=True)
    else:
        for k, item in enumerate(payload, 1):
            idx, ra, rb, rc, ref = analyse_one_shot(item)
            A += ra; B += rb; C += rc; REF.append(ref)
            print(f"      shot {idx+1:>2d} done ({k}/{len(payload)})", flush=True)

    dfA, dfB, dfC = pd.DataFrame(A), pd.DataFrame(B), pd.DataFrame(C)
    dfR = pd.DataFrame(REF)

    print("[3/4] writing CSVs ...", flush=True)
    dfA.to_csv(os.path.join(RESULTS, "sweep_gridA.csv"), index=False)
    dfB.to_csv(os.path.join(RESULTS, "sweep_gridB.csv"), index=False)
    dfC.to_csv(os.path.join(RESULTS, "sweep_gridC.csv"), index=False)
    dfR.to_csv(os.path.join(RESULTS, "raw_reference.csv"), index=False)

    # ---- nominal result per shot (feeds the H1 uncertainty table) --------
    nom = dfA[(dfA.sg_polyorder == BASELINE["sg_polyorder"]) &
              (dfA.sg_window_pulse == BASELINE["sg_window_pulse"]) &
              (dfA.clip_pct == BASELINE["clip_pct"])].copy()
    nom = nom.merge(dfR[["shot", "t_pinch_raw_us", "t_pinch_raw_med5_us",
                         "coil_resonance_MHz", "coil_ring_period_ns"]],
                    on="shot", how="left")
    nom["delta_raw_ns"] = (nom.t_pinch_us - nom.t_pinch_raw_us) * 1e3
    nom["delta_raw_med5_ns"] = (nom.t_pinch_us - nom.t_pinch_raw_med5_us) * 1e3
    # Classify the residual: on a few shots the argmin of the *unfiltered*
    # trace locks onto the adjacent trough of the coil self-resonance, i.e.
    # it is displaced by exactly one ringing period.  That is a failure of the
    # filter-free estimator, not of the processing chain.
    nom["delta_in_ring_periods"] = nom.delta_raw_ns / nom.coil_ring_period_ns
    nom["ring_period_lock"] = (
        np.abs(np.abs(nom.delta_in_ring_periods) - 1.0) < 0.15)
    nom.to_csv(os.path.join(RESULTS, "nominal_per_shot.csv"), index=False)

    # ---- summary statistics ---------------------------------------------
    print("[4/4] summarising ...", flush=True)
    rows = []

    def add(label, series, unit, extra=""):
        s = pd.Series(series).dropna()
        if s.empty:
            return
        rows.append(dict(quantity=label, unit=unit, n=len(s),
                         mean=s.mean(), sd=s.std(ddof=1),
                         min=s.min(), max=s.max(),
                         peak_to_peak=s.max() - s.min(), note=extra))

    # within-shot spread across the whole of grid A, then pooled
    g = dfA.groupby("shot")
    add("t_pinch, spread within grid A (per shot, max-min)",
        g.t_pinch_us.apply(lambda s: (s.max() - s.min()) * 1e3), "ns",
        "processing-induced spread, 75 combinations")
    add("I_peak, relative spread within grid A (per shot)",
        g.I_peak_kA.apply(lambda s: 100 * (s.max() - s.min()) / s.mean()), "%",
        "processing-induced spread, 75 combinations")
    add("pinch FWHM, spread within grid A (per shot, max-min)",
        g.pinch_fwhm_ns.apply(lambda s: s.max() - s.min()), "ns", "")

    gb = dfB.groupby("shot")
    add("t_pinch, spread over W_flank (per shot)",
        gb.t_pinch_us.apply(lambda s: (s.max() - s.min()) * 1e3), "ns", "grid B")
    add("I_peak, relative spread over W_flank (per shot)",
        gb.I_peak_kA.apply(lambda s: 100 * (s.max() - s.min()) / s.mean()), "%",
        "grid B")

    gc = dfC[dfC.get("detected", True) == True].groupby("shot")  # noqa: E712
    add("t_pinch, spread over detection thresholds (per shot)",
        gc.t_pinch_us.apply(lambda s: (s.max() - s.min()) * 1e3), "ns", "grid C")
    add("I_peak, relative spread over detection thresholds (per shot)",
        gc.I_peak_kA.apply(lambda s: 100 * (s.max() - s.min()) / s.mean()), "%",
        "grid C")

    # shot-to-shot statistics at nominal parameters (this is the H1 number)
    add("t_pinch, shot-to-shot at nominal parameters", nom.t_pinch_us, "us",
        "physical shot-to-shot variability")
    add("I_peak, shot-to-shot at nominal parameters", nom.I_peak_kA, "kA",
        "physical shot-to-shot variability")
    add("pinch FWHM, shot-to-shot at nominal parameters",
        nom.pinch_fwhm_ns, "ns", "physical shot-to-shot variability")
    add("t_pinch(processed) - t_pinch(raw, unfiltered)",
        nom.delta_raw_ns, "ns", "independent filter-free check")
    add("t_pinch(processed) - t_pinch(raw, 5-pt median)",
        nom.delta_raw_med5_ns, "ns", "independent filter-free check")
    add("t_pinch(processed) - t_pinch(raw), excluding ring-period lock",
        nom.loc[~nom.ring_period_lock, "delta_raw_ns"], "ns",
        f"{int((~nom.ring_period_lock).sum())} of {len(nom)} shots")
    add("Rogowski coil self-resonance", nom.coil_resonance_MHz, "MHz",
        "dominant post-pinch ringing frequency")

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "summary_stats.csv"),
                              index=False)

    print(f"\nDone in {time.time()-t0:.1f} s. Results in {RESULTS}")
    with pd.option_context("display.width", 200,
                           "display.max_columns", 20,
                           "display.float_format", lambda v: f"{v:,.4g}"):
        print(pd.DataFrame(rows)[["quantity", "unit", "n", "mean", "sd",
                                  "peak_to_peak"]].to_string(index=False))


if __name__ == "__main__":
    main()
