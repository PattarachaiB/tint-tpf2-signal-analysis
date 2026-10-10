#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNR_Analysis.py
===============
Batch computation of the 5 V-referenced interference ratio ("SNR") at the
microcontroller analog inputs V_A0 and V_A1, over a fixed number of screened
shots per experimental day, with shot-to-shot repeatability statistics.

Implements the definition given in Supplementary Information S6 of

    "Integrated Control and Power System Engineering for Thailand Plasma Focus-2"

        SNR = 10 * log10( V_ref^2 / <V_AC^2> )

    V_ref     = 5 V, the nominal full-scale reference of the ATmega2560 analog
                input (a fixed nominal reference, NOT a measured signal)
    <V_AC^2>  = mean square of the record after removal of its mean, i.e. the
                variance of the record. By Parseval's theorem this equals the
                sum of |X_k/N|^2 over all non-zero DFT bins, so it is computed
                here directly in the time domain.

    No noise-floor threshold is applied: the denominator is the total AC mean
    square over the whole record and the whole digitised band, and therefore
    includes the pinch transient itself.

Shot screening
--------------
Every record is screened before it enters the statistics. A shot is rejected,
with the reason recorded in the per-shot CSV, when:

  clipped   the waveform is flat-topped against the vertical range selected at
            acquisition time: >= --clip-samples samples sit exactly at the
            extreme value AND that extreme exceeds --clip-frac of the nominal
            full scale (5 divisions x Vertical Scale). Such a record has had
            its peak energy truncated, so its variance - and therefore its
            SNR - is not valid. This reflects only the y-axis setting used on
            that shot, not any limit of the instrument.
  config    the probe attenuation string differs from --expect-atten, i.e. the
            channel was not acquired in the reference probe configuration
  length    the record length differs from --expect-length, so the acquisition
            window is not comparable with the other shots
  short     fewer than 1000 usable samples in the analysis window

With --select N the script then takes the first N surviving shots of each day,
in filename order, so every day contributes the same sample size. A day with
fewer than N surviving shots is skipped entirely and reported.

Input
-----
Tektronix MSO44 "DataLog_XXXX_Analog.csv" exports. On Scope1 the reference
channel assignment used in this work is

    CH1   Rogowski coil output (dI/dt proportional), 1X
    CH2   Diagnostic-ground current I_DG via Pearson 2877, 1X
    CH3   V_A0, microcontroller analog input, 10X probe
    CH4   V_A1, microcontroller analog input, 10X probe
    MATH1 integrated discharge current

The exporter already applies the probe attenuation, so the CSV columns are in
volts at the probe tip and no further scaling is applied here.

Output
------
  <outdir>/snr_per_shot.csv   one row per record, including rejected ones and
                              the reason for rejection
  <outdir>/snr_summary.csv    one row per (day, channel): n, mean, s.d., SEM,
                              min, max, median, mean Vpp, acquisition config
  stdout                      the summary table and the screening report

Usage
-----
  # exactly 20 screened shots from each day, common 20 us window
  python SNR_Analysis.py --root "." --select 20 --window=-2e-6,18e-6 --jobs 8

  # one day, keep every shot, no screening
  python SNR_Analysis.py --root "." --day "DAY 4-10-2025" --no-screen

  # accumulate several runs into one table
  python SNR_Analysis.py --root "." --day "DAY 4-9-2025" --select 20 --append

Author: prepared for the TPF-2 revision, 2026-09-21
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

V_REF = 5.0                      # V, nominal analog-input full scale
DEFAULT_CHANNELS = ("CH3", "CH4")
CHANNEL_ALIASES = {"CH3": "V_A0", "CH4": "V_A1"}
MAX_HEADER_LINES = 60
DIVISIONS_HALF = 5.0             # Tektronix: +/- 5 divisions about centre


# --------------------------------------------------------------------------- #
#  File parsing
# --------------------------------------------------------------------------- #
def parse_header(path):
    """Read the Tektronix header block. Returns (data_start, columns, meta)."""
    meta = {}
    with open(path, "r", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i > MAX_HEADER_LINES:
                break
            cells = [c.strip().strip('"') for c in line.rstrip("\n").split(",")]
            if not cells:
                continue
            key = cells[0]
            if key == "TIME":
                return i + 1, cells, meta
            if key in ("Probe Attenuation", "Vertical Units", "Vertical Scale",
                       "Sample Interval", "Record Length", "Horizontal Units"):
                meta[key] = cells[1:]
    raise ValueError("no TIME header row found in %s" % path)


def _vscale(meta, col_index):
    """Vertical scale in V/div for a CSV column index, or None."""
    vs = meta.get("Vertical Scale")
    if not vs or col_index - 1 >= len(vs):
        return None
    try:
        return float(vs[col_index - 1])
    except (TypeError, ValueError):
        return None


def shot_snr(task):
    """Compute the 5 V-referenced ratio for one file, with screening."""
    (path, channels, window, screen, expect_atten, expect_length,
     clip_samples, clip_frac) = task

    try:
        start, cols, meta = parse_header(path)
    except Exception as exc:                                   # noqa: BLE001
        return {"file": path, "reject": "header", "detail": str(exc)}

    for ch in channels:
        if ch not in cols:
            return {"file": path, "reject": "config",
                    "detail": "column %s absent" % ch}

    atten = ",".join(meta.get("Probe Attenuation", []))
    length = ",".join(meta.get("Record Length", []))[:6]

    out = {"file": path, "atten": atten,
           "units": ",".join(meta.get("Vertical Units", [])),
           "vscale": ",".join(meta.get("Vertical Scale", [])),
           "reject": "", "detail": ""}

    if screen and expect_atten and atten.rstrip(",") != expect_atten.rstrip(","):
        out.update(reject="config", detail="probe attenuation %s" % atten)
        return out
    if screen and expect_length and length and length != str(expect_length):
        out.update(reject="length", detail="record length %s" % length)
        return out

    usecols = [cols.index("TIME")] + [cols.index(ch) for ch in channels]
    try:
        data = np.genfromtxt(path, delimiter=",", skip_header=start,
                             usecols=usecols, invalid_raise=False)
    except Exception as exc:                                   # noqa: BLE001
        return {"file": path, "reject": "read", "detail": str(exc)}

    if data.ndim != 2 or data.shape[0] < 1000:
        out.update(reject="short", detail="fewer than 1000 rows")
        return out

    t = data[:, 0]
    keep = np.isfinite(t)
    if window is not None:
        keep &= (t >= window[0]) & (t <= window[1])
    if keep.sum() < 1000:
        out.update(reject="short", detail="fewer than 1000 samples in window")
        return out

    out["n_samples"] = int(keep.sum())
    out["t_start"] = float(t[keep].min())
    out["t_end"] = float(t[keep].max())

    for k, ch in enumerate(channels, start=1):
        v = data[keep, k]
        v = v[np.isfinite(v)]
        if v.size < 1000:
            out.update(reject="short", detail="%s has too few samples" % ch)
            return out

        hi, lo = float(v.max()), float(v.min())
        if screen:
            fs = _vscale(meta, cols.index(ch))
            if fs:
                fs_v = DIVISIONS_HALF * fs
                n_hi = int((v >= hi - 1e-9).sum())
                n_lo = int((v <= lo + 1e-9).sum())
                if ((n_hi >= clip_samples and abs(hi) >= clip_frac * fs_v) or
                        (n_lo >= clip_samples and abs(lo) >= clip_frac * fs_v)):
                    out.update(reject="clipped",
                               detail="%s flat-topped at %+.1f/%.1f V "
                                      "(range +/-%.0f V, %d/%d samples at rail)"
                                      % (ch, hi, lo, fs_v, n_hi, n_lo))
                    return out

        ac = v - v.mean()
        ms = float((ac ** 2).mean())            # = variance of the record
        out[ch] = 10.0 * math.log10(V_REF ** 2 / ms) if ms > 0 else float("nan")
        out[ch + "_vrms"] = math.sqrt(ms)
        out[ch + "_vpp"] = hi - lo
    return out


# --------------------------------------------------------------------------- #
#  Grouping and statistics
# --------------------------------------------------------------------------- #
def collect_files(root, day, scope):
    pattern = os.path.join(root, day or "DAY*", "*", scope, "*_Analog.csv")
    groups = {}
    for f in sorted(glob.glob(pattern)):
        rel = os.path.relpath(f, root)
        groups.setdefault(rel.split(os.sep)[0], []).append(f)
    return groups


def summarise(values):
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = a.size
    if n == 0:
        return dict(n=0)
    sd = float(a.std(ddof=1)) if n > 1 else float("nan")
    return dict(n=n, mean=float(a.mean()), sd=sd,
                sem=sd / math.sqrt(n) if n > 1 else float("nan"),
                vmin=float(a.min()), vmax=float(a.max()),
                median=float(np.median(a)))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".",
                    help="folder containing the DAY* directories (default: .)")
    ap.add_argument("--day", default=None,
                    help="restrict to one DAY* folder, e.g. 'DAY 4-10-2025'")
    ap.add_argument("--scope", default="",
                    help="optional subfolder inside each ExNN folder (default: none; CSVs sit directly in DAY*/ExNN/)")
    ap.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS),
                    help="channel columns to analyse (default: CH3 CH4)")
    ap.add_argument("--select", type=int, default=0,
                    help="use exactly N screened shots per day, in filename "
                         "order; a day with fewer is skipped (0 = use all)")
    ap.add_argument("--window", default=None, metavar="T0,T1",
                    help="analysis window in seconds as one comma-separated "
                         "pair, e.g. --window=-2e-6,18e-6 (default: whole "
                         "record, as in S6)")
    ap.add_argument("--no-screen", dest="screen", action="store_false",
                    help="disable shot screening entirely")
    ap.add_argument("--expect-atten", default="1X,1X,10X,10X",
                    help="reference probe attenuation string; shots that "
                         "differ are rejected (default: 1X,1X,10X,10X)")
    ap.add_argument("--expect-length", type=int, default=125000,
                    help="reference record length; shots that differ are "
                         "rejected (default: 125000, 0 = do not check)")
    ap.add_argument("--clip-samples", type=int, default=5,
                    help="samples sitting exactly at the extreme value before "
                         "a record counts as flat-topped (default: 5)")
    ap.add_argument("--clip-frac", type=float, default=0.95,
                    help="fraction of nominal full scale above which a flat "
                         "top is treated as clipping (default: 0.95)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (default: 1)")
    ap.add_argument("--outdir", default="snr_results",
                    help="output folder (default: snr_results)")
    ap.add_argument("--append", action="store_true",
                    help="append to existing CSVs instead of overwriting")
    args = ap.parse_args(argv)

    if args.window:
        try:
            t0, t1 = (float(x) for x in args.window.replace(" ", "").split(","))
        except ValueError:
            sys.exit("--window expects two comma-separated numbers, "
                     "e.g. --window=-2e-6,18e-6")
        window = (t0, t1)
    else:
        window = None

    groups = collect_files(args.root, args.day, args.scope)
    if not groups:
        sys.exit("no *_Analog.csv found under %s (scope=%s)"
                 % (args.root, args.scope))

    os.makedirs(args.outdir, exist_ok=True)

    tasks, owner = [], []
    for g, files in groups.items():
        for f in files:
            tasks.append((f, tuple(args.channels), window, args.screen,
                          args.expect_atten, args.expect_length,
                          args.clip_samples, args.clip_frac))
            owner.append(g)

    print("SNR_Analysis.py  |  V_ref = %.1f V  |  definition of "
          "Supplementary Information S6" % V_REF)
    print("window    : %s" % ("whole record" if window is None
                              else "%g to %g s" % window))
    print("screening : %s" % ("off" if not args.screen else
                              "atten=%s, length=%s, clipping >= %d samples at "
                              ">= %.0f%% of full scale"
                              % (args.expect_atten, args.expect_length or "any",
                                 args.clip_samples, args.clip_frac * 100)))
    print("selection : %s" % ("all screened shots" if not args.select
                              else "first %d screened shots per day"
                                   % args.select))
    print("scanning %d record(s) in %d day(s)...\n" % (len(tasks), len(groups)))

    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            results = list(ex.map(shot_snr, tasks, chunksize=1))
    else:
        results = [shot_snr(t) for t in tasks]

    # ---- screening bookkeeping -------------------------------------------
    accepted, rejected = {}, {}
    for g, r in zip(owner, results):
        if r.get("reject"):
            rejected.setdefault(g, {}).setdefault(r["reject"], []).append(r)
        else:
            accepted.setdefault(g, []).append(r)

    used, used_files, skipped_days = {}, set(), []
    for g, rs in accepted.items():
        rs.sort(key=lambda r: r["file"])
        if args.select:
            if len(rs) < args.select:
                skipped_days.append((g, len(rs)))
                continue
            used[g] = rs[:args.select]
        else:
            used[g] = rs
        used_files.update(r["file"] for r in used[g])

    # ---- per-shot CSV -----------------------------------------------------
    per_shot = os.path.join(args.outdir, "snr_per_shot.csv")
    fields = (["day", "shot", "status", "reject_reason", "detail",
               "n_samples", "t_start_us", "t_end_us", "atten", "vscale"]
              + [c for ch in args.channels
                 for c in (ch + "_dB", ch + "_Vrms", ch + "_Vpp")]
              + ["file"])
    mode = "a" if (args.append and os.path.exists(per_shot)) else "w"
    with open(per_shot, mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for g, r in zip(owner, results):
            row = {"day": g, "shot": os.path.basename(r["file"]),
                   "status": ("used" if r["file"] in used_files else
                              ("rejected" if r.get("reject")
                               else "not selected")),
                   "reject_reason": r.get("reject", ""),
                   "detail": r.get("detail", ""),
                   "n_samples": r.get("n_samples", ""),
                   "t_start_us": ("%.2f" % (r["t_start"] * 1e6)
                                  if "t_start" in r else ""),
                   "t_end_us": ("%.2f" % (r["t_end"] * 1e6)
                                if "t_end" in r else ""),
                   "atten": r.get("atten", ""), "vscale": r.get("vscale", ""),
                   "file": r["file"]}
            for ch in args.channels:
                row[ch + "_dB"] = ("%.3f" % r[ch]) if ch in r else ""
                row[ch + "_Vrms"] = (("%.4f" % r[ch + "_vrms"])
                                     if ch + "_vrms" in r else "")
                row[ch + "_Vpp"] = (("%.2f" % r[ch + "_vpp"])
                                    if ch + "_vpp" in r else "")
            w.writerow(row)

    # ---- summary ----------------------------------------------------------
    summary_path = os.path.join(args.outdir, "snr_summary.csv")
    header = ("%-18s %-6s %4s %9s %8s %8s %9s %9s %9s %10s"
              % ("day", "chan", "n", "mean(dB)", "s.d.", "SEM", "min", "max",
                 "median", "Vpp mean"))
    print(header)
    print("-" * len(header))
    smode = "a" if (args.append and os.path.exists(summary_path)) else "w"
    with open(summary_path, smode, newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if smode == "w":
            w.writerow(["day", "channel", "label", "n", "mean_dB", "sd_dB",
                        "sem_dB", "min_dB", "max_dB", "median_dB",
                        "Vpp_mean_V", "acquisition_config"])
        for g in sorted(used):
            rs = used[g]
            cfg = "atten=%s N=%s t=[%.1f,%.1f]us" % (
                rs[0].get("atten", ""), rs[0].get("n_samples", ""),
                rs[0].get("t_start", 0) * 1e6, rs[0].get("t_end", 0) * 1e6)
            for ch in args.channels:
                st = summarise([r[ch] for r in rs if ch in r])
                vpp = [r[ch + "_vpp"] for r in rs if ch + "_vpp" in r]
                vppm = sum(vpp) / len(vpp) if vpp else float("nan")
                label = CHANNEL_ALIASES.get(ch, ch)
                if st["n"] == 0:
                    continue
                print("%-18s %-6s %4d %9.2f %8.2f %8.2f %9.2f %9.2f %9.2f %10.1f"
                      % (g, label, st["n"], st["mean"], st["sd"], st["sem"],
                         st["vmin"], st["vmax"], st["median"], vppm))
                w.writerow([g, ch, label, st["n"],
                            round(st["mean"], 3), round(st["sd"], 3),
                            round(st["sem"], 3), round(st["vmin"], 3),
                            round(st["vmax"], 3), round(st["median"], 3),
                            round(vppm, 2), cfg])

    # ---- screening report -------------------------------------------------
    print("\nScreening report")
    print("%-18s %8s %8s %6s  %s"
          % ("day", "records", "screened", "used", "rejected (reason x n)"))
    for g in sorted(groups):
        n_all = len(groups[g])
        n_ok = len(accepted.get(g, []))
        n_used = len(used.get(g, []))
        rej = rejected.get(g, {})
        detail = ", ".join("%s x%d" % (k, len(v)) for k, v in sorted(rej.items()))
        print("%-18s %8d %8d %6s  %s"
              % (g, n_all, n_ok, n_used if n_used else "-", detail or "none"))
    if skipped_days:
        print("\nDays skipped (fewer than %d screened shots):" % args.select)
        for g, n in skipped_days:
            print("   %-18s only %d screened shot(s)" % (g, n))

    print("\nManuscript form, mean +/- sample s.d. over n shots:")
    for g in sorted(used):
        for ch in args.channels:
            st = summarise([r[ch] for r in used[g] if ch in r])
            if st["n"] > 1:
                print("   %-18s %-6s %6.2f +/- %.2f dB  (n = %d)"
                      % (g, CHANNEL_ALIASES.get(ch, ch), st["mean"], st["sd"],
                         st["n"]))

    print("\nwrote %s" % per_shot)
    print("wrote %s" % summary_path)
    print("\nNOTE: the +/- figure is the sample standard deviation of the "
          "per-shot values\n      (shot-to-shot repeatability), not a "
          "measurement uncertainty. Day names are\n      folder names; map them "
          "to the pre- and post-modification configurations\n      from the "
          "laboratory record before quoting any number in the manuscript.")


if __name__ == "__main__":
    main()
