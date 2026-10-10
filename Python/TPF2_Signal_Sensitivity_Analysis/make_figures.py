"""
make_figures.py
===============
Publication-quality figures for the H3 robustness check.

Every figure is drawn in Times New Roman (see ``figstyle.py``) at 160 mm width,
i.e. exactly the width at which it is placed in the manuscript, so that Word
never rescales it and the type sizes on the printed page are the ones set here.
Sub-panels are marked a), b), c) … at the top-left, and all explanatory text
lives in the caption rather than in the figure, as required by the journal.

Outputs (PNG 600 dpi + vector PDF) in ./figures/ :

  FigS1_pipeline_overview      a-d  nominal chain, representative shot
  FigS2_heatmap_p_Wpulse       a-c  deviation over p x W_pulse
  FigS3_clipping_threshold     a-c  dependence on the clipping threshold
  FigS4_independent_check      a-c  processed vs unfiltered pinch time
  FigS5_tornado                a-b  contribution of each parameter
  FigS6_waveform_overlay       a-b  pinch region under extreme parameter sets
  FigS7_shot_to_shot           a-c  repeatability at nominal parameters

Usage:  python make_figures.py [--data ../PD_Ex12.txt] [--shot 12]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs                                       # noqa: E402
from figstyle import (FIG_W, BLUE, RED, ORANGE, GREEN, PURPLE, TEAL,  # noqa: E402
                      GREY, LIGHT, CMAP_DIV, FS_ANNOT, FS_LEGEND, panel_auto)
from pipeline import (                                      # noqa: E402
    BASELINE, load_all_shots, trim_and_time_axis, pre_stage,
    multi_segment_filter, clip_positive, integrate_kA, extract_features,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")

fs.apply()
save = lambda fig, name: fs.save(fig, name, FIGS)


# ==========================================================================
def fig_overview(t, data, pre, shot_no):
    """FigS1 - the nominal chain: raw, detection, filtered, integrated."""
    filt = multi_segment_filter(data, pre, BASELINE["sg_window_pulse"],
                                BASELINE["sg_window_flank"],
                                BASELINE["sg_polyorder"])
    clipped, upper, lower = clip_positive(filt, pre, BASELINE["clip_pct"])
    integ = integrate_kA(t, clipped)
    feat = extract_features(t, clipped, integ, pre)

    xlim = (-0.5, 6.0)
    m = (t >= xlim[0]) & (t <= xlim[1])

    fig, ax = plt.subplots(4, 1, figsize=(FIG_W, 6.9), sharex=True)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.965, bottom=0.062,
                        hspace=0.42)

    # ---- a) raw ---------------------------------------------------------
    ax[0].plot(t[m], data[m], color=BLUE, lw=0.5)
    ax[0].set_ylabel(r"$V_{\mathrm{coil}}$  (V)")
    ax[0].set_ylim(-3.6, 3.6)

    # ---- b) baseline + detection ---------------------------------------
    ax[1].plot(t[m], pre.smoothed[m], color=GREEN, lw=0.7,
               label=r"pre-smoothed ($W$ = 51, $p$ = 3)")
    ax[1].plot(pre.t_win, pre.baseline_win, color=ORANGE, lw=1.6,
               label=u"fitted baseline (2–4 µs)")
    off_i = BASELINE["pct_threshold_inner"] / 100 * abs(pre.baseline_win.mean())
    off_o = BASELINE["pct_threshold_outer"] / 100 * abs(pre.baseline_win.mean())
    ax[1].plot(pre.t_win, pre.baseline_win - off_i, "--", color=PURPLE, lw=1.0,
               label=r"inner threshold ($-$50 %)")
    ax[1].plot(pre.t_win, pre.baseline_win - off_o, ":", color=PURPLE, lw=1.2,
               label=r"outer threshold ($-$70 %)")
    if pre.pulse_start is not None:
        ax[1].axvspan(pre.pulse_start, pre.pulse_end, color="#FFD24D",
                      alpha=0.65, lw=0,
                      label=f"detected pulse ({feat['pulse_duration_ns']:.0f} ns)")
    ax[1].set_ylabel("pre-smoothed  (V)")
    ax[1].set_ylim(-4.7, 3.6)
    # Opaque backing: the detected-pulse span runs the full height of the
    # panel and would otherwise strike through the legend text.
    ax[1].legend(loc="lower left", ncol=2, fontsize=FS_LEGEND,
                 frameon=True, facecolor="white", edgecolor="none",
                 framealpha=0.92)

    # ---- c) filtered + clipped -----------------------------------------
    ax[2].plot(t[m], clipped[m], color=TEAL, lw=0.75,
               label="multi-segment SG filter + one-sided clip")
    ax[2].plot(pre.t_win, upper, color=PURPLE, lw=1.2,
               label="clipping bound (+40 %)")
    ax[2].plot(pre.t_win, lower, color=PURPLE, lw=1.2, ls=":",
               label="mirror bound (never applied)")
    ax[2].set_ylabel("processed  (V)")
    ax[2].set_ylim(-4.7, 3.6)
    ax[2].legend(loc="lower left", ncol=2, fontsize=FS_LEGEND,
                 frameon=True, facecolor="white", edgecolor="none",
                 framealpha=0.92)

    # ---- d) integrated current -----------------------------------------
    ax[3].plot(t[m], integ[m], color=RED, lw=1.4)
    ax[3].axvline(feat["t_pinch_us"], color=GREY, ls="--", lw=0.9)
    ax[3].annotate(
        f"pinch: $t$ = {feat['t_pinch_us']:.3f} " u"µs\n"
        f"FWHM = {feat['pinch_fwhm_ns']:.1f} ns\n"
        f"$I_{{\\mathrm{{peak}}}}$ = {feat['I_peak_kA']:.1f} kA",
        xy=(feat["t_pinch_us"], feat["I_at_pinch_kA"]),
        xytext=(0.62, 0.22), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.9, color=GREY),
        fontsize=FS_ANNOT)
    ax[3].set_ylabel(r"$I$  (kA)")
    ax[3].set_xlabel(u"time  (µs)")
    ax[3].set_xlim(xlim)
    ax[3].set_ylim(-8, 145)
    panel_auto(fig, ax, "abcd")
    save(fig, "FigS1_pipeline_overview")
    return feat


# ==========================================================================
def fig_heatmap(dfA):
    """FigS2 - deviation from nominal over p x W_pulse."""
    nom = dfA[(dfA.sg_polyorder == BASELINE["sg_polyorder"]) &
              (dfA.sg_window_pulse == BASELINE["sg_window_pulse"]) &
              (dfA.clip_pct == BASELINE["clip_pct"])
              ][["shot", "t_pinch_us", "I_peak_kA", "pinch_fwhm_ns"]]
    nom = nom.rename(columns={"t_pinch_us": "t0", "I_peak_kA": "I0",
                              "pinch_fwhm_ns": "F0"})
    d = dfA[dfA.clip_pct == BASELINE["clip_pct"]].merge(nom, on="shot")
    d["dt_ns"] = (d.t_pinch_us - d.t0) * 1e3
    d["dI_pct"] = 100 * (d.I_peak_kA - d.I0) / d.I0
    d["dF_ns"] = d.pinch_fwhm_ns - d.F0

    specs = [("dt_ns", r"$\Delta t_{\mathrm{pinch}}$  (ns)", "a"),
             ("dI_pct", r"$\Delta I_{\mathrm{peak}}$  (%)", "b"),
             ("dF_ns", r"$\Delta$FWHM  (ns)", "c")]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.45))
    fig.subplots_adjust(left=0.082, right=0.925, top=0.84, bottom=0.20,
                        wspace=0.52)

    for k, (ax, (col, lab, ltr)) in enumerate(zip(axes, specs)):
        piv = d.pivot_table(index="sg_polyorder", columns="sg_window_pulse",
                            values=col, aggfunc="mean")
        v = np.nanmax(np.abs(piv.values)) or 1.0
        im = ax.imshow(piv.values, cmap=CMAP_DIV, vmin=-v, vmax=v,
                       aspect="auto", origin="lower")
        ax.set_xticks(range(piv.shape[1]), piv.columns)
        ax.set_yticks(range(piv.shape[0]), piv.index)
        ax.set_xlabel(r"$W_{\mathrm{pulse}}$  (samples)")
        if k == 0:
            ax.set_ylabel(r"polynomial order  $p$")
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                val = piv.values[i, j]
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=6.3,
                        color="white" if abs(val) > 0.62 * v else "black")
        for sp in ax.spines.values():
            sp.set_visible(True)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.035)
        cb.ax.tick_params(labelsize=7.5, length=2)
        cb.set_label(lab, fontsize=8.8, labelpad=2)

    panel_auto(fig, axes, "abc")
    save(fig, "FigS2_heatmap_p_Wpulse")


# ==========================================================================
def fig_clipping(dfA):
    """FigS3 - dependence on the amplitude-clipping threshold."""
    d = dfA[(dfA.sg_polyorder == BASELINE["sg_polyorder"]) &
            (dfA.sg_window_pulse == BASELINE["sg_window_pulse"])]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.65))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.86, bottom=0.30,
                        wspace=0.42)

    specs = [("t_pinch_us", r"$t_{\mathrm{pinch}}$  (" u"µs)", "a"),
             ("I_peak_kA", r"$I_{\mathrm{peak}}$  (kA)", "b"),
             ("pinch_fwhm_ns", "pinch FWHM  (ns)", "c")]

    for ax, (col, lab, ltr) in zip(axes, specs):
        for _, g in d.groupby("shot"):
            g = g.sort_values("clip_pct")
            ax.plot(g.clip_pct, g[col], color=GREY, lw=0.55, alpha=0.35)
        agg = d.groupby("clip_pct")[col].agg(["mean", "std"])
        ax.errorbar(agg.index, agg["mean"], yerr=agg["std"], color=RED,
                    marker="o", ms=4.2, lw=1.5, capsize=3, zorder=5)
        ax.axvline(BASELINE["clip_pct"], color=BLUE, ls="--", lw=1.1)
        ax.set_ylabel(lab)
        ax.set_xticks(sorted(d.clip_pct.unique()))
        ax.set_xlim(15, 65)

    fig.supxlabel("positive clipping threshold  (%)", fontsize=10, y=0.135)

    handles = [Line2D([], [], color=GREY, lw=0.8, alpha=0.6),
               Line2D([], [], color=RED, marker="o", ms=4.2, lw=1.5),
               Line2D([], [], color=BLUE, ls="--", lw=1.1)]
    fig.legend(handles,
               [f"individual shots ($N$ = {d.shot.nunique()})",
                r"mean $\pm$ 1 s.d.", "value used in this work (40 %)"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.012),
               fontsize=FS_LEGEND)

    panel_auto(fig, axes, "abc")
    save(fig, "FigS3_clipping_threshold")


# ==========================================================================
def fig_independent(nom):
    """FigS4 - processed vs filter-free pinch time."""
    lock = nom.ring_period_lock.astype(bool)
    ok = ~lock

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.70))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.21,
                        wspace=0.44)

    # ---- a) 1:1 ---------------------------------------------------------
    ax = axes[0]
    lo = min(nom.t_pinch_raw_us.min(), nom.t_pinch_us.min())
    hi = max(nom.t_pinch_raw_us.max(), nom.t_pinch_us.max())
    pad = 0.06 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, lw=0.9,
            ls="--", zorder=1, label="1:1")
    ax.scatter(nom.t_pinch_raw_us[ok], nom.t_pinch_us[ok], s=26, color=BLUE,
               edgecolor="white", linewidth=0.5, zorder=3,
               label=f"agreement ($N$ = {int(ok.sum())})")
    ax.scatter(nom.t_pinch_raw_us[lock], nom.t_pinch_us[lock], s=40,
               facecolor="none", edgecolor=RED, linewidth=1.4, zorder=4,
               label=f"ring-locked ($N$ = {int(lock.sum())})")
    ax.set_xlabel(r"$t_{\mathrm{pinch}}$, unfiltered  (" u"µs)")
    ax.set_ylabel(r"$t_{\mathrm{pinch}}$, processed  (" u"µs)")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.legend(loc="upper left", fontsize=7.8)

    # ---- b) residual histogram -----------------------------------------
    ax = axes[1]
    d_ok = nom.delta_raw_ns[ok]
    mu, sd = d_ok.mean(), d_ok.std(ddof=1)
    ax.hist(d_ok, bins=12, color=BLUE, alpha=0.9, edgecolor="white", lw=0.6)
    ax.axvline(0, color=GREY, lw=1.0, ls="--")
    ax.axvline(mu, color=RED, lw=1.4)
    ax.set_xlabel(r"$\Delta t_{\mathrm{pinch}}$  (ns)")
    ax.set_ylabel("number of shots")
    ax.text(0.04, 0.95,
            f"$N$ = {int(ok.sum())}\n"
            f"mean = {mu:+.2f} ns\n"
            f"s.d. = {sd:.2f} ns\n"
            f"max = {d_ok.abs().max():.2f} ns",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_ANNOT,
            linespacing=1.35)
    ax.margins(y=0.30)

    # ---- c) per shot ----------------------------------------------------
    ax = axes[2]
    x = np.arange(len(nom))
    period = float(np.nanmedian(nom.coil_ring_period_ns))
    ax.bar(x - 0.21, nom.delta_raw_ns, width=0.42, color=BLUE,
           label="vs unfiltered")
    ax.bar(x + 0.21, nom.delta_raw_med5_ns, width=0.42, color=ORANGE,
           label="vs 5-point median")
    ax.axhline(0, color="black", lw=0.8)
    # The reference level is identified in the legend rather than by a label
    # inside the axes: at this aspect ratio any in-plot label sits on top of
    # either the dashed line itself or the three tall bars.
    ax.axhline(-period, color=RED, ls="--", lw=1.2,
               label=f"one coil-ring period ({period:.1f} ns)")
    ax.set_xticks(x[::3], nom.shot.astype(int).to_numpy()[::3])
    ax.set_xlabel("shot number")
    ax.set_ylabel(r"$\Delta t_{\mathrm{pinch}}$  (ns)")
    ax.set_ylim(-55, 14)
    ax.legend(loc="lower left", fontsize=7.2, ncol=1,
              borderaxespad=0.15, handlelength=1.5)

    panel_auto(fig, axes, "abc")
    save(fig, "FigS4_independent_check")


# ==========================================================================
def fig_tornado(dfA, dfB, dfC, nom):
    """FigS5 - each parameter's contribution, against the physical spread.

    Laid out as two stacked full-width panels: the parameter names are long,
    and side-by-side panels would squeeze the plotting area to nothing.
    """
    def spread(df, col, scale, rel=False):
        g = df.groupby("shot")[col]
        if rel:
            s = g.apply(lambda x: 100 * (x.max() - x.min()) / x.mean())
        else:
            s = g.apply(lambda x: (x.max() - x.min()) * scale)
        return s.mean(), s.max()

    d_p = dfA[(dfA.sg_window_pulse == BASELINE["sg_window_pulse"]) &
              (dfA.clip_pct == BASELINE["clip_pct"])]
    d_w = dfA[(dfA.sg_polyorder == BASELINE["sg_polyorder"]) &
              (dfA.clip_pct == BASELINE["clip_pct"])]
    d_c = dfA[(dfA.sg_polyorder == BASELINE["sg_polyorder"]) &
              (dfA.sg_window_pulse == BASELINE["sg_window_pulse"])]

    items = [
        (r"polynomial order $p$  (2$-$4)", d_p),
        (r"$W_{\mathrm{pulse}}$  (5$-$21 samples)", d_w),
        (r"clipping threshold  (20$-$60 %)", d_c),
        (r"$W_{\mathrm{flank}}$  (501$-$1501 samples)", dfB),
        ("detection thresholds  (40/60$-$60/80 %)",
         dfC[dfC.get("detected", True) == True]),            # noqa: E712
        ("all processing parameters jointly", dfA),
    ]

    FLOOR = 1e-3

    # Generous hspace: each panel carries its own legend above it, so the two
    # graphs need clear separation or the legend of the lower panel reads as
    # part of the upper one.
    # Wide left margin: the parameter names are long and the canvas is a fixed
    # size, so the margin has to be reserved rather than grown by a tight bbox.
    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, 5.4))
    fig.subplots_adjust(left=0.375, right=0.925, top=0.88, bottom=0.09,
                        hspace=0.95)

    cfg = [("t_pinch_us", 1e3, False,
            r"peak-to-peak spread of $t_{\mathrm{pinch}}$  (ns)", "ns", "a"),
           ("I_peak_kA", 1.0, True,
            r"peak-to-peak spread of $I_{\mathrm{peak}}$  (%)", "%", "b")]

    for ax, (col, scale, rel, lab, unit, ltr) in zip(axes, cfg):
        names, means, maxs = [], [], []
        for nm, df in items:
            if df.empty or col not in df:
                continue
            m, mx = spread(df, col, scale, rel)
            names.append(nm); means.append(m); maxs.append(mx)
        means = np.maximum(np.array(means), FLOOR)
        maxs = np.maximum(np.array(maxs), FLOOR)

        y = np.arange(len(names))
        ax.barh(y, maxs, left=FLOOR, color=LIGHT, height=0.66,
                label="worst individual shot")
        ax.barh(y, means, left=FLOOR, color=BLUE, height=0.66,
                label="mean over shots")
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(lab)

        if col == "t_pinch_us":
            phys = (nom.t_pinch_us.max() - nom.t_pinch_us.min()) * 1e3
        else:
            phys = 100 * (nom.I_peak_kA.max() -
                          nom.I_peak_kA.min()) / nom.I_peak_kA.mean()

        ax.set_xlim(FLOOR, phys * 5)
        for yi, (m, mx) in enumerate(zip(means, maxs)):
            tiny = mx <= FLOOR * 10
            txt = "< 0.01" if tiny else f"{m:.2f} / {mx:.2f}"
            ax.text(FLOOR * 4 if tiny else mx * 1.5, yi, txt, va="center",
                    fontsize=7.8, color=GREY)

        # the reference line carries its own label, set vertically alongside
        # it so that it cannot collide with the bars or the tick labels
        # Named in the legend, not with a rotated in-plot label: on a log axis
        # the label has to sit next to the line, which is exactly where the
        # numeric bar annotations are.
        phys_txt = f"{phys:.0f}" if unit == "ns" else f"{phys:.2f}"
        ax.axvline(phys, color=RED, ls="--", lw=1.3,
                   label=f"physical shot-to-shot spread ({phys_txt} {unit})")
        ax.grid(axis="x", which="major", ls=":", lw=0.6, color="#D0D0D0")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=2,
                  fontsize=FS_LEGEND, columnspacing=1.6)

    panel_auto(fig, axes, "ab")
    save(fig, "FigS5_tornado")


# ==========================================================================
def fig_overlay(t, data, pre, shot_no):
    """FigS6 - the pinch region under the extreme parameter combinations."""
    combos = [
        (2, 5, 20, BLUE), (2, 21, 60, PURPLE), (3, 11, 40, "#000000"),
        (4, 5, 60, ORANGE), (4, 21, 20, GREEN),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.85))
    fig.subplots_adjust(left=0.095, right=0.935, top=0.90, bottom=0.40,
                        wspace=0.34)

    tp = None
    handles = []
    for poly, wp, cl, c in combos:
        filt = multi_segment_filter(data, pre, wp,
                                    BASELINE["sg_window_flank"], poly)
        clipped, _, _ = clip_positive(filt, pre, cl)
        integ = integrate_kA(t, clipped)
        f = extract_features(t, clipped, integ, pre)
        nominal = (poly, wp, cl) == (3, 11, 40)
        if nominal:
            tp = f["t_pinch_us"]
        lab = (f"$p$ = {poly}, $W_{{\\mathrm{{pulse}}}}$ = {wp}, "
               f"clip = {cl} %" + ("  (nominal)" if nominal else ""))
        lw = 1.8 if nominal else 1.0
        ln, = axes[0].plot(t, clipped, color=c, lw=lw, alpha=0.92, label=lab)
        handles.append(ln)
        axes[1].plot(t, integ, color=c, lw=lw, alpha=0.92)

    win = 0.25
    mm = (t > tp - win) & (t < tp + win)

    raw, = axes[0].plot(t, data, color="#BFBFBF", lw=0.45, zorder=0,
                        label="unfiltered d$I$/d$t$")
    axes[0].set_xlim(tp - win, tp + win)
    axes[0].set_ylim(np.min(data[mm]) * 1.10, np.max(data[mm]) * 1.10)
    axes[0].set_xlabel(u"time  (µs)")
    axes[0].set_ylabel(r"d$I$/d$t$ signal  (V)")
    axes[0].axvline(tp, color=GREY, ls=":", lw=0.9)

    lines_b = [ln.get_ydata()[mm] for ln in axes[1].get_lines()]
    ylo = min(np.min(y) for y in lines_b)
    yhi = max(np.max(y) for y in lines_b)
    pad = max(0.06 * (yhi - ylo), 0.05)
    axes[1].set_xlim(tp - win, tp + win)
    axes[1].set_ylim(ylo - pad, yhi + pad)
    axes[1].set_xlabel(u"time  (µs)")
    axes[1].set_ylabel(r"$I$  (kA)")
    axes[1].axvline(tp, color=GREY, ls=":", lw=0.9)

    fig.legend(handles + [raw], [h.get_label() for h in handles] +
               [raw.get_label()],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.012),
               fontsize=FS_LEGEND)

    panel_auto(fig, axes, "ab")
    save(fig, "FigS6_waveform_overlay")


# ==========================================================================
def fig_shot_to_shot(nom):
    """FigS7 - repeatability at nominal parameters (revision item H1)."""
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 2.60))
    fig.subplots_adjust(left=0.095, right=0.99, top=0.88, bottom=0.24,
                        wspace=0.44)

    specs = [("t_pinch_us", r"$t_{\mathrm{pinch}}$  (" u"µs)", "{:.3f}", "a"),
             ("I_peak_kA", r"$I_{\mathrm{peak}}$  (kA)", "{:.1f}", "b"),
             ("pinch_fwhm_ns", "pinch FWHM  (ns)", "{:.1f}", "c")]

    from scipy import stats

    for ax, (col, lab, fmt, ltr) in zip(axes, specs):
        v = nom[col].to_numpy()
        s = nom.shot.to_numpy()
        mu, sd = np.nanmean(v), np.nanstd(v, ddof=1)
        ax.axhspan(mu - sd, mu + sd, color=RED, alpha=0.13, lw=0)
        ax.axhline(mu, color=RED, lw=1.2)
        ax.plot(s, v, "o-", ms=4.0, color=BLUE, lw=1.0)

        # A monotonic drift through a shot sequence is not random scatter, and
        # quoting a standard deviation for it would overstate the random
        # repeatability.  Where the trend is significant, show it and quote the
        # residual spread about it as well.
        lr = stats.linregress(s, v)
        note = (f"{fmt.format(mu)} " r"$\pm$ " f"{fmt.format(sd)}"
                f"  ($N$ = {int(np.isfinite(v).sum())})")
        if lr.pvalue < 0.01 and lr.rvalue ** 2 > 0.4:
            ax.plot(s, lr.intercept + lr.slope * s, ls="--", color=GREEN,
                    lw=1.3, zorder=4)
            resid = v - (lr.intercept + lr.slope * s)
            unit = "ns" if "FWHM" in lab else ("kA" if "peak" in col else "ns")
            drift = lr.slope * (1e3 if col == "t_pinch_us" else 1.0)
            note += (f"\ntrend {drift:+.2f} {unit}/shot "
                     f"($R^2$ = {lr.rvalue**2:.2f})"
                     f"\nresidual s.d. "
                     f"{(resid.std(ddof=2) * (1e3 if col == 't_pinch_us' else 1)):.1f}"
                     f" {unit}")
        ax.set_xlabel("shot number")
        ax.set_ylabel(lab)
        ax.set_xticks(nom.shot[::4])
        ax.text(0.96, 0.04, note, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.6, linespacing=1.35)
        ax.margins(y=0.30)

    panel_auto(fig, axes, "abc")
    save(fig, "FigS7_shot_to_shot")


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "PD_Ex12.txt"))
    ap.add_argument("--shot", type=int, default=12)
    args = ap.parse_args()

    fs.report_font()

    dfA = pd.read_csv(os.path.join(RESULTS, "sweep_gridA.csv"))
    dfB = pd.read_csv(os.path.join(RESULTS, "sweep_gridB.csv"))
    dfC = pd.read_csv(os.path.join(RESULTS, "sweep_gridC.csv"))
    nom = pd.read_csv(os.path.join(RESULTS, "nominal_per_shot.csv"))

    shots = load_all_shots(os.path.abspath(args.data))
    t, data = trim_and_time_axis(shots[args.shot - 1])
    pre = pre_stage(t, data)

    feat = fig_overview(t, data, pre, args.shot)
    fig_heatmap(dfA)
    fig_clipping(dfA)
    fig_independent(nom)
    fig_tornado(dfA, dfB, dfC, nom)
    fig_overlay(t, data, pre, args.shot)
    fig_shot_to_shot(nom)

    print(f"\nrepresentative shot {args.shot}: "
          f"t_pinch = {feat['t_pinch_us']:.4f} us, "
          f"FWHM = {feat['pinch_fwhm_ns']:.2f} ns, "
          f"I_peak = {feat['I_peak_kA']:.2f} kA")


if __name__ == "__main__":
    main()
