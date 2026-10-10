"""
figstyle.py
===========
Shared publication style for every figure in this analysis.

Typography
----------
Times New Roman is requested first.  On machines where it is not installed
(most Linux systems, including the container this analysis may be run in)
matplotlib falls back to **Liberation Serif**, which is metrically identical
to Times New Roman, and then to Nimbus Roman.  Maths is set with the STIX
font set, which is the Times-compatible companion face, so that symbols such
as t_pinch and I_peak match the surrounding text instead of switching to the
default sans-serif italic.

Geometry
--------
Figures are drawn at exactly the width at which they are placed in the
manuscript (160 mm = 6.3 in, the full text width of the Word template).  This
means Word inserts them at 1:1 and a 10 pt label in the figure really is
10 pt on the printed page, matching the body text.  Never let Word rescale a
figure: that is what makes axis labels look small and inconsistent.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------
# geometry (inches)
# --------------------------------------------------------------------------
MM = 1 / 25.4
#: 6.3 in — the full text width of the Word template, and exactly the width
#: at which ``build_revision.py`` inserts the figures, so Word never rescales.
FIG_W = 160 * MM
FIG_W_HALF = 80 * MM

# --------------------------------------------------------------------------
# type sizes (points, at 1:1 placement)
# --------------------------------------------------------------------------
FS_BASE = 9.5
FS_LABEL = 10.0
FS_TICK = 9.0
FS_LEGEND = 8.5
FS_PANEL = 11.0             # the a) b) c) markers
FS_ANNOT = 8.5

SERIF_STACK = ["Times New Roman", "Liberation Serif", "Nimbus Roman",
               "DejaVu Serif"]

# --------------------------------------------------------------------------
# palette — colour-blind safe, and legible when printed in greyscale
# --------------------------------------------------------------------------
BLUE = "#1F5C99"
RED = "#B02418"
ORANGE = "#E08214"
GREEN = "#2E7D32"
PURPLE = "#6A3D9A"
TEAL = "#00707C"
GREY = "#5A5A5A"
LIGHT = "#C8DCEB"
CMAP_DIV = "RdBu_r"


def apply():
    """Install the style globally.  Call once at the top of a figure script."""
    plt.rcParams.update({
        # ---- typography -------------------------------------------------
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "stix",
        "font.size": FS_BASE,
        "axes.labelsize": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FS_LABEL,

        # ---- lines and axes ---------------------------------------------
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.3,
        "xtick.major.width": 0.9,
        "ytick.major.width": 0.9,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,

        # ---- legends ------------------------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.labelspacing": 0.35,
        "legend.borderaxespad": 0.3,

        # ---- output -------------------------------------------------------
        "figure.dpi": 120,
        # NOT "tight": a tight bounding box crops to the ink, so each figure
        # would come out a slightly different width and Word would rescale
        # each one by a different factor — which is exactly what makes type
        # sizes inconsistent between figures.  The canvas is laid out by hand
        # instead, so every file is exactly FIG_W wide.
        "savefig.bbox": None,
        "pdf.fonttype": 42,       # embed as TrueType, keep text selectable
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def panel(ax, letter, dx=-0.02, dy=1.015, fontsize=FS_PANEL):
    """Axes-relative panel marker.  Prefer :func:`panel_auto` — see there."""
    return ax.text(dx, dy, f"{letter})", transform=ax.transAxes,
                   fontsize=fontsize, fontweight="bold",
                   ha="right", va="bottom")


def panel_auto(fig, axes, letters, pad_x=0.006, pad_y=0.008,
               fontsize=FS_PANEL):
    """Place a), b), c) … at the true top-left of each panel.

    Axes-relative placement is unreliable: a fixed offset such as ``-0.05``
    lands on top of the y-axis tick labels when the numbers are wide, and
    floats far from the panel when they are narrow.  Here the *tight* bounding
    box of each axes is measured after a trial draw — which includes the tick
    labels and the axis label — and the marker is placed just outside its
    top-left corner.  The result cannot collide with the axis decorations
    whatever the data happen to be.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    out = []
    for ax, letter in zip(axes, letters):
        bb = ax.get_tightbbox(renderer).transformed(inv)
        out.append(fig.text(max(bb.x0 - pad_x, 0.002),
                            min(bb.y1 + pad_y, 0.998),
                            f"{letter})", fontsize=fontsize,
                            fontweight="bold", ha="left", va="bottom"))
    return out


def check_clipping(fig, name, tol=0.002):
    """Warn if any text or legend falls outside the fixed canvas.

    With ``savefig.bbox = None`` the canvas no longer grows to fit stray
    artists, so anything pushed past an edge is silently cropped.  This walks
    the drawn artists and reports the offenders instead of letting a cropped
    legend reach the manuscript.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    bad = []

    def _check(artist, label):
        try:
            bb = artist.get_window_extent(r).transformed(inv)
        except Exception:
            return
        if (bb.x0 < -tol or bb.x1 > 1 + tol
                or bb.y0 < -tol or bb.y1 > 1 + tol):
            bad.append(f"{label} [{bb.x0:.3f}, {bb.x1:.3f}] x "
                       f"[{bb.y0:.3f}, {bb.y1:.3f}]")

    for t in fig.texts:
        _check(t, f"figure text {t.get_text()[:28]!r}")
    if fig.legends:
        for lg in fig.legends:
            _check(lg, "figure legend")
    for i, ax in enumerate(fig.axes):
        for t in ax.texts:
            _check(t, f"ax{i} text {t.get_text()[:28]!r}")
        if ax.get_legend() is not None:
            _check(ax.get_legend(), f"ax{i} legend")
        _check(ax.yaxis.label, f"ax{i} ylabel")
        _check(ax.xaxis.label, f"ax{i} xlabel")
        # Tick labels for values outside the view limits are culled at draw
        # time but still report a window extent, so only consider ticks whose
        # centre sits next to the axes they belong to.
        abox = ax.get_window_extent(r).transformed(inv)
        for lbl in ax.get_yticklabels() + ax.get_xticklabels():
            if not lbl.get_text():
                continue
            try:
                bb = lbl.get_window_extent(r).transformed(inv)
            except Exception:
                continue
            cx, cy = (bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2
            near = (abox.x0 - 0.08 <= cx <= abox.x1 + 0.08 and
                    abox.y0 - 0.08 <= cy <= abox.y1 + 0.08)
            if near:
                _check(lbl, f"ax{i} tick {lbl.get_text()[:14]!r}")

    if bad:
        print(f"   !! {name}: {len(bad)} artist(s) outside the canvas")
        for b in bad[:6]:
            print(f"      - {b}")
    return bad


def save(fig, name, outdir):
    """Write PNG (600 dpi, for Word) and PDF (vector, for the publisher)."""
    import os
    check_clipping(fig, name)
    os.makedirs(outdir, exist_ok=True)
    for ext, dpi in (("png", 600), ("pdf", None)):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"),
                    dpi=dpi if dpi else "figure")
    plt.close(fig)
    print(f"   figures/{name}.png + .pdf")


def report_font():
    """Say which face was actually used, so the output is never a surprise."""
    from matplotlib.font_manager import findfont, FontProperties
    path = findfont(FontProperties(family="serif"))
    import os
    name = os.path.basename(path)
    if "Times" in name or "times" in name:
        note = "genuine Times New Roman"
    elif "Liberation" in name:
        note = "Liberation Serif — metrically identical to Times New Roman"
    elif "Nimbus" in name:
        note = "Nimbus Roman — URW clone of Times"
    else:
        note = "WARNING: not a Times-compatible face"
    print(f"[font] {name}  ({note})")
