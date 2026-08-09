#!/usr/bin/env python
"""The corrected top_p result, as one figure.

One counting rule throughout: spoiled-ballot. Every panel states its temperature arm in the
title, because the two arms are easy to confuse and the answer differs between them.

Deliberately NOT built like combined_chart.py, which puts premise soundness (a fraction),
Vendi diversity (an index >= 1) and accuracy (a fraction) on ONE shared y-axis and then
annotates the legend with "not a fraction" to explain why one series cannot be compared to
the others. Different measures get different panels here; nothing shares an axis with
something it is not commensurable with.

Panels:
  A  maj@k at T=1.0: flat, argmax at the edge. No interior optimum at this temperature.
  B  T=1.0 vs T=1.6 at k=1: the falling arm appears only once temperature can inflate the
     tail, which is the whole prediction.
  C  the headline itself -- maj@k at T=1.6, where the interior optimum lives. Earlier
     versions of this figure never plotted it, so the strongest arm of the claim (curvature
     rising with k, P(shape) 0.916 -> 0.998) was the one arm with no panel.

Per-top_p spoil rates are printed by analyze.py into RESULT_T*.md and are not plotted. They
are a diagnostic, not a result: nothing here is computed by dropping them.
"""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical slots (scripts/validate_palette.js, light mode, surface #fcfcfb):
# blue/orange PASS all six checks; aqua carries a contrast WARN so it is always direct-
# labelled, which is the relief the validator requires.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"

def style(ax, title, ylab, xlab="top_p"):
    ax.set_facecolor(SURF)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_ylabel(ylab, color=INK2, fontsize=9)
    ax.set_xlabel(xlab, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)

def line(ax, xs, ys, color, label, marker="o"):
    # thin marks: 2px lines, >=8px markers, 2px surface ring so overlaps stay readable
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, label=label,
            markeredgecolor=SURF, markeredgewidth=2, zorder=3)

def callout(ax, x, y, text, color, dy, grid, fontsize=8, bold=True):
    """Annotate without spilling past the axes: a label centred on the LAST grid point
    overruns the right spine, so edge points are aligned inward instead."""
    if x >= grid[-1]:
        ha, dx = "right", 4
    elif x <= grid[0]:
        ha, dx = "left", -4
    else:
        ha, dx = "center", 0
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                color=color, fontsize=fontsize, ha=ha,
                fontweight="bold" if bold else "normal")


def load(p):
    return json.load(open(p)) if Path(p).exists() else None

def row(d, k):
    return next(r for r in d["results"] if r["k"] == k)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--t10", default="outputs/RESULT_T10.json",
                    help="T=1.0 analysis json (falls back to the 7-point grid if absent)")
    ap.add_argument("--t16", default="outputs/RESULT_T16.json")
    ap.add_argument("--out", default="outputs/corrected_topp_result.png")
    a = ap.parse_args()
    d10 = load(a.t10) or None
    if d10 is None or "grid" not in d10:
        sys.exit("no T=1.0 analysis json with a stamped grid yet "
                 "(re-run phase1_analysis.py; older jsons predate the grid field)")
    d16 = load(a.t16)
    if d16 is not None and "grid" not in d16:
        d16 = None
    grid = d10["grid"]

    n = 3 if d16 else 1
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor(SURF)

    # --- A: no interior optimum at T=1.0, at any k ---------------------------
    ax = axes[0]; style(ax, "A. Majority vote (T=1.0)", "fraction correct")
    for k, c, m in ((1, BLUE, "o"), (16, AQUA, "^")):
        r = row(d10, k)
        line(ax, grid, r["means"], c, f"maj@{k}", m)
        if c is AQUA:                    # direct label = the contrast-WARN relief for aqua
            callout(ax, grid[-1], r["means"][-1], "maj@16", AQUA, 10, grid, fontsize=7.5)
    ax.margins(x=0.12, y=0.16)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    # --- B: the temperature test ---------------------------------------------
    if d16:
        ax = axes[1]; style(ax, "B. T=1.0 vs T=1.6 (k=1)", "fraction correct")
        line(ax, grid, row(d10, 1)["means"], BLUE, "T = 1.0", "o")
        line(ax, d16["grid"], row(d16, 1)["means"], ORANGE, "T = 1.6", "s")
        # T=1.6 ends in the lower right and T=1.0 runs along the top, so both right-hand
        # corners are occupied; the lower left is empty because T=1.0 starts at top_p=0.5.
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")

    # --- C: the headline, which no earlier version of this figure plotted -----
    # The right arm of the inverted-U is a 65 pp cliff and the left arm is a few pp, so on
    # the full y-range (panel B) the interior peak is a flat smear. This panel is therefore
    # the 0.1-0.7 detail; B carries the full range so the cliff is never hidden.
    if d16:
        ax = axes[2]
        style(ax, "C. Inverted-U at T=1.6 (maj@k, 0.1-0.7 detail)", "fraction correct")
        g16 = d16["grid"]
        keep = [i for i, p in enumerate(g16) if p <= 0.7]
        xs = [g16[i] for i in keep]
        # three argmax labels in this little space collide; the claim is a peak REGION
        # (argmax oscillates 0.3/0.5 with k), so band it once instead of labelling each.
        ax.axvspan(0.3, 0.5, color=GRID, alpha=0.55, lw=0, zorder=0)
        for k, c, m in ((1, BLUE, "o"), (16, AQUA, "^")):
            r = row(d16, k)
            ys = [r["means"][i] for i in keep]
            line(ax, xs, ys, c, f"maj@{k}", m)
            if c is AQUA:                # direct label = the contrast-WARN relief for aqua
                callout(ax, xs[-1], ys[-1], "maj@16", AQUA, 10, xs, fontsize=7.5)
        ax.margins(x=0.12, y=0.20)
        ax.annotate("argmax 0.3-0.5\nat every k", (0.4, 1.0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -14), ha="center",
                    color=INK2, fontsize=8, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")

    fig.tight_layout()
    out = a.out
    fig.savefig(out, dpi=170, facecolor=SURF)
    print(f"[chart] -> {out}")

if __name__ == "__main__":
    main()
