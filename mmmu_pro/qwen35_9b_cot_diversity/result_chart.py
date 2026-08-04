#!/usr/bin/env python
"""The corrected top_p result, as one figure.

Deliberately NOT built like combined_chart.py, which puts premise soundness (a fraction),
Vendi diversity (an index >= 1) and accuracy (a fraction) on ONE shared y-axis and then
annotates the legend with "not a fraction" to explain why one series cannot be compared to
the others. Different measures get different panels here; nothing shares an axis with
something it is not commensurable with.

Panels:
  A  per-sample accuracy vs top_p under the two counting rules. This is the whole finding:
     the "interior peak at 0.5" exists under `exclude` and vanishes under the ballot rule.
  B  the spoil rate that causes A -- the differential the exclude rule silently applies.
  C  maj@k, to show the same flatness is not an artifact of the k=1 metric.
  D  T=1.0 vs T=1.6 at k=1, once the T-sweep lands (HANDOFF 8): the test of whether the
     falling arm appears once temperature can inflate the tail.
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

def row(d, arm, k):
    return next(r for r in d[arm] if r["k"] == k)

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

    n = 4 if d16 else 3
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4))
    fig.patch.set_facecolor(SURF)

    # --- A: the artifact -----------------------------------------------------
    ax = axes[0]; style(ax, "A. Per-sample accuracy (k=1)", "fraction correct")
    line(ax, grid, row(d10, "exclude_full", 1)["means"], ORANGE,
         "exclude truncated (original rule)", "s")
    line(ax, grid, row(d10, "spoiled_full", 1)["means"], BLUE,
         "spoiled ballot (corrected)", "o")
    ex, sp = row(d10, "exclude_full", 1), row(d10, "spoiled_full", 1)
    callout(ax, ex["argmax"], max(ex["means"]), f"argmax {ex['argmax']}", ORANGE, 12, grid)
    callout(ax, sp["argmax"], max(sp["means"]), f"argmax {sp['argmax']}", BLUE, -20, grid)
    ax.margins(y=0.22)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    # --- B: the cause --------------------------------------------------------
    ax = axes[1]; style(ax, "B. Spoiled ballots (why A differs)", "% of generations")
    sr = [100 * (d10["spoil"][str(p)]["trunc"] + d10["spoil"][str(p)]["unparsed"])
          / d10["spoil"][str(p)]["n"] for p in grid]
    line(ax, grid, sr, AQUA, "truncated or unparseable", "^")
    for x, y in zip(grid, sr):                       # direct labels = contrast-WARN relief
        callout(ax, x, y, f"{y:.1f}%", INK2, 9, grid, fontsize=7.5, bold=False)
    ax.margins(y=0.20)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    # --- C: not a k=1 artifact ----------------------------------------------
    ax = axes[2]; style(ax, "C. Majority vote (spoiled ballot)", "fraction correct")
    for k, c, m in ((1, BLUE, "o"), (4, ORANGE, "s"), (16, AQUA, "^")):
        r = row(d10, "spoiled_full", k)
        line(ax, grid, r["means"], c, f"maj@{k}", m)
    ax.margins(x=0.12, y=0.14)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    # --- D: the temperature test --------------------------------------------
    if d16:
        ax = axes[3]; style(ax, "D. T=1.0 vs T=1.6 (k=1)", "fraction correct")
        g16 = d16["grid"]
        line(ax, grid, row(d10, "spoiled_full", 1)["means"], BLUE, "T = 1.0", "o")
        line(ax, g16, row(d16, "spoiled_full", 1)["means"], ORANGE, "T = 1.6", "s")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")

    fig.tight_layout()
    out = a.out
    fig.savefig(out, dpi=170, facecolor=SURF)
    print(f"[chart] -> {out}")

if __name__ == "__main__":
    main()
