#!/usr/bin/env python
"""The top_p result as one figure: maj@1 and maj@16 at T=1.6.

  python result_chart.py [--t16 outputs/RESULT_T16.json] [--out outputs/corrected_topp_result.png]

One counting rule throughout: spoiled-ballot. Both curves are exact and read straight from the
committed RESULT_T16.json (analyze.py --ks 1,16): the arm carries 16 ballots per cell, so maj@16 is
the majority over every ballot, not a subsample estimate. Only the T=1.6 arm is plotted; the T=1.0
arm's numbers are in outputs/RESULT_T10.md and the README tables.

Per-top_p spoil rates are printed by analyze.py into RESULT_T*.md and are not plotted. They are a
diagnostic, not a result: nothing here is computed by dropping them.
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
SAMPLES = 16          # ballots per cell
K = 16                # the second curve: the majority over all 16 ballots
PASS = 0.95           # the pre-registered bar: a shape claim stands only at P(joint) >= 0.95

def row(d, k):
    return next(r for r in d["results"] if r["k"] == k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t16", default="outputs/RESULT_T16.json")
    ap.add_argument("--out", default="outputs/corrected_topp_result.png")
    a = ap.parse_args()
    d = json.load(open(a.t16))
    ks = [r["k"] for r in d["results"]]
    if 1 not in ks or K not in ks:
        raise SystemExit(f"{a.t16} has k={ks}; this figure needs k=1 and k={K} (analyze.py --ks 1,{K})")
    grid = d["grid"]
    n_q = round(d["spoil"][str(grid[0])]["n"] / SAMPLES)
    r1, rk = row(d, 1), row(d, K)
    m1, s1, mk = np.array(r1["means"]), np.array(r1["sem"]), np.array(rk["means"])

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150); fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.fill_between(grid, m1 - s1, m1 + s1, color=BLUE, alpha=0.12, lw=0)
    ax.plot(grid, m1, color=BLUE, lw=1.8, marker="o", ms=5.5, label="maj@1 (±1 SEM)")
    ax.plot(grid, mk, color=ORANGE, lw=1.8, marker="o", ms=5.5, label=f"maj@{K}")
    # Each curve's argmax as a dotted line in its own colour, and each curve's verdict against the
    # pre-registered bar written ABOVE the plot area, where no data can run into it. The two k reach
    # different verdicts here -- maj@1 misses the bar, maj@16 clears it -- so one line would mislead.
    for j, (r, m, c) in enumerate(((r1, m1, BLUE), (rk, mk, ORANGE))):
        i = int(np.argmax(m))
        ax.axvline(grid[i], color=c, lw=0.9, ls=":", alpha=0.9, zorder=1)
        ok = "passes 0.95" if r["p_joint"] >= PASS else "below 0.95"
        ax.annotate(f"maj@{r['k']}  argmax {grid[i]} · P(joint) {r['p_joint']:.3f} ({ok})",
                    (0, 1), xycoords="axes fraction", textcoords="offset points",
                    xytext=(0, 15 - 12 * j), ha="left", va="bottom", color=c, fontsize=7.5)
    ax.set_title(f"Qwen3.5-9B, T={d['temperature']}, n={n_q} questions",
                 loc="left", color=INK, fontsize=10, pad=30)
    ax.set_xlabel("top_p"); ax.set_ylabel("accuracy")
    ax.margins(x=0.06, y=0.14)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="lower left")   # the collapse leaves it empty
    fig.suptitle(f"maj@1 and maj@{K} vs top_p", x=0.01, ha="left", color=INK, fontsize=10.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, facecolor=SURF)
    print(f"[chart] -> {a.out}")

if __name__ == "__main__":
    main()
