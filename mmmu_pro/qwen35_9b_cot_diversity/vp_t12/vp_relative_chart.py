#!/usr/bin/env python
"""Soundness, maj@k accuracy and premise diversity on ONE axis, each series divided by its own
value at the leftmost top_p (0.5). Labels carry the ABSOLUTE value at every point; bars are
+/-1 SEM scaled by the same baseline. Same style as ../t12_chart.py.

  python vp_relative_chart.py [--vp outputs/RESULT_VP.json] [--div outputs/RESULT_VP_DIVERSITY.json]
                              [--out outputs/vp_relative.png]
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1fa36b"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vp", default="outputs/RESULT_VP.json")
    ap.add_argument("--div", default="outputs/RESULT_VP_DIVERSITY.json")
    ap.add_argument("--card", default="outputs/card_point.json",
                    help="model-card reference from card_point.py; skipped if absent")
    ap.add_argument("--out", default="outputs/vp_relative.png")
    a = ap.parse_args()
    vp, dv = json.load(open(a.vp)), json.load(open(a.div))
    grid = vp["grid"]; assert dv["grid"] == grid
    k, cell = vp["majk_k"], dv["stats"]["vendi_cell"]
    Fs, ps = vp["anova"]["soundness"]; Fk, pk = vp["anova"]["majk"]

    # (name, means, sem, colour, marker, abs-label format, label offset in points, n, F, p)
    series = [
        (f"answer accuracy (maj@{k})", vp["majk_accuracy_mean"], vp["majk_accuracy_sem"], BLUE, "o",
         "{:.3f}", -9, vp["n_questions"], Fk, pk),
        ("visual-premise soundness", vp["soundness_mean"], vp["soundness_sem"], ORANGE, "s",
         "{:.3f}", -20, vp["n_questions"], Fs, ps),
        (f"premise diversity (Vendi, count-matched K={dv['k_cell']})", cell["means"], cell["sem"], GREEN, "^",
         "{:.2f}", 8, cell["n"], cell["F"], cell["p"]),
    ]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=150); fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(1.0, color=INK2, lw=0.8, ls="--", alpha=0.5)
    lab = dict(textcoords="offset points", ha="center", fontsize=7,
               bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.85))
    rel = {nm: np.array(m) / m[0] for nm, m, *_ in series}
    acc_r, div_r = rel[series[0][0]], rel[series[2][0]]
    # accuracy labels go above their point once the diversity curve is clear of it, else below;
    # soundness labels stack below whichever of the two is already there
    acc_dy = np.where(div_r - acc_r > 0.02, 8, -9)
    snd_dy = np.where(acc_dy < 0, -20, -9)
    for name, m, s, col, mk, fmt, dy, n, F, p in series:
        m, s = np.array(m), np.array(s)
        base = m[0]
        r, e = m / base, s / base
        dys = acc_dy if name.startswith("answer") else snd_dy if name.startswith("visual") else np.full(len(grid), dy)
        ax.fill_between(grid, r - e, r + e, color=col, alpha=0.10, lw=0)
        ax.plot(grid, r, color=col, lw=1.8, marker=mk, ms=5.5,
                label=f"{name}   [{m[0]:{fmt[2:-1]}} → {m[-1]:{fmt[2:-1]}}, {(r[-1]-1)*100:+.1f}%]  "
                      f"n={n}, F={F:.1f}, p={p:.1e}")
        for x, y, v, d in zip(grid, r, m, dys):
            ax.annotate(fmt.format(v), (x, y), xytext=(0, int(d)), va="bottom" if d > 0 else "top",
                        color=col, **lab)
    # the model-card defaults as a point: card maj@k on the same paired questions, divided by the
    # sweep's accuracy at top_p = 0.5 (outputs/card_point.json, written by card_point.py)
    cp = Path(a.card)
    if cp.is_file():
        c = json.load(open(cp))
        base = series[0][1][0]
        y, e = c["majk"] / base, c["majk_sem"] / base
        ax.errorbar([c["top_p"]], [y], yerr=[e], fmt="none", ecolor=BLUE, elinewidth=1.2, capsize=3, zorder=4)
        ax.plot([c["top_p"]], [y], marker="*", ms=14, color=BLUE, mec=INK, mew=0.8, ls="none", zorder=5,
                label=f"model card maj@{c['k']} ({c['majk']:.3f}, {(y-1)*100:+.1f}%) · "
                      f"T=1.0, top_k 20, p=.95, presence 1.5 · same n={c['n_questions']}")
        ax.annotate(f"card {c['majk']:.3f}", (c["top_p"], y), xytext=(0, 10), va="bottom", color=BLUE, **lab)
    ax.set_xlabel("top_p"); ax.set_xticks(grid)
    ax.set_ylabel("relative to the value at top_p = 0.5  (labels: absolute values)")
    ax.margins(x=0.05, y=0.12)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2, loc="lower left")
    ax.set_title(f"{vp['label']}\n{vp['n_traces']:,} traces · {vp['n_premises']:,} premises · "
                 f"{vp['n_samples']} samples/cell · judge InternVL3-8B-AWQ (gold withheld)",
                 loc="left", color=INK2, fontsize=8, pad=8)
    fig.suptitle("T=1.2: premise soundness, answer accuracy and premise diversity vs top_p,\n"
                 "each relative to its own value at top_p = 0.5",
                 x=0.01, y=0.995, ha="left", color=INK, fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(a.out, facecolor=SURF)
    print(f"[chart] -> {a.out}")

if __name__ == "__main__":
    main()
