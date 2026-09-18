#!/usr/bin/env python
"""The three quantities on ONE axis: soundness, accuracy (maj@k), diversity vs top_p, one arm.

They have incompatible units -- soundness and accuracy are fractions, Vendi is an effective
COUNT bounded [1, K]. Two ways to share an axis:

  Vendi -> (VS-1)/(K-1).  REJECTED. A defensible 0-1 rescaling, but with K=20 it lands the
  diversity series at 0.126-0.142, a flat sliver below fractions sitting at 0.68-0.94, so a
  +10% change reads as no change. A normalisation that hides the effect it is meant to
  display is worse than no chart.

  Index each series to its own value at the left edge.  USED. Every line starts at 1.00 and
  the y-axis reads "relative to top_p = 0.5", so what is compared is the SHAPE and the SIZE
  OF THE CHANGE -- the actual question -- rather than levels that were never comparable.
  Absolute values are annotated on every point so nothing is lost to the rescaling.

Error bars are SEM/baseline: they carry each point's own uncertainty but not the baseline's,
so the leftmost point necessarily shows none. That is a property of indexing, stated rather
than hidden; premise_soundness_vs_topp.png carries the honest absolute bars.
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vp", default="outputs/RESULT_VP.json")
    ap.add_argument("--div", default="outputs/RESULT_VP_DIVERSITY.json")
    ap.add_argument("--out", default="outputs/topp_correctness_and_diversity.png")
    ap.add_argument("--accuracy", choices=["majk", "maj1"], default="majk",
                    help="majk: majority vote over the judged samples (maj@8 here); maj1: per-sample")
    ap.add_argument("--label", default=None,
                    help="arm label for the title; defaults to the one stored in --vp")
    a = ap.parse_args()
    vp, dv = json.load(open(a.vp)), json.load(open(a.div))
    g, st = vp["grid"], dv["stats"]
    arm_label = a.label or vp["label"]
    n_layers = len(vp.get("layers", [])) or 8
    n_prem = dv.get("n_premises")
    prem_note = f"{n_prem:,} visual premises" if n_prem else "all extracted visual premises"

    # (label, values, sems, colour, marker, label-side, formatter, x-nudge)
    series = [
        ("visual-premise soundness", vp["soundness_mean"], vp["soundness_sem"], ORANGE, "s",
         -13, lambda v: f"{v:.3f}", 8),
        ((f"answer accuracy (maj@{vp['majk_k']})", vp["majk_accuracy_mean"], vp["majk_accuracy_sem"])
         if a.accuracy == "majk" else ("answer accuracy (maj@1)", vp["accuracy_mean"], vp["accuracy_sem"]))
        + (BLUE, "o", -13, lambda v: f"{v:.3f}", -8),
        (f"premise diversity (Vendi, count-matched to K={dv.get('k_cell', 20)})", st["vendi_cell"]["means"],
         st["vendi_cell"].get("sem"), AQUA, "^", 13, lambda v: f"{v:.2f}", 0),
    ]
    # The model-free numeric-read measure is deliberately NOT plotted: it rests on roughly half
    # the paired questions of the Vendi series, its error bars are 3-4x wider, and its magnitude
    # is not commensurable. It lives in RESULT_VP_DIVERSITY.md.

    # Wider than the lines alone need: every point carries its pre-indexing value, and six of
    # the ten grid points sit in the last tenth of the axis.
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ax.axhline(1.0, color=GRID, lw=1.2, zorder=1)
    for label, ys, es, c, mk, dy, fmt, dx in series:
        ys = np.asarray(ys, float)
        base = ys[0]
        idx = ys / base
        ax.plot(g, idx, color=c, lw=2, marker=mk, ms=7,
                label=f"{label}   [{ys[0]:.3f} → {ys[-1]:.3f}]",
                markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        rel = (np.asarray(es, float) / base) if es is not None else np.zeros(len(g))
        if es is not None:
            ax.errorbar(g, idx, yerr=rel, fmt="none", ecolor=c, elinewidth=1.3,
                        capsize=3, capthick=1.3, alpha=0.7, zorder=2)
        for x, yv, av, e in zip(g, idx, ys, rel):
            # anchor OUTSIDE the error-bar cap, rotated, and nudged sideways: both fraction
            # series label downwards and the soundness line runs below the accuracy line, so
            # their labels would otherwise land in the same strip
            anchor = yv + e if dy > 0 else yv - e
            ax.annotate(fmt(av), (x, anchor), textcoords="offset points",
                        xytext=(dx, 6 if dy > 0 else -6),
                        ha="center", va="bottom" if dy > 0 else "top", rotation=90,
                        color=c, fontsize=6.4, fontweight="bold", zorder=5,
                        bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.75, pad=0.6))

    ax.set_title("What top_p changes: visual-premise soundness, answer accuracy and premise "
                 f"diversity\n{arm_label}, every series indexed to its own top_p = {g[0]} value",
                 color=INK, fontsize=10.5, loc="left", pad=10)
    ax.set_xlabel("top_p", color=INK2, fontsize=9)
    ax.set_ylabel(f"relative to top_p = {g[0]}   (labels: the value before indexing)",
                  color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7); ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.margins(x=0.06, y=0.34)
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    leg.set_title(f"series   [absolute value at {g[0]} → {g[-1]}]", prop={"size": 8})
    leg.get_title().set_color(INK2)
    fig.text(0.008, 0.048, f"{n_layers} samples per (question, top_p) cell; {prem_note}, "
             "judged by InternVL3-8B-AWQ with the gold answer withheld.", color=INK2, fontsize=7.5)
    fig.text(0.008, 0.026, "Diversity is count-matched, because premise count itself drifts "
             f"along the axis. Paired questions: soundness/accuracy n={vp.get('n_questions')}, "
             f"diversity n={st['vendi_cell'].get('n')}.", color=INK2, fontsize=7.5)
    fig.text(0.008, 0.004, "Error bars are +/-1 SEM / baseline, so they omit the baseline's "
             "own uncertainty; absolute ones are in premise_soundness_vs_topp.png.",
             color=INK2, fontsize=7.5)
    fig.tight_layout(rect=[0, 0.075, 1, 1])
    fig.savefig(a.out, dpi=170, facecolor=SURF)
    print(f"[chart] -> {a.out}")
    for label, ys, *_ in series:
        ys = np.asarray(ys, float)
        print(f"  {label:44s} {ys[0]:.4f} -> {ys[-1]:.4f}  ({100*(ys[-1]/ys[0]-1):+.1f}%)")


if __name__ == "__main__":
    main()
