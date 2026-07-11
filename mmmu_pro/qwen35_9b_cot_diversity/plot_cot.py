#!/usr/bin/env python
"""Figures for the CoT-diversity analysis (865 = 173 Q x 5 top_p cells).

Per-example legends don't scale to 173 questions, so these are AGGREGATE views:
  cot_evolution.png              : (L) accuracy/soundness/majority vs top_p; (R) diversity vs top_p.
  cot_diversity_vs_correctness.png: (L) vendi vs CoT-soundness; (R) vendi vs answer-accuracy;
                                    each a per-cell scatter colored by top_p with a trend line.
"""
import json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep = json.load(open("outputs/cot_report.json"))
cells = rep["per_cell"]; ps = rep["top_ps"]; evo = rep["evolution"]; corr = rep["correlations"]
cmap = plt.get_cmap("viridis"); pcol = {p: cmap(i / max(1, len(ps) - 1)) for i, p in enumerate(ps)}

# ---------- Figure 1: evolution over top_p ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
for key, lab, mk in [("answer_acc", "answer accuracy", "-o"),
                     ("cot_correct", "CoT soundness (self-judge)", "-s"),
                     ("majority_acc", "majority-vote accuracy", "-^")]:
    ax1.plot(ps, [e[key] for e in evo], mk, lw=2, ms=7, label=lab)
ax1.set_xlabel("nucleus top_p"); ax1.set_ylabel("fraction"); ax1.set_xticks(ps)
ax1.set_ylim(0.6, 0.95); ax1.grid(alpha=0.3); ax1.legend(fontsize=9)
ax1.set_title("Accuracy & CoT-soundness vs top_p")

ax2.plot(ps, [e["vendi"] for e in evo], "-o", color="tab:purple", lw=2, ms=7, label="Vendi score")
ax2.set_xlabel("nucleus top_p"); ax2.set_ylabel("Vendi (6 CoTs / cell)", color="tab:purple")
ax2.tick_params(axis="y", labelcolor="tab:purple"); ax2.set_xticks(ps); ax2.grid(alpha=0.3)
axb = ax2.twinx()
axb.plot(ps, [e["cos_dist"] for e in evo], "-s", color="tab:orange", lw=2, ms=6, label="mean cosine dist")
axb.set_ylabel("mean pairwise cosine distance", color="tab:orange")
axb.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_title("CoT diversity vs top_p (near-flat)")
fig.tight_layout(); fig.savefig("outputs/cot_evolution.png", dpi=150, bbox_inches="tight")
print("saved outputs/cot_evolution.png")

# ---------- Figure 2: diversity vs correctness (per-cell scatter) ----------
def scatter(ax, xkey, ykey, ylab, ckey):
    for p in ps:
        cs = [c for c in cells if c["top_p"] == p and c[ykey] is not None]
        ax.scatter([c[xkey] for c in cs], [c[ykey] for c in cs], s=18, alpha=0.5,
                   color=pcol[p], label=f"top_p={p}")
    x = np.array([c[xkey] for c in cells if c[ykey] is not None])
    y = np.array([c[ykey] for c in cells if c[ykey] is not None])
    m, b = np.polyfit(x, y, 1); xx = np.linspace(x.min(), x.max(), 50)
    pe = corr[ckey]["pearson"]; sp = corr[ckey]["spearman"]
    ax.plot(xx, m * xx + b, "k--", lw=1.8, label=f"trend (Pearson {pe:+.2f}, Spearman {sp:+.2f})")
    ax.set_xlabel("Vendi diversity (6 CoTs / cell)"); ax.set_ylabel(ylab); ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))
scatter(a1, "vendi", "cot_correct", "CoT soundness (fraction sound)", "vendi__cot_correct")
a1.set_title("CoT diversity vs reasoning soundness")
scatter(a2, "vendi", "answer_acc", "answer accuracy (fraction)", "vendi__answer_acc")
a2.set_title("CoT diversity vs answer accuracy")
fig.tight_layout(); fig.savefig("outputs/cot_diversity_vs_correctness.png", dpi=150, bbox_inches="tight")
print("saved outputs/cot_diversity_vs_correctness.png")
