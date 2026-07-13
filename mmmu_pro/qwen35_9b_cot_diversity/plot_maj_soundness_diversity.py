#!/usr/bin/env python
"""One chart vs top_p: majority-vote accuracy (inverted-U) + CoT soundness (left axis) and
Vendi diversity (right axis). Uses the completed-only report by default."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "outputs/cot_report_stopped.json"))
evo = rep["evolution"]; ps = [e["top_p"] for e in evo]; x = np.arange(len(ps))

fig, axL = plt.subplots(figsize=(10, 6)); axR = axL.twinx()

axL.plot(x, [e["majority_acc"] for e in evo], "-^", color="#2c8c3c", lw=2.6, ms=10,
         label="majority-vote accuracy (self-consistency)")
axL.plot(x, [e["cot_correct"] for e in evo], "-s", color="#c85200", lw=2.4, ms=8,
         label="CoT soundness (self-judge)")
axR.plot(x, [e["vendi"] for e in evo], "--D", color="#6a3d9a", lw=2.4, ms=7,
         label="Vendi diversity (d)")

imax = int(np.argmax([e["majority_acc"] for e in evo]))
axL.annotate("inverted-U peak", xy=(x[imax], evo[imax]["majority_acc"]),
             xytext=(x[imax], evo[imax]["majority_acc"] + 0.02), ha="center", fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#2c8c3c"))

axL.set_xticks(x); axL.set_xticklabels([str(p) for p in ps])
axL.set_xlabel("nucleus sampling  top_p", fontsize=12)
axL.set_ylabel("accuracy / soundness  (fraction)", fontsize=12)
axR.set_ylabel("Vendi diversity  d", color="#6a3d9a", fontsize=12)
axR.tick_params(axis="y", labelcolor="#6a3d9a")
axL.set_ylim(0.70, 0.95); axR.set_ylim(1.0, 1.6); axL.grid(alpha=0.25)

h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="center right", fontsize=10, framealpha=0.9)

c = rep["correlations"]
axL.set_title("MMMU-Pro / Qwen3.5-9B (173 Q x 6, completed-only): "
              "majority-vote accuracy vs CoT soundness vs diversity\n"
              f"maj-vote inverted-U (peak top_p 0.7)  |  soundness flat  |  "
              f"top_p↔diversity {c['top_p__vendi']['pearson']:+.2f}", fontsize=10.5)
fig.tight_layout(); fig.savefig("outputs/cot_maj_soundness_diversity.png", dpi=150, bbox_inches="tight")
print("saved outputs/cot_maj_soundness_diversity.png")
