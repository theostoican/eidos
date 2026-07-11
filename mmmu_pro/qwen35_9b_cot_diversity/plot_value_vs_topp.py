#!/usr/bin/env python
"""Single 'value vs top_p' overlay (the hand-sketched layout): overall answer accuracy,
CoT soundness, and majority-vote accuracy on the left axis; Vendi diversity on the right
axis, all over the nucleus top_p sweep. Mirrors the sibling premise figure's intent but
for the real CoT-diversity run (173 Q x 5 top_p x 6 samples).
"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = sys.argv[1] if len(sys.argv) > 1 else "outputs/cot_report.json"
OUT    = sys.argv[2] if len(sys.argv) > 2 else "outputs/cot_value_vs_topp.png"
NOTE   = sys.argv[3] if len(sys.argv) > 3 else ""
rep = json.load(open(REPORT))
evo = rep["evolution"]; ps = [e["top_p"] for e in evo]
x = np.arange(len(ps))                      # even spacing so 0.9/0.95/1.0 don't bunch up

fig, axL = plt.subplots(figsize=(10, 6))
axR = axL.twinx()

# left axis: fraction-scale "value" metrics
series = [("answer_acc",  "overall answer accuracy", "#1b6ca8", "-o"),
          ("cot_correct", "CoT soundness (self-judge)", "#c85200", "-s"),
          ("majority_acc","majority-vote accuracy", "#2c8c3c", "-^")]
for key, lab, col, mk in series:
    axL.plot(x, [e[key] for e in evo], mk, color=col, lw=2.4, ms=8, label=lab)

# right axis: diversity
axR.plot(x, [e["vendi"] for e in evo], "--D", color="#6a3d9a", lw=2.4, ms=7,
         label="Vendi diversity (d)")

axL.set_xlabel("nucleus sampling  top_p  (P)", fontsize=12)
axL.set_ylabel("accuracy / soundness  (fraction)", fontsize=12)
axR.set_ylabel("Vendi diversity  d  (6 CoTs / cell)", color="#6a3d9a", fontsize=12)
axR.tick_params(axis="y", labelcolor="#6a3d9a")
axL.set_xticks(x); axL.set_xticklabels([str(p) for p in ps])
axL.set_ylim(0.60, 0.95)
axR.set_ylim(1.0, 1.6)
axL.grid(alpha=0.25)

# combined legend
h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="center right", fontsize=10, framealpha=0.9)

c = rep["correlations"]
axL.set_title(f"CoT-diversity sweep (MMMU-Pro, Qwen3.5-9B, 173 Q x 6 samples){NOTE}\n"
              f"diversity↔soundness Pearson {c['vendi__cot_correct']['pearson']:+.2f}   |   "
              f"diversity↔answer-acc {c['vendi__answer_acc']['pearson']:+.2f}   |   "
              f"top_p↔diversity {c['top_p__vendi']['pearson']:+.2f}", fontsize=10.5)

fig.tight_layout(); fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
