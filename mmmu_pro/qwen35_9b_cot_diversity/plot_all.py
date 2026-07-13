#!/usr/bin/env python
"""One summary chart vs top_p:
  - majority-vote accuracy (inverted-U, peak 0.7)   [173Q x 6, cot_report_stopped.json]
  - per-sample answer accuracy (flat)               [same]
  - Qwen self-judge soundness (flat ~0.89, UNRELIABLE/anchored)  [same]
  - Sonnet independent soundness (flat ~0.61, no gold) + SE      [40Q x 1, sonnet_verdicts.json]
  - Vendi diversity (rising, right axis)            [173Q x 6]
Two data sources (noted in caption); both are top_p sweeps on MMMU-Pro / Qwen3.5-9B.
"""
import json, collections, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep = json.load(open("outputs/cot_report_stopped.json"))
evo = rep["evolution"]; ps = [e["top_p"] for e in evo]; x = np.arange(len(ps))

sv = json.load(open("outputs/sonnet_verdicts.json"))
byp = collections.defaultdict(lambda: [0, 0])
for r in sv:
    byp[r["top_p"]][1] += 1; byp[r["top_p"]][0] += 1 if r["sound"] else 0
son = [byp[p][0] / byp[p][1] if p in byp else np.nan for p in ps]
son_se = [math.sqrt(a*(1-a)/byp[p][1]) if p in byp else 0 for a, p in zip(son, ps)]

fig, axL = plt.subplots(figsize=(11, 6.4)); axR = axL.twinx()

axL.plot(x, [e["majority_acc"] for e in evo], "-^", color="#2c8c3c", lw=2.6, ms=10,
         label="majority-vote accuracy  (INVERTED-U, peak 0.7)")
axL.plot(x, [e["answer_acc"] for e in evo], "-o", color="#1b6ca8", lw=2.2, ms=7,
         label="per-sample answer accuracy  (flat)")
axL.errorbar(x, son, yerr=son_se, fmt="-s", color="#b5179e", lw=2.4, ms=8, capsize=4,
             label="CoT soundness — Sonnet, independent  (flat ~0.61)")
axL.plot(x, [e["cot_correct"] for e in evo], ":", color="#c85200", lw=2.2,
         label="CoT soundness — Qwen self-judge  (flat ~0.89, unreliable)")
axR.plot(x, [e["vendi"] for e in evo], "--D", color="#6a3d9a", lw=2.4, ms=7,
         label="Vendi diversity  (rises with top_p)")

imax = int(np.argmax([e["majority_acc"] for e in evo]))
axL.annotate("peak", xy=(x[imax], evo[imax]["majority_acc"]),
             xytext=(x[imax], evo[imax]["majority_acc"] + 0.03), ha="center", fontsize=9,
             arrowprops=dict(arrowstyle="->", color="#2c8c3c"))

axL.set_xticks(x); axL.set_xticklabels([str(p) for p in ps])
axL.set_xlabel("nucleus sampling  top_p", fontsize=12)
axL.set_ylabel("accuracy / soundness  (fraction)", fontsize=12)
axR.set_ylabel("Vendi diversity  d", color="#6a3d9a", fontsize=12)
axR.tick_params(axis="y", labelcolor="#6a3d9a")
axL.set_ylim(0.55, 0.95); axR.set_ylim(1.0, 1.6); axL.grid(alpha=0.25)

h1, l1 = axL.get_legend_handles_labels(); h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=9, framealpha=0.92, ncol=1)

axL.set_title("MMMU-Pro / Qwen3.5-9B: everything vs top_p — only MAJORITY VOTE humps; "
              "per-trace metrics are flat; diversity rises\n"
              "(accuracy/diversity/Qwen: 173 Q x 6 samples;  Sonnet soundness: 40 Q, 1 sample/cell, no gold)",
              fontsize=10)
fig.tight_layout(); fig.savefig("outputs/cot_summary_all.png", dpi=150, bbox_inches="tight")
print("saved outputs/cot_summary_all.png")
