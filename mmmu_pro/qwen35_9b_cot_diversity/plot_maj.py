#!/usr/bin/env python
"""Majority-vote (self-consistency) accuracy vs top_p — the metric that traces the INVERTED-U.
Also overlays per-sample accuracy (flat) for contrast. Reads outputs/cot_report.json."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "outputs/cot_report.json"))
evo = rep["evolution"]; ps = [e["top_p"] for e in evo]; x = np.arange(len(ps))

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.plot(x, [e["majority_acc"] for e in evo], "-^", color="#2c8c3c", lw=2.6, ms=10,
        label="majority-vote accuracy (self-consistency)")
ax.plot(x, [e["answer_acc"] for e in evo], "-o", color="#1b6ca8", lw=2.0, ms=7, alpha=0.85,
        label="per-sample answer accuracy")
imax = int(np.argmax([e["majority_acc"] for e in evo]))
ax.annotate("peak (inverted-U)", xy=(x[imax], evo[imax]["majority_acc"]),
            xytext=(x[imax], evo[imax]["majority_acc"] + 0.03),
            ha="center", fontsize=9, arrowprops=dict(arrowstyle="->", color="#2c8c3c"))
ax.set_xticks(x); ax.set_xticklabels([str(p) for p in ps])
ax.set_xlabel("nucleus sampling  top_p", fontsize=12)
ax.set_ylabel("accuracy (fraction)", fontsize=12)
ax.set_ylim(0.68, 0.82); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="lower left")
ax.set_title("MMMU-Pro / Qwen3.5-9B: majority-vote accuracy is inverted-U (peak ~top_p 0.7);\n"
             "per-sample accuracy is flat  (173 Q x 6 samples — rerun with 16-32 for a cleaner hump)",
             fontsize=10.5)
fig.tight_layout(); fig.savefig("outputs/cot_maj_vote.png", dpi=150, bbox_inches="tight")
print("saved outputs/cot_maj_vote.png")
