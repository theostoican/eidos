#!/usr/bin/env python
"""Independent Sonnet CoT-soundness (no gold) vs top_p — flat, and far below the Qwen
self-judge. Reads outputs/sonnet_verdicts.json."""
import json, collections, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

v = json.load(open("outputs/sonnet_verdicts.json"))
byp = collections.defaultdict(lambda: [0, 0])
for r in v:
    byp[r["top_p"]][1] += 1; byp[r["top_p"]][0] += 1 if r["sound"] else 0
ps = sorted(byp); x = np.arange(len(ps))
acc = [byp[p][0] / byp[p][1] for p in ps]
se = [math.sqrt(a * (1 - a) / byp[p][1]) for a, p in zip(acc, ps)]

fig, ax = plt.subplots(figsize=(9, 5.6))
ax.errorbar(x, acc, yerr=se, fmt="-o", color="#b5179e", lw=2.4, ms=9, capsize=5,
            label="Sonnet soundness (independent, no gold)")
ax.axhline(0.89, ls=":", color="#c85200", lw=2,
           label="Qwen self-judge (~0.89, anchored/lenient)")
ax.set_xticks(x); ax.set_xticklabels([str(p) for p in ps])
ax.set_xlabel("nucleus sampling  top_p", fontsize=12)
ax.set_ylabel("fraction of CoTs judged sound", fontsize=12)
ax.set_ylim(0.45, 0.95); ax.grid(alpha=0.3); ax.legend(fontsize=10, loc="center right")
ax.set_title("Independent Sonnet judge: CoT soundness is FLAT vs top_p (~0.61)\n"
             "40 CoTs/top_p, no gold shown — and far stricter than the Qwen self-judge (~0.89)",
             fontsize=10.5)
fig.tight_layout(); fig.savefig("outputs/cot_sonnet_soundness.png", dpi=150, bbox_inches="tight")
print("saved outputs/cot_sonnet_soundness.png")
