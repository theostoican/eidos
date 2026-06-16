#!/usr/bin/env python
"""Plot the harmonic mean of correctness & diversity vs top_p, ONE LINE PER EXAMPLE.

harmonic mean  H = 2*c*d / (c + d)   (0 if c+d == 0)
  c = frac_correct                       in [0,1]
  d = (vendi - 1)/(N - 1)  (Vendi norm)  in [0,1]
"""
import json, collections, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/mmmupro_qwen3vl/outputs/nucleus_report.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/mmmupro_qwen3vl/outputs/harmonic_per_example_vs_topp.png"
rep = json.load(open(REPORT))
cells = rep["per_cell"]
ps = rep["top_ps"]
N = max(c.get("n", 8) for c in cells)   # samples per cell (8 for v1/v2, 16 for v3)

def div_norm(v):
    return (v - 1.0) / (N - 1.0)
def hmean(c, d):
    return 0.0 if (c + d) == 0 else 2.0 * c * d / (c + d)

# per example: top_p -> harmonic mean
byq = collections.defaultdict(dict)
subj = {}
for c in cells:
    h = hmean(c["frac_correct"], div_norm(c["vendi"]))
    byq[c["id"]][c["top_p"]] = h
    subj[c["id"]] = c["subject"]

fig, ax = plt.subplots(figsize=(11, 5.4))
cmap = plt.get_cmap("tab10")
for i, qid in enumerate(sorted(byq)):
    ys = [byq[qid][p] for p in ps]
    ax.plot(ps, ys, "-", color=cmap(i), lw=2, marker="o", ms=6,
            label=f"{subj[qid]}  ({qid.replace('test_', '')})")

ax.set_xlabel("nucleus sampling top-p  (temperature=1.0, top-k off)", fontsize=11)
ax.set_ylabel("harmonic mean  2·c·d/(c+d)", fontsize=11)
ax.set_title("Qwen3.5-9B on MMMU-Pro failures — harmonic mean of correctness & diversity\n"
             f"per example ({N} samples/cell)", fontsize=12)
ax.set_xticks(ps)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.0,
          framealpha=0.9, title="example", title_fontsize=7,
          labelspacing=0.25, handlelength=1.4, borderpad=0.3, ncol=2)
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
print("\ntop_p:", ps)
for qid in sorted(byq):
    print(f"  {subj[qid]:>22} {qid.replace('test_',''):>20}: "
          f"{[round(byq[qid][p], 3) for p in ps]}")
