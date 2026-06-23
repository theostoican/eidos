#!/usr/bin/env python
"""Two figures for the comprehensive-premise analysis:
 comp_diversity_vs_correctness.png : (L) Vendi vs frac_correct scatter; (R) accuracy vs top_p.
 harmonic_per_example_comp.png     : per-example harmonic mean of correctness & diversity."""
import json, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

rep = json.load(open("outputs/premise_report_comp.json")); cells = rep["per_cell"]; ps = rep["top_ps"]
byq = collections.defaultdict(dict); subj = {}
for c in cells: byq[c["id"]][c["top_p"]] = c; subj[c["id"]] = c["subject"]
qids = sorted(byq); cmap = plt.get_cmap("tab10"); col = {q: cmap(i) for i, q in enumerate(qids)}
def lbl(q): return f"{subj[q]} ({q.replace('test_','').replace('validation_','val_')})"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
for q in qids:
    ax1.scatter([byq[q][p]["vendi"] for p in ps], [byq[q][p]["frac_correct"] for p in ps], color=col[q], s=55, label=lbl(q))
allx = np.array([c["vendi"] for c in cells]); ally = np.array([c["frac_correct"] for c in cells])
m, b = np.polyfit(allx, ally, 1); xx = np.linspace(allx.min(), allx.max(), 50)
ax1.plot(xx, m * xx + b, "k--", lw=1.5, alpha=0.7, label=f"trend (Pearson {rep['correlations']['vendi__frac_correct']['pearson']:+.2f})")
ax1.set_xlabel("Vendi diversity (16 premises / cell)"); ax1.set_ylabel("fraction correct")
ax1.set_title("Comprehensive premise: diversity vs correctness"); ax1.grid(alpha=0.3); ax1.legend(fontsize=7)
for q in qids:
    ax2.plot(ps, [byq[q][p]["frac_correct"] for p in ps], "-o", color=col[q], lw=2, ms=6, label=lbl(q))
ax2.set_xlabel("nucleus top_p"); ax2.set_ylabel("fraction correct"); ax2.set_xticks(ps)
ax2.set_title("Comprehensive premise: accuracy vs top_p"); ax2.grid(alpha=0.3); ax2.legend(fontsize=7)
fig.tight_layout(); fig.savefig("outputs/comp_diversity_vs_correctness.png", dpi=150, bbox_inches="tight")
print("saved comp_diversity_vs_correctness.png")

def dn(c): return (c["vendi"] - 1.0) / (c["n"] - 1.0)
def hm(c, d): return 0.0 if (c + d) == 0 else 2 * c * d / (c + d)
byh = collections.defaultdict(dict)
for c in cells: byh[c["id"]][c["top_p"]] = hm(c["frac_correct"], dn(c))
fig, ax = plt.subplots(figsize=(11, 5.6))
for i, q in enumerate(qids):
    ax.plot(ps, [byh[q][p] for p in ps], "-o", color=cmap(i), lw=2, ms=6, label=lbl(q))
ax.set_xlabel("nucleus sampling top-p"); ax.set_ylabel("harmonic mean  2·c·d/(c+d)")
ax.set_title("Comprehensive premise — harmonic mean of correctness & diversity per example")
ax.set_xticks(ps); ax.set_ylim(bottom=0); ax.grid(alpha=0.3)
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, title="example")
fig.tight_layout(); fig.savefig("outputs/harmonic_per_example_comp.png", dpi=150, bbox_inches="tight")
print("saved harmonic_per_example_comp.png")
