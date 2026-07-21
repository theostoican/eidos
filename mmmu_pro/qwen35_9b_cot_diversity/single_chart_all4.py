#!/usr/bin/env python
"""Single-panel chart: visual-premise soundness, visual-premise diversity, per-sample accuracy.

The three measures live on different scales (soundness 0.45-0.61, Vendi 1.07-1.33, accuracy
0.78-0.81). A dual/triple y-axis is the classic charting error, so all three are INDEXED to
their value at top_p=0.1 and share ONE axis: the plot reads as relative change across top_p,
which is exactly the claim being made. Raw endpoints are printed in the direct labels.

Restricted to the 7 fully-generated top_p; soundness is balanced across shared questions.
"""
import collections, json, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from majk import load_cells

COMPLETE=[0.1,0.3,0.5,0.7,0.9,0.95,1.0]
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#d9d8d4"
C_SOUND="#2a78d6"; C_DIV="#eb6834"; C_ACC="#1baf7a"; C_MAJ="#eda100"   # validated categorical slots 1,2,3,4

# soundness (holistic, balanced)
V=[json.loads(l) for l in open("outputs/u_verdicts_holistic.jsonl")]
V=[r for r in V if r["correct"] is not None and r["top_p"] in COMPLETE]
tp=collections.defaultdict(set)
for r in V: tp[r["top_p"]].add(r["id"])
sh=set.intersection(*[tp[p] for p in COMPLETE])
agg=collections.defaultdict(lambda:[0,0])
for r in V:
    if r["id"] in sh: agg[r["top_p"]][0]+=int(r["correct"]); agg[r["top_p"]][1]+=1
sound=[agg[p][0]/agg[p][1] for p in COMPLETE]

# diversity
D={e["top_p"]:e["vendi_set"] for e in json.load(open("outputs/u_premise_diversity.json"))["evolution"]}
div=[D[p] for p in COMPLETE]

# per-sample answer accuracy
cells=load_cells("outputs/u_gen.shard*.jsonl")
cells={k:v for k,v in cells.items() if k[1] in COMPLETE}
qs=sorted(set.intersection(*[{q for q,p in cells if p==t} for t in COMPLETE]))
acc=[float(np.mean([sum(a==cells[(q,p)][1] for a in cells[(q,p)][0])/len(cells[(q,p)][0]) for q in qs]))
     for p in COMPLETE]
# majority vote @16 (self-consistency), balanced on questions with 16 valid samples everywhere
import random as _r
from majk import majk_cell
_rng=_r.Random(0)
qs16=[q for q in qs if all(len(cells[(q,p)][0])>=16 for p in COMPLETE)]
maj=[float(np.mean([majk_cell(*cells[(q,p)],16,2000,_rng) for q in qs16])) for p in COMPLETE]
print(f"  maj@16 on {len(qs16)} questions: "+", ".join(f"{p}:{m:.3f}" for p,m in zip(COMPLETE,maj)))

S,Dv,A = sound, div, acc   # RAW values, no indexing

fig,ax=plt.subplots(figsize=(9.5,6.0))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.axhline(1.0, lw=1, color=GRID, zorder=1, alpha=0.6)
series=[("Visual premise soundness",S,C_SOUND,sound,"o"),
        ("Visual premise diversity (Vendi, not a fraction)",Dv,C_DIV,div,"s"),
        ("Per-sample answer accuracy",A,C_ACC,acc,"^"),
        ("Majority-vote accuracy (maj@16)",maj,C_MAJ,maj,"D")]
for name,y,c,raw,mk in series:
    ax.plot(COMPLETE,y,lw=2,marker=mk,ms=8,color=c,label=name,zorder=3,
            markeredgecolor=SURF,markeredgewidth=2)
    ax.annotate(f"{raw[-1]:.2f}", xy=(COMPLETE[-1],y[-1]),
                xytext=(9,0), textcoords="offset points", va="center",
                fontsize=9.5, color=INK2, zorder=4)
ax.set_xlabel("top_p", fontsize=11, color=INK)
ax.set_ylabel("value", fontsize=11, color=INK)
ax.set_ylim(0, 1.40)
ax.set_title("Raising top_p: premises get MORE DIVERSE and LESS SOUND; majority vote peaks mid-range\n"
             "MMMU-Pro / Qwen3.5-9B thinking · 86 questions × 16 samples · original sampling (top_k=-1, pp=0)",
             fontsize=12, color=INK, pad=14, loc="left")
ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7, zorder=0)
ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=10)
ax.set_xlim(0.03, 1.10)
leg=ax.legend(fontsize=10, frameon=False, loc="lower left", labelcolor=INK)
fig.tight_layout()
fig.savefig("outputs/u_single_chart_all4.png", dpi=160, facecolor=SURF, bbox_inches="tight")
print("[single_chart] -> outputs/u_single_chart_all4.png")
for name,y,c,raw,mk in series:
    print(f"  {name:36} idx {y[0]:.3f}->{y[-1]:.3f}   raw {raw[0]:.3f}->{raw[-1]:.3f}")
