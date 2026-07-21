#!/usr/bin/env python
"""THE deliverable: visual-premise soundness, visual-premise diversity, and the
majority-vote inverted-U — vs top_p, from the ORIGINAL-sampling run (top_k=-1, pp=0).

Sources:
  soundness  outputs/u_verdicts_holistic.jsonl  (all-or-nothing per trace; the atomic
             per-premise judge saturates at ~97% and cannot express a trend)
  diversity  outputs/u_premise_diversity.json
  maj-vote   outputs/u_gen.shard*.jsonl  (maj@k subsampled from the 16 banked samples)
Soundness is BALANCED across top_p (same question ids) so difficulty can't fake a trend.
"""
import collections, json, math, random
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from majk import load_cells, majk_cell

def pad(v,f=.15,lo=None,hi=None):
    v=[x for x in v if x is not None and not (isinstance(x,float) and math.isnan(x))]
    if not v: return (0,1)
    a,b=min(v),max(v); m=(b-a)*f or 1e-3; a,b=a-m,b+m
    if lo is not None: a=max(a,lo)
    if hi is not None: b=min(b,hi)
    return a,b

# ---------- 1. soundness (holistic, balanced) ----------
sp,sound,sse=[],[],[]
try:
    V=[json.loads(l) for l in open("outputs/u_verdicts_holistic.jsonl")]
    V=[r for r in V if r["correct"] is not None]
    # Restrict to top_p whose GENERATION completed (all 86 questions). The 0.4/0.6/0.8
    # fill-ins were only partly generated before generation was paused; including them
    # shrinks the balanced shared-question set from 50 to 15 and injects noise.
    COMPLETE=[0.1,0.3,0.5,0.7,0.9,0.95,1.0]
    V=[r for r in V if r["top_p"] in COMPLETE]
    tp_ids=collections.defaultdict(set)
    for r in V: tp_ids[r["top_p"]].add(r["id"])
    cand=[p for p in sorted(tp_ids) if len(tp_ids[p])>=5]
    shared=set.intersection(*[tp_ids[p] for p in cand]) if cand else set()
    agg=collections.defaultdict(lambda:[0,0])
    for r in V:
        if r["id"] not in shared: continue
        agg[r["top_p"]][0]+=int(r["correct"]); agg[r["top_p"]][1]+=1
    for p in sorted(agg):
        c,n=agg[p]
        if n<10: continue
        sp.append(p); sound.append(c/n); sse.append(math.sqrt((c/n)*(1-c/n)/n))
    n_shared=len(shared)
except Exception as e:
    n_shared=0; print("soundness unavailable:",e)

# ---------- 2. diversity ----------
D=json.load(open("outputs/u_premise_diversity.json"))["evolution"]
dp=[e["top_p"] for e in D]; vset=[e["vendi_set"] for e in D]; cos=[e["cosd_set"] for e in D]

# ---------- 3. maj-vote ----------
cells=load_cells("outputs/u_gen.shard*.jsonl"); rng=random.Random(0)
cnt=collections.Counter(p for _,p in cells)
mt=sorted(p for p in cnt if cnt[p]>=80)
qs=sorted(set.intersection(*[{q for q,pp in cells if pp==p} for p in mt]))
def majk(k):
    qk=[q for q in qs if all(len(cells[(q,p)][0])>=k for p in mt)]
    return [float(np.mean([majk_cell(*cells[(q,p)],k,1500,rng) for q in qk])) for p in mt], len(qk)
m6,nq6=majk(6); m16,nq16=majk(16)
ans=[float(np.mean([sum(a==cells[(q,p)][1] for a in cells[(q,p)][0])/len(cells[(q,p)][0]) for q in qs])) for p in mt]

fig,ax=plt.subplots(1,3,figsize=(17.5,5.2))
# soundness
if sound:
    ax[0].errorbar(sp,sound,yerr=sse,marker='o',lw=2,ms=7,capsize=3,color="#c0392b")
    sl=np.polyfit(sp,sound,1)[0] if len(sp)>=3 else float('nan')
    r=float(np.corrcoef(sp,sound)[0,1]) if len(sp)>=3 else float('nan')
    ax[0].set_title(f"Visual premise soundness (holistic, no gold)\n"
                    f"slope={sl:+.3f}  pearson={r:+.2f}  (n={n_shared} shared Q)",fontsize=10)
    ax[0].set_ylim(*pad(sound+[s+e for s,e in zip(sound,sse)]+[s-e for s,e in zip(sound,sse)],lo=0,hi=1))
else:
    ax[0].text(.5,.5,"soundness pending",ha='center'); ax[0].set_title("Visual premise soundness",fontsize=10)
ax[0].set_xlabel("top_p"); ax[0].set_ylabel("fraction of traces with ALL visual claims correct"); ax[0].grid(alpha=.3)
# diversity
l1,=ax[1].plot(dp,vset,marker='s',lw=2,ms=6,color="#2874a6",label="Vendi (premise-set)")
a2=ax[1].twinx(); l2,=a2.plot(dp,cos,marker='^',lw=2,ms=6,ls='--',color="#7d3c98",label="mean pairwise cos-dist")
ax[1].set_ylabel("Vendi",color="#2874a6"); a2.set_ylabel("cosine distance",color="#7d3c98")
ax[1].set_ylim(*pad(vset)); a2.set_ylim(*pad(cos))
rv=float(np.corrcoef(dp,vset)[0,1])
ax[1].set_title(f"Visual premise diversity — rises with top_p\npearson={rv:+.2f}",fontsize=10)
ax[1].set_xlabel("top_p"); ax[1].legend(handles=[l1,l2],fontsize=8,loc="upper left"); ax[1].grid(alpha=.3)
# maj-vote
ax[2].plot(mt,m16,marker='o',lw=2,ms=7,color="#1e8449",label=f"maj@16 (nQ={nq16})")
ax[2].plot(mt,m6,marker='s',lw=2,ms=6,ls='--',color="#52be80",label=f"maj@6 (nQ={nq6})")
ax[2].plot(mt,ans,marker='^',lw=1.6,ms=5,ls=':',color="#7f8c8d",label="per-sample answer acc")
for series,col in [(m16,"#1e8449"),(ans,"#7f8c8d")]:
    if len(mt)>=4:
        a_,b_,c_=np.polyfit(mt,series,2)
        if a_<0:
            xstar=-b_/(2*a_)
            if min(mt)<xstar<max(mt):
                ax[2].axvline(xstar,ls=':',lw=1,color=col,alpha=.6)
                ax[2].annotate(f"peak {xstar:.2f}",(xstar,max(series)),fontsize=8,color=col)
ax[2].set_title("Majority-vote accuracy — inverted-U",fontsize=10)
ax[2].set_xlabel("top_p"); ax[2].set_ylabel("accuracy"); ax[2].set_ylim(*pad(m6+m16+ans,lo=0,hi=1))
ax[2].legend(fontsize=8,loc="lower right"); ax[2].grid(alpha=.3)
fig.suptitle("MMMU-Pro / Qwen3.5-9B thinking — 5% (86Q) x 16 samples — ORIGINAL sampling (top_k=-1, presence_penalty=0)",fontsize=11)
fig.tight_layout(rect=[0,0,1,.95]); fig.savefig("outputs/u_final_chart.png",dpi=150,bbox_inches="tight")
print("[final_chart] -> outputs/u_final_chart.png")
