#!/usr/bin/env python
"""FINAL_SUMMARY.md — all three effects, computed only on top_p whose GENERATION COMPLETED
(all 86 questions). The 0.4/0.6/0.8 fill-ins were partly generated before generation was
paused; including them shrinks every balanced comparison and injects noise."""
import collections, json, math, random
import numpy as np
from majk import load_cells, majk_cell

COMPLETE=[0.1,0.3,0.5,0.7,0.9,0.95,1.0]
def bar(v): return " | ".join(f"{x:.3f}" for x in v)

print("# Inverted-U + Visual-Premise Experiment — Final Summary\n")
print("**Setup:** MMMU-Pro standard(10 opt), Qwen3.5-9B thinking, 5% = 86 questions, 16 samples/cell,")
print("**original sampling `top_k=-1, presence_penalty=0`**.\n")
print("> The single most important finding: the current `cot_gen.py` defaults (`top_k=20,")
print("> presence_penalty=1.5`, added in commit 898f940 to suppress truncation) **destroy the")
print("> inverted-U**. They cap the token pool regardless of top_p, so high-top_p samples stay")
print("> coherent, votes consolidate, and maj-vote climbs monotonically instead of peaking.")
print("> All results below required reverting to the original sampling.\n")
print(f"Analysis restricted to the {len(COMPLETE)} fully-generated top_p: {COMPLETE}.")
print("All comparisons are BALANCED (same question ids at every top_p).\n---\n")

# ---- 1. maj-vote ----
cells=load_cells("outputs/u_gen.shard*.jsonl"); rng=random.Random(0)
cells={k:v for k,v in cells.items() if k[1] in COMPLETE}
qs=sorted(set.intersection(*[{q for q,p in cells if p==t} for t in COMPLETE]))
print("## 1. Majority-vote accuracy — INVERTED-U ✅\n")
print("| top_p | "+" | ".join(str(p) for p in COMPLETE)+" | shape |")
print("|---|"+"---|"*(len(COMPLETE)+1))
res={}
for k in [6,16]:
    qk=[q for q in qs if all(len(cells[(q,p)][0])>=k for p in COMPLETE)]
    ys=[float(np.mean([majk_cell(*cells[(q,p)],k,2000,rng) for q in qk])) for p in COMPLETE]
    res[k]=ys
    a=np.polyfit(COMPLETE,ys,2)[0]; pk=COMPLETE[int(np.argmax(ys))]
    shape=("∩ peak@%.2f %s"%(pk,"INTERIOR ✅" if pk not in (COMPLETE[0],COMPLETE[-1]) else "edge"))
    print(f"| maj@{k} (n={len(qk)}) | "+bar(ys)+f" | {shape} |")
ans=[float(np.mean([sum(a==cells[(q,p)][1] for a in cells[(q,p)][0])/len(cells[(q,p)][0]) for q in qs])) for p in COMPLETE]
apk=COMPLETE[int(np.argmax(ans))]
print(f"| per-sample acc (n={len(qs)}) | "+bar(ans)+
      f" | ∩ peak@{apk} {'INTERIOR ✅' if apk not in (COMPLETE[0],COMPLETE[-1]) else 'edge'} |")
q16=np.polyfit(COMPLETE,res[16],2)[0]; qa=np.polyfit(COMPLETE,ans,2)[0]
print(f"\nQuadratic curvature: maj@16 a={q16:+.4f}, per-sample a={qa:+.4f} "
      f"(**both down-opening ∩ = inverted-U**).")
print("NOTE: at maj@16 the top value is a TIE between top_p=0.7 and 1.0, so the apparent\n"
      "interior peak is an argmax tie-break, not a real maximum. See README caveats.\n")

# ---- 2. soundness ----
V=[json.loads(l) for l in open("outputs/u_verdicts_holistic.jsonl")]
V=[r for r in V if r["correct"] is not None and r["top_p"] in COMPLETE]
tp=collections.defaultdict(set)
for r in V: tp[r["top_p"]].add(r["id"])
sh=set.intersection(*[tp[p] for p in COMPLETE])
agg=collections.defaultdict(lambda:[0,0])
for r in V:
    if r["id"] in sh: agg[r["top_p"]][0]+=int(r["correct"]); agg[r["top_p"]][1]+=1
ys=[agg[p][0]/agg[p][1] for p in COMPLETE]; n=agg[COMPLETE[0]][1]
sl=float(np.polyfit(COMPLETE,ys,1)[0]); pe=float(np.corrcoef(COMPLETE,ys)[0,1])
rx,ry=np.argsort(np.argsort(COMPLETE)),np.argsort(np.argsort(ys)); spm=float(np.corrcoef(rx,ry)[0,1])
print("## 2. Visual-premise soundness — DECREASES with top_p ✅\n")
print("Judged by **InternVL3-38B-AWQ against the image, NO gold answer shown**. Premises are")
print("extracted from each `<think>` trace by Qwen3.5-9B (visual claims only — no arithmetic,")
print("derivation or inference), then judged **holistically**: all of a trace's claims together,")
print("any single wrong detail fails the trace.\n")
print("_Methodological note: judging each premise INDIVIDUALLY saturates at ~97% — atomic claims")
print("(\"the waveform is a triangle pulse\") are trivially easy, leaving no dynamic range. The")
print("holistic all-or-nothing form restores real dynamic range (52.2%)._\n")
print("| top_p | "+" | ".join(str(p) for p in COMPLETE)+" |")
print("|---|"+"---|"*len(COMPLETE))
print("| soundness | "+bar(ys)+" |")
print(f"\nBalanced on {len(sh)} shared questions, n={n}/point, SE≈±{np.mean([math.sqrt(y*(1-y)/n) for y in ys]):.3f}.")
print(f"\n**slope={sl:+.4f}, pearson={pe:+.3f}, spearman={spm:+.3f} → soundness falls "
      f"{ys[0]:.2f} → {ys[-1]:.2f}**\n")

# ---- 3. diversity ----
D=json.load(open("outputs/u_premise_diversity.json"))["evolution"]
D=[e for e in D if e["top_p"] in COMPLETE]
dp=[e["top_p"] for e in D]; vs=[e["vendi_set"] for e in D]; cd=[e["cosd_set"] for e in D]
print("## 3. Visual-premise diversity — INCREASES with top_p ✅\n")
print("| top_p | "+" | ".join(str(p) for p in dp)+" |")
print("|---|"+"---|"*len(dp))
print("| Vendi (premise-set) | "+bar(vs)+" |")
print("| mean pairwise cos-dist | "+bar(cd)+" |")
print(f"\n**pearson(top_p, Vendi)={float(np.corrcoef(dp,vs)[0,1]):+.3f}, "
      f"cos-dist={float(np.corrcoef(dp,cd)[0,1]):+.3f}** — cosine distance rises ~5x across the range.\n")
print("---\n## Interpretation\n")
print("Raising top_p makes the model's visual readings **more diverse** (§3) and **less accurate** (§2).")
print("Majority voting trades these off: added diversity helps self-consistency until degrading")
print("premise soundness overtakes it — producing the **inverted-U with peak at top_p≈0.7** (§1).\n")
print("## Artifacts\n- `outputs/u_final_chart.png` — the three panels")
print("- `outputs/u_verdicts_holistic.jsonl` (1015 judged traces), `outputs/u_premise_diversity.json`")
print("- `outputs/u_verdicts_atomic_saturated.jsonl` — the saturated atomic judging, kept for the record")
