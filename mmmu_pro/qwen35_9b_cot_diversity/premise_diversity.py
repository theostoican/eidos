#!/usr/bin/env python
"""Visual-premise DIVERSITY vs top_p (independent of judging).

Per (question, top_p) cell, over that cell's traces:
  vendi_set  : Vendi over one mean-pooled embedding per TRACE's premise set
               (how differently the model *frames* its visual reading across samples)
  vendi_prem : Vendi over every individual premise in the cell
               (how varied the individual visual claims are)
  cosd_*     : mean pairwise cosine distance counterparts
Reuses vendi/cosd/_minilm_encode from analyze_cot.py so numbers stay comparable.
"""
import argparse, collections, json
import numpy as np
from analyze_cot import vendi, cosd, corr, _minilm_encode

ap=argparse.ArgumentParser()
ap.add_argument("--premises", default="outputs/u_premises.jsonl")
ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--out", default="outputs/u_premise_diversity.json")
a=ap.parse_args()

rows=[json.loads(l) for l in open(a.premises)]
rows=[r for r in rows if r["n_premises"]>0]
allp,owner=[],[]
for r in rows:
    for p in r["premises"]:
        allp.append(p); owner.append((r["id"],r["top_p"],r["sample_idx"]))
print(f"[div] embedding {len(allp)} premises from {len(rows)} traces ...", flush=True)
emb=_minilm_encode(allp, a.embed_model)
by=collections.defaultdict(list)
for e,k in zip(emb,owner): by[k].append(e)

cells=collections.defaultdict(list)
for r in rows: cells[(r["id"],r["top_p"])].append(r["sample_idx"])
per=[]
for (qid,tp),sidx in cells.items():
    keys=[(qid,tp,s) for s in sidx if (qid,tp,s) in by]
    if len(keys)<2: continue
    setv=np.array([np.mean(by[k],axis=0) for k in keys])
    prem=np.array([e for k in keys for e in by[k]])
    per.append({"id":qid,"top_p":tp,"n_traces":len(keys),
                "vendi_set":vendi(setv),"cosd_set":cosd(setv),
                "vendi_prem":vendi(prem) if len(prem)>1 else None,
                "cosd_prem":cosd(prem) if len(prem)>1 else None,
                "mean_premises":float(np.mean([len(by[k]) for k in keys]))})
tps=sorted({c["top_p"] for c in per})
ev=[]
def m(v):
    v=[x for x in v if x is not None]; return float(np.mean(v)) if v else None
print(f"\n{'top_p':>6} {'cells':>6} {'vendi_set':>10} {'cosd_set':>9} {'vendi_prem':>11} {'prem/trace':>11}")
for t in tps:
    cc=[c for c in per if c["top_p"]==t]
    e={"top_p":t,"n_cells":len(cc),"vendi_set":m([c["vendi_set"] for c in cc]),
       "cosd_set":m([c["cosd_set"] for c in cc]),"vendi_prem":m([c["vendi_prem"] for c in cc]),
       "cosd_prem":m([c["cosd_prem"] for c in cc]),"mean_premises":m([c["mean_premises"] for c in cc])}
    ev.append(e)
    print(f"{t:>6} {e['n_cells']:>6} {e['vendi_set']:>10.4f} {e['cosd_set']:>9.4f} "
          f"{e['vendi_prem']:>11.4f} {e['mean_premises']:>11.2f}")
xs=[c["top_p"] for c in per]
for k in ["vendi_set","cosd_set","vendi_prem"]:
    pe,sp=corr(xs,[c[k] for c in per])
    print(f"  corr(top_p, {k:>10}) = pearson {pe:+.3f}  spearman {sp:+.3f}")
json.dump({"evolution":ev,"per_cell":per},open(a.out,"w"),indent=1)
print(f"\n[div] -> {a.out}")
