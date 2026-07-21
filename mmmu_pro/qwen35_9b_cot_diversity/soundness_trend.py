#!/usr/bin/env python
"""Premise soundness vs top_p — reported BOTH over all premises and over SPECIFIC ones.

Why the split: extracted premises skew toward trivially-true structural claims ("the
waveform is a triangle pulse", "for t<0 the value is 0") that the model gets right ~always,
pushing aggregate soundness to a ~98% ceiling that compresses any top_p trend. The
discriminative errors live in SPECIFIC reads -- exact numbers, labels, quantities -- e.g.
the judge correctly flagged 'the "Nanling" row is 7442' as UNSOUND. Restricting to those
removes the ceiling and is where a real decline should appear.
"""
import argparse, collections, json, math, re
import numpy as np

SPECIFIC = re.compile(r"\d")          # contains a digit: a value/label/count read
def is_specific(p): return bool(SPECIFIC.search(p))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="outputs/u_verdicts.jsonl")
    ap.add_argument("--premises", default="outputs/u_premises.jsonl")
    ap.add_argument("--min-judged", type=int, default=30)
    ap.add_argument("--balanced", action="store_true", default=True,
                    help="restrict to questions judged at EVERY compared top_p. Without this, "
                         "each top_p is scored on a different question subset and difficulty "
                         "differences masquerade as a top_p trend (this confound flipped the "
                         "sign of the maj-vote result earlier).")
    ap.add_argument("--unbalanced", dest="balanced", action="store_false")
    a=ap.parse_args()
    P={(r["id"],r["top_p"],r["sample_idx"]):r["premises"] for r in (json.loads(l) for l in open(a.premises))}
    V=[json.loads(l) for l in open(a.verdicts)]
    # ---- balance: keep only question ids judged at every top_p that has enough data ----
    tp_ids=collections.defaultdict(set)
    for r in V: tp_ids[r["top_p"]].add(r["id"])
    cand=[p for p in tp_ids if len(tp_ids[p])>=5]
    shared=set.intersection(*[tp_ids[p] for p in cand]) if (a.balanced and cand) else None
    if shared is not None:
        print(f"[balanced] {len(shared)} question ids judged at all of {sorted(cand)}\n")
    agg=collections.defaultdict(lambda:[0,0]); spec=collections.defaultdict(lambda:[0,0])
    # trace-level metrics: conjunction (ALL premises sound) is the analogue of the sibling
    # experiment's single COMPREHENSIVE premise -- atomic premises are individually easy
    # (~97%), so the pooled rate ceilings out; the conjunction compounds errors and has
    # far more headroom to express a top_p effect.
    conj=collections.defaultdict(lambda:[0,0]); frac=collections.defaultdict(list)
    for r in V:
        if shared is not None and r["id"] not in shared: continue
        prem=P.get((r["id"],r["top_p"],r["sample_idx"]),[])
        for p,v in zip(prem,r["verdicts"]):
            if v is None: continue
            agg[r["top_p"]][0]+=int(v); agg[r["top_p"]][1]+=1
            if is_specific(p):
                spec[r["top_p"]][0]+=int(v); spec[r["top_p"]][1]+=1
        vs=[v for v in r["verdicts"] if v is not None]
        if vs:
            conj[r["top_p"]][0]+=int(all(vs)); conj[r["top_p"]][1]+=1
            frac[r["top_p"]].append(sum(vs)/len(vs))
    ps=sorted(k for k in agg if agg[k][1]>=a.min_judged)
    if not ps: print("[trend] not enough judged premises yet"); return
    print(f"{'top_p':>6} | {'POOLED premises':>17} | {'SPECIFIC(numeric)':>17} | {'ALL-SOUND trace':>17} | {'mean frac':>9}")
    print(f"{'':>6} | {'sound':>8} {'n':>8} | {'sound':>8} {'n':>8} | {'rate':>8} {'n':>8} | {'':>9}")
    print("-"*82)
    rows={}
    for p in ps:
        s,n=agg[p]; ss,sn=spec[p]
        f=s/n
        fs=ss/sn if sn else float('nan')
        cs,cn=conj[p]; cf=cs/cn if cn else float('nan')
        mf=float(np.mean(frac[p])) if frac[p] else float('nan')
        rows[p]=(f,fs,cf,mf)
        print(f"{p:>6} | {f:>8.4f} {n:>8} | {fs:>8.4f} {sn:>8} | {cf:>8.4f} {cn:>8} | {mf:>9.4f}")
    for lab,idx in [("POOLED",0),("SPECIFIC",1),("ALL-SOUND",2),("MEAN-FRAC",3)]:
        ys=[rows[p][idx] for p in ps]
        if len(ps)>=3 and not any(math.isnan(y) for y in ys):
            slope=float(np.polyfit(ps,ys,1)[0]); r=float(np.corrcoef(ps,ys)[0,1])
            print(f"\n{lab:>9}: slope={slope:+.4f}  pearson(top_p,soundness)={r:+.3f}  "
                  f"-> {'DECREASES ✅' if slope<0 else 'does not decrease ❌'}")
    print("\n(sibling premise_diversity target: pearson −0.15, soundness ~0.59→0.50)")

if __name__=="__main__": main()
