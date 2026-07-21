#!/usr/bin/env python
"""Fail-fast inverted-U check on the ORIGINAL-sampling run (u_gen.*), at maj@6 (prior run's k)
and maj@16. Compares directly against the prior committed inverted-U:
  prior maj@6:  0.5=.765  0.7=.780*  0.9=.763  0.95=.740  1.0=.746   (peak 0.7, 1.0<0.7)
"""
import collections, random
import numpy as np
from majk import load_cells, majk_cell

rng=random.Random(0)
cells=load_cells("outputs/u_gen.shard*.jsonl")
if not cells:
    print("[verify] no u_gen data yet"); raise SystemExit
tops=sorted({p for _,p in cells})
prior={0.5:.7647,0.7:.7803,0.9:.7630,0.95:.7399,1.0:.7457}
print(f"[verify] top_p available: {tops}")
for k in [6,16]:
    print(f"\n=== maj@{k} ===")
    qs=sorted(set.intersection(*[{q for q,pp in cells if pp==p} for p in tops])) if len(tops)>1 else \
       [q for q,_ in cells]
    qk=[q for q in qs if all(len(cells[(q,p)][0])>=k for p in tops)]
    print(f"{'top_p':>6} {'maj@'+str(k):>8} {'prior':>8} {'delta':>8}")
    m={}
    for p in tops:
        vals=[majk_cell(*cells[(q,p)],k,1500,rng) for q in qk]
        m[p]=float(np.mean(vals)) if vals else float('nan')
        pr=prior.get(p)
        d=f"{m[p]-pr:+.4f}" if pr else "  -"
        print(f"{p:>6} {m[p]:>8.4f} {(f'{pr:.4f}' if pr else '  -'):>8} {d:>8}")
    if len(tops)>=3:
        peak=max(tops,key=lambda p:m[p])
        interior = peak not in (min(tops),max(tops))
        v1=m.get(1.0); vpk=m[peak]
        print(f"  peak@top_p={peak} ({'INTERIOR - inverted-U!' if interior else 'EDGE'})", end="")
        if v1 is not None and peak!=1.0:
            print(f" | 1.0={v1:.4f} {'<' if v1<vpk else '>='} peak={vpk:.4f} "
                  f"{'(U CONFIRMED: 1.0 below peak)' if v1<vpk else '(not yet)'}")
        else: print()
    print(f"  (nQ={len(qk)})")
