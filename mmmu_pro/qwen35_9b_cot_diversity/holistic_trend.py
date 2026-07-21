#!/usr/bin/env python
"""Holistic premise soundness vs top_p — the measurement with real dynamic range.

Atomic per-premise judging saturates (~97%): each claim is individually trivial. Judging a
trace's visual claims as ONE composite (any wrong detail => INCORRECT) reproduces the
sibling experiment's scale (~56%) and can actually express a top_p effect.

BALANCED by default: only question ids judged at every compared top_p contribute, so
question-difficulty differences cannot masquerade as a trend.
"""
import argparse, collections, json, math
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--verdicts", default="outputs/u_verdicts_holistic.jsonl")
ap.add_argument("--min-n", type=int, default=15)
ap.add_argument("--unbalanced", dest="balanced", action="store_false"); ap.set_defaults(balanced=True)
a=ap.parse_args()

V=[json.loads(l) for l in open(a.verdicts)]
V=[r for r in V if r["correct"] is not None]
if not V: print("[holistic] no verdicts yet"); raise SystemExit

tp_ids=collections.defaultdict(set)
for r in V: tp_ids[r["top_p"]].add(r["id"])
cand=[p for p in sorted(tp_ids) if len(tp_ids[p])>=5]
shared=set.intersection(*[tp_ids[p] for p in cand]) if (a.balanced and cand) else None
if shared is not None:
    print(f"[balanced] {len(shared)} question ids judged at all of {cand}\n")

agg=collections.defaultdict(lambda:[0,0])
for r in V:
    if shared is not None and r["id"] not in shared: continue
    agg[r["top_p"]][0]+=int(r["correct"]); agg[r["top_p"]][1]+=1

ps=sorted(p for p in agg if agg[p][1]>=a.min_n)
if len(ps)<2: print(f"[holistic] only {len(ps)} top_p with >={a.min_n} traces; wait for more"); raise SystemExit
print(f"{'top_p':>6} {'correct':>9} {'n':>6} {'rate':>8} {'SE':>7}")
print("-"*42)
xs,ys=[],[]
for p in ps:
    c,n=agg[p]; r=c/n; se=math.sqrt(r*(1-r)/n)
    xs.append(p); ys.append(r)
    print(f"{p:>6} {c:>9} {n:>6} {r:>8.4f} {se:>7.4f}")
if len(xs)>=3:
    slope=float(np.polyfit(xs,ys,1)[0]); pe=float(np.corrcoef(xs,ys)[0,1])
    rx,ry=np.argsort(np.argsort(xs)),np.argsort(np.argsort(ys))
    sp=float(np.corrcoef(rx,ry)[0,1])
    print(f"\nslope={slope:+.4f}  pearson={pe:+.3f}  spearman={sp:+.3f}")
    print(f"-> premise soundness {'DECREASES with top_p ✅' if slope<0 else 'does NOT decrease ❌'}")
    print(f"   (sibling target: pearson −0.15, ~0.59→0.50)")
