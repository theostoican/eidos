#!/usr/bin/env python
"""Every arm vs the card baseline at maj@1 and maj@8, paired on the frozen question set.
An arm counts a question only if its cell (8 ballots) and the baseline cell are complete."""
import json, collections, random, sys
sys.path.insert(0, "../common")
from analyze import parse_answer, majk_cell
import numpy as np
from scipy import stats

gold = {}
def load(path):
    c = collections.defaultdict(list)
    try: f = open(path)
    except FileNotFoundError: return c
    for l in f:
        try: r = json.loads(l)
        except Exception: continue
        a = parse_answer(r["text"], r["n_options"]) if r["finish_reason"] == "stop" else None
        c[(r["id"], r["top_p"])].append(a); gold[r["id"]] = r["gold"]
    return c

bl = load("outputs/baseline_card.jsonl")
t12c, t16, t12p, sw, t13c, t14c, t12c1, t11c, t10c1, t11c1 = (load(f"outputs/{f}.jsonl") for f in
    ("t12_card", "t16_full", "t12_penalty", "t12_sweep", "t13_card", "t14_card", "t12_card_p1",
     "t11_card", "t10_card_p1", "t11_card_p1"))
arms = [("T1.0 top_k=20 presence1.5 p1.0", t10c1, 1.0),
        ("T1.1 top_k=20 presence1.5 p.95", t11c, 0.95), ("T1.1 top_k=20 presence1.5 p1.0", t11c1, 1.0),
        ("T1.2 top_k=20 presence1.5 p.95", t12c, 0.95), ("T1.2 top_k=20 presence1.5 p1.0", t12c1, 1.0),
        ("T1.3 top_k=20 presence1.5 p.95", t13c, 0.95), ("T1.4 top_k=20 presence1.5 p.95", t14c, 0.95)]
arms += [(f"T1.2 top_k=-1 presence1.5 p{p}", sw, p) for p in (0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)]
arms += [("T1.2 top_k=-1 presence1.5 p.95", t12p, 0.95)]
arms += [(f"T1.6 top_k=-1 presence0 p{p}", t16, p) for p in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
frozen = set(json.load(open("outputs/baseline_qids.json")))
HOLDOUT = "--holdout" in sys.argv          # the 322 questions outside the frozen set: untouched by any selection
ALL = "--all" in sys.argv                  # every question with a baseline cell (frozen + holdout)
qids = set(gold) if ALL else (set(gold) - frozen) if HOLDOUT else frozen
rng = random.Random(0)
print(f"# All arms vs card baseline (T=1.0, top_p .95, top_k 20, presence 1.5), paired, "
      f"{'ALL questions (frozen + holdout)' if ALL else 'HOLDOUT (outside frozen set)' if HOLDOUT else 'frozen set'}\n")
print("| arm | n | maj@1 | card | d | p | maj@8 | card | d | p |")
print("|---|---|---|---|---|---|---|---|---|---|")
for name, c, p in arms:
    Q = [q for q in qids if len(c.get((q, p), [])) == 8 and len(bl.get((q, 0.95), [])) == 8]
    if len(Q) < 20: continue
    row = f"| {name} | {len(Q)}"
    for k in (1, 8):
        S = np.array([majk_cell(c[(q, p)], gold[q], k, 400, rng) for q in Q])
        B = np.array([majk_cell(bl[(q, 0.95)], gold[q], k, 400, rng) for q in Q])
        row += f" | {S.mean():.4f} | {B.mean():.4f} | {(S-B).mean()*100:+.2f} pp | {stats.ttest_rel(S, B).pvalue:.3f}"
    print(row + " |")
