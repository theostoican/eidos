#!/usr/bin/env python
"""Mixed-sampler 8-ballot votes vs 8 card ballots, on the frozen set, SPLIT-HALF:
the best mix is chosen on half A (seeded random split) and reported on half B, paired against
the card's own maj@8 on half B. Chains are taken by sample_idx (card chains first), so the
budget is always exactly 8 samples per question. Arms need full frozen-set coverage."""
import json, collections, random, sys, itertools
sys.path.insert(0, "../common")
from analyze import parse_answer
import numpy as np
from scipy import stats

gold = {}
def load(path, p):
    c = collections.defaultdict(dict)
    try: f = open(path)
    except FileNotFoundError: return c
    for l in f:
        try: r = json.loads(l)
        except Exception: continue
        if r["top_p"] != p: continue
        a = parse_answer(r["text"], r["n_options"]) if r["finish_reason"] == "stop" else None
        c[r["id"]][r["sample_idx"]] = a; gold[r["id"]] = r["gold"]
    return {q: [d[i] for i in range(8)] for q, d in c.items() if len(d) == 8}

arms = {"card": load("outputs/baseline_card.jsonl", 0.95),
        "T1.2k20": load("outputs/t12_card.jsonl", 0.95),
        "T1.2p.7": load("outputs/t12_sweep.jsonl", 0.7),
        "T1.2p.8": load("outputs/t12_sweep.jsonl", 0.8),
        "T1.2k20p1": load("outputs/t12_card_p1.jsonl", 1.0),
        "T1.1k20": load("outputs/t11_card.jsonl", 0.95),
        "T1.1k20p1": load("outputs/t11_card_p1.jsonl", 1.0),
        "T1.0k20p1": load("outputs/t10_card_p1.jsonl", 1.0),
        "T1.3k20": load("outputs/t13_card.jsonl", 0.95),
        "T1.4k20": load("outputs/t14_card.jsonl", 0.95)}
for p in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7): arms[f"T1.6p{p}"] = load("outputs/t16_full.jsonl", p)
frozen = sorted(json.load(open("outputs/baseline_qids.json")))
holdout = sorted(q for q in gold if q not in set(frozen))
arms = {k: v for k, v in arms.items() if sum(q in v for q in frozen) >= len(frozen) - 20}
partners = [k for k in arms if k != "card"]
def vote(ballots, g):
    valid = [a for a in ballots if a is not None]
    return float(bool(valid) and collections.Counter(valid).most_common(1)[0][0] == g)
def score(mix, Q):   # mix: list of (arm, n_chains); chains taken from sample_idx 0..
    return np.array([vote([a for arm, n in mix for a in arms[arm][q][:n]], gold[q]) for q in Q])
cands = [[("card", 4), (P, 4)] for P in partners] + [[("card", 6), (P, 2)] for P in partners]
cands += [[("card", 4), (P1, 2), (P2, 2)] for P1, P2 in itertools.combinations(partners, 2)]
cands += [[("card", 2), (P1, 2), (P2, 2), (P3, 2)] for P1, P2, P3 in itertools.combinations(partners, 3)]
rng = random.Random(0); Qs = list(frozen); rng.shuffle(Qs)
A, B = sorted(Qs[:len(Qs)//2]), sorted(Qs[len(Qs)//2:])
def ok(mix, Q): return all(all(q in arms[a] for q in Q) for a, _ in mix)
print(f"# Mixed-sampler ballots (8 per question) vs 8 card ballots -- split-half on the frozen set\n")
print(f"arms with full coverage: {list(arms)} | half A n={len(A)} | half B n={len(B)} | holdout n={len(holdout)} | candidates {len(cands)}\n")
cardA, cardB = score([("card", 8)], A), score([("card", 8)], B)
res = []
for mix in cands:
    if not ok(mix, A + B): continue
    dA = (score(mix, A) - cardA).mean() * 100
    res.append((dA, mix))
res.sort(key=lambda x: -x[0])
print("top 8 on half A (selection half):")
for dA, mix in res[:8]: print(f"  {'+'.join(f'{a}x{n}' for a, n in mix):40} {dA:+.2f} pp")
best = res[0][1]
sB = score(best, B); dB = sB - cardB; t = stats.ttest_rel(sB, cardB)
print(f"\nSELECTED on A: {'+'.join(f'{a}x{n}' for a, n in best)}")
print(f"CONFIRMATION on B: mix maj@8 {sB.mean():.4f} vs card {cardB.mean():.4f}  delta {dB.mean()*100:+.2f} pp  "
      f"SE {dB.std(ddof=1)/np.sqrt(len(dB))*100:.2f}  t={t.statistic:.2f}  p={t.pvalue:.4f}  (n={len(B)})")
pre = [("card", 4), ("T1.2k20", 4)]
if ok(pre, A + B):
    sB = score(pre, B); dB = sB - cardB; t = stats.ttest_rel(sB, cardB)
    print(f"A-PRIORI mix card4+T1.2k20x4 on B: {sB.mean():.4f} vs {cardB.mean():.4f}  delta {dB.mean()*100:+.2f} pp  p={t.pvalue:.4f}")
    sF = score(pre, frozen); cF = score([("card", 8)], frozen); t = stats.ttest_rel(sF, cF)
    print(f"A-PRIORI mix on full frozen set: {sF.mean():.4f} vs {cF.mean():.4f}  delta {(sF-cF).mean()*100:+.2f} pp  p={t.pvalue:.4f}  (n={len(frozen)})")
if holdout and ok(best, holdout):
    sH = score(best, holdout); cH = score([("card", 8)], holdout); t = stats.ttest_rel(sH, cH)
    print(f"HOLDOUT (322 q outside frozen set), selected mix: {sH.mean():.4f} vs card {cH.mean():.4f}  delta {(sH-cH).mean()*100:+.2f} pp  p={t.pvalue:.4f}  (n={len(holdout)})")
else:
    print("HOLDOUT: not yet generated for the selected mix's arms")
