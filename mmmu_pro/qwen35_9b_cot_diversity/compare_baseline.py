#!/usr/bin/env python
"""maj@1 / maj@8 of the top_p sweep (T=1.6, neutral profile) vs the model-card baseline
(T=1.0, top_p=0.95, top_k=20, presence_penalty=1.5) on the SAME frozen question set
(outputs/baseline_qids.json), under the sweep's ballot rule. Paired: a question counts only
if its baseline cell and every sweep cell are complete. Reports a paired t-test of the best
sweep top_p vs the baseline at each k."""
import json, collections, random, sys
sys.path.insert(0, "../common")
from analyze import parse_answer, majk_cell
import numpy as np
from scipy import stats

def load(path, label):
    cells, gold, spoil = collections.defaultdict(list), {}, collections.Counter()
    for l in open(path):
        try: r = json.loads(l)
        except Exception: continue
        stop = r["finish_reason"] == "stop"
        a = parse_answer(r["text"], r["n_options"]) if stop else None
        cells[(r["id"], r["top_p"])].append(a); gold[r["id"]] = r["gold"]
        spoil[(r["top_p"], "n")] += 1; spoil[(r["top_p"], "spoiled")] += a is None
    return cells, gold, spoil

qids = set(json.load(open("outputs/baseline_qids.json")))
sw, gold, sw_sp = load("outputs/t16_full.jsonl", "sweep")
bl, gold_b, bl_sp = load("outputs/baseline_card.jsonl", "baseline")
ps = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
paired = sorted(q for q in qids if len(bl.get((q, 0.95), [])) == 8
                and all(len(sw.get((q, p), [])) == 8 for p in ps))
print(f"frozen set {len(qids)} | baseline complete {sum(len(bl[(q,0.95)])==8 for q in qids)} "
      f"| paired {len(paired)}")
print(f"spoiled: baseline {bl_sp[(0.95,'spoiled')]/max(bl_sp[(0.95,'n')],1):.3f} | sweep "
      + " ".join(f"p{p}={sw_sp[(p,'spoiled')]/max(sw_sp[(p,'n')],1):.3f}" for p in ps))
rng = random.Random(0)
out = {}
for k in (1, 8):
    B = np.array([majk_cell(bl[(q, 0.95)], gold[q], k, 400, rng) for q in paired])
    S = {p: np.array([majk_cell(sw[(q, p)], gold[q], k, 400, rng) for q in paired]) for p in ps}
    best = max(ps, key=lambda p: S[p].mean())
    d = S[best] - B
    t = stats.ttest_rel(S[best], B)
    print(f"\nk={k}  baseline(card) {B.mean():.4f} | sweep: "
          + " ".join(f"p{p}={S[p].mean():.4f}" for p in ps))
    print(f"      best sweep p={best}: {S[best].mean():.4f}  delta {d.mean()*100:+.2f} pp  "
          f"SE {d.std(ddof=1)/np.sqrt(len(d))*100:.2f} pp  t={t.statistic:.2f}  p={t.pvalue:.4f}")
    out[k] = {"baseline": B.mean(), "sweep": {str(p): S[p].mean() for p in ps}, "best_p": best,
              "delta_pp": d.mean() * 100, "p_value": t.pvalue, "n": len(paired)}
json.dump(out, open("outputs/BASELINE_VS_SWEEP.json", "w"), indent=1, default=float)
