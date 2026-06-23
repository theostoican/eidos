#!/usr/bin/env python
"""Comprehensive-premise analysis. Unit = one comprehensive premise per sample.
Per (id, top_p) cell: frac_correct (strict-binary verdicts) + diversity (Vendi +
mean pairwise cosine over MiniLM embeddings of the 16 premises). Then top_p /
diversity / correctness correlations."""
import argparse, json, collections, math
import numpy as np

def vendi(emb):
    n = len(emb)
    if n <= 1: return 1.0
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    K = (X @ X.T); K = (K + K.T) / 2
    w = np.linalg.eigvalsh(K / n); w = np.clip(w.real, 0, None); w = w[w > 1e-12]
    return float(math.exp(-np.sum(w * np.log(w))))

def cosd(emb):
    n = len(emb)
    if n <= 1: return 0.0
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12); S = X @ X.T
    iu = np.triu_indices(n, k=1); return float(np.mean(1 - S[iu]))

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0: return float("nan"), float("nan")
    pe = float(np.corrcoef(x, y)[0, 1]); rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return pe, float(np.corrcoef(rx, ry)[0, 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", default="outputs/premises_comp_extracted.jsonl")
    ap.add_argument("--verdicts", default="outputs/verdicts_comp.jsonl")
    ap.add_argument("--out", default="outputs/premise_report_comp.json")
    args = ap.parse_args()
    prem = [json.loads(l) for l in open(args.premises)]
    vmap = {(v["id"], v["top_p"], v["sample_idx"]): v["correct"]
            for v in (json.loads(l) for l in open(args.verdicts))}
    cells = collections.defaultdict(list)
    for r in prem:
        cells[(r["id"], r["top_p"])].append(r)
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda").encode(
        [r["premise"] for r in prem], batch_size=64, show_progress_bar=False)
    eby = {id(r): emb[i] for i, r in enumerate(prem)}
    per_cell = []
    for (qid, p), samp in sorted(cells.items()):
        E = np.array([eby[id(r)] for r in samp])
        have = [vmap.get((qid, p, r["sample_idx"])) for r in samp]; have = [v for v in have if v is not None]
        per_cell.append({"id": qid, "subject": samp[0]["subject"], "top_p": p, "n": len(samp),
                         "gold": samp[0]["gold"], "vendi": round(vendi(E), 4), "cos_dist": round(cosd(E), 4),
                         "frac_correct": round(sum(have) / len(have), 4) if have else None})
    ps = sorted({c["top_p"] for c in per_cell})
    P = [c["top_p"] for c in per_cell]; FC = [c["frac_correct"] for c in per_cell]
    correlations = {}
    for nm in ["vendi", "cos_dist", "frac_correct"]:
        pe, sp = corr(P, [c[nm] for c in per_cell]); correlations[f"top_p__{nm}"] = {"pearson": round(pe, 3), "spearman": round(sp, 3)}
    for nm in ["vendi", "cos_dist"]:
        pe, sp = corr([c[nm] for c in per_cell], FC); correlations[f"{nm}__frac_correct"] = {"pearson": round(pe, 3), "spearman": round(sp, 3)}
    json.dump({"unit": "comprehensive single premise", "n_cells": len(per_cell), "top_ps": ps,
               "correlations": correlations, "per_cell": per_cell}, open(args.out, "w"), indent=2)
    byp = collections.defaultdict(list)
    for c in per_cell: byp[c["top_p"]].append(c)
    print(f"{'top_p':>6}{'vendi':>8}{'cos_d':>8}{'fr_corr':>9}")
    for p in ps:
        cs = byp[p]; print(f"{p:>6}{np.mean([c['vendi'] for c in cs]):>8.2f}{np.mean([c['cos_dist'] for c in cs]):>8.3f}{np.mean([c['frac_correct'] for c in cs]):>9.3f}")
    print("\ncorrelations:")
    for k, v in correlations.items(): print(f"  {k:>24}: pearson={v['pearson']:+.3f} spearman={v['spearman']:+.3f}")
    allv = [json.loads(l)["correct"] for l in open(args.verdicts)]
    print(f"\nOVERALL premise accuracy: {sum(allv)}/{len(allv)} = {sum(allv)/len(allv):.1%}")

if __name__ == "__main__":
    main()
