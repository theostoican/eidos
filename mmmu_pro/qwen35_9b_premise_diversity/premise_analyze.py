#!/usr/bin/env python
"""Analyze the premise sweep: semantic diversity per (question, top_p) cell, and
(optionally) correctness merged from a judge verdicts file.

verdicts file (jsonl): {"id":..., "top_p":..., "sample_idx":..., "correct": true/false}
If absent, frac_correct is left null (diversity-only pass).
"""
import argparse, json, collections, math
import numpy as np

def final_answer_text(text):
    if "</think>" in text:
        seg = text.split("</think>")[-1].strip()
        if seg:
            return seg
    return text[-1500:].strip()

def vendi_score(emb):
    n = len(emb)
    if n <= 1:
        return 1.0
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    K = (X @ X.T); K = (K + K.T) / 2.0
    w = np.linalg.eigvalsh(K / n)
    w = np.clip(w.real, 0, None); w = w[w > 1e-12]
    return float(math.exp(-np.sum(w * np.log(w))))

def mean_pairwise_cos_dist(emb):
    n = len(emb)
    if n <= 1:
        return 0.0
    X = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    iu = np.triu_indices(n, k=1)
    return float(np.mean(1.0 - S[iu]))

def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(x, y)[0, 1])
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return pear, float(np.corrcoef(rx, ry)[0, 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", default="/workspace/mmmupro_qwen3vl/outputs/premises_v3.jsonl")
    ap.add_argument("--verdicts", default="/workspace/mmmupro_qwen3vl/outputs/verdicts_v3.jsonl")
    ap.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="/workspace/mmmupro_qwen3vl/outputs/premise_report_v3.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.premises)]
    # verdict map
    vmap = {}
    try:
        for l in open(args.verdicts):
            v = json.loads(l)
            vmap[(v["id"], float(v["top_p"]), int(v["sample_idx"]))] = bool(v["correct"])
        print(f"[verdicts] loaded {len(vmap)}")
    except FileNotFoundError:
        print("[verdicts] none found -> diversity-only pass")

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["id"], r["top_p"])].append(r)

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(args.embed_model, device="cuda")
    texts = [final_answer_text(r["text"]) for r in rows]
    emb = embedder.encode(texts, batch_size=64, normalize_embeddings=False, show_progress_bar=False)
    eby = {id(r): emb[i] for i, r in enumerate(rows)}

    per_cell = []
    for (qid, p), samp in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        E = np.array([eby[id(r)] for r in samp])
        n = len(samp)
        verds = [vmap.get((qid, p, r["sample_idx"])) for r in samp]
        have = [v for v in verds if v is not None]
        frac_correct = (sum(have) / len(have)) if have else None
        per_cell.append({
            "id": qid, "subject": samp[0]["subject"], "top_p": p, "n": n,
            "gold": samp[0]["gold"],
            "vendi": round(vendi_score(E), 4),
            "cos_dist": round(mean_pairwise_cos_dist(E), 4),
            "frac_correct": (round(frac_correct, 4) if frac_correct is not None else None),
            "n_judged": len(have),
            "mean_tokens": round(float(np.mean([r["out_tokens"] for r in samp])), 1),
        })

    ps = sorted({c["top_p"] for c in per_cell})
    correlations = {}
    if all(c["frac_correct"] is not None for c in per_cell):
        P = [c["top_p"] for c in per_cell]
        FC = [c["frac_correct"] for c in per_cell]
        for name, key in [("vendi", "vendi"), ("cos_dist", "cos_dist"), ("frac_correct", "frac_correct")]:
            pe, sp = corr(P, [c[key] for c in per_cell])
            correlations[f"top_p__{name}"] = {"pearson": round(pe, 3), "spearman": round(sp, 3)}
        for name, key in [("vendi", "vendi"), ("cos_dist", "cos_dist")]:
            pe, sp = corr([c[key] for c in per_cell], FC)
            correlations[f"{name}__frac_correct"] = {"pearson": round(pe, 3), "spearman": round(sp, 3)}

    report = {"n_cells": len(per_cell), "top_ps": ps, "correlations": correlations, "per_cell": per_cell}
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"[saved] {args.out}")
    print(f"{'top_p':>6}{'vendi':>8}{'cos_d':>8}{'fr_corr':>9}")
    byp = collections.defaultdict(list)
    for c in per_cell:
        byp[c["top_p"]].append(c)
    for p in ps:
        cs = byp[p]
        fc = [c["frac_correct"] for c in cs if c["frac_correct"] is not None]
        print(f"{p:>6}{np.mean([c['vendi'] for c in cs]):>8.3f}"
              f"{np.mean([c['cos_dist'] for c in cs]):>8.3f}"
              f"{(np.mean(fc) if fc else float('nan')):>9.3f}")
    if correlations:
        print("\ncorrelations:")
        for k, v in correlations.items():
            print(f"  {k:>22}: pearson={v['pearson']:+.3f} spearman={v['spearman']:+.3f}")

if __name__ == "__main__":
    main()
