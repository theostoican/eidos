#!/usr/bin/env python
"""CoT-diversity analysis. Unit = one <think> CoT per sample.

Per (id, top_p) cell over the 16 samples:
  - diversity : Vendi score + mean pairwise cosine distance over embeddings of the CoT.
                CoTs are long, so each CoT is chunked and its chunk-embeddings are
                mean-pooled (MiniLM caps at ~512 tokens; embedding only the opening
                would misrepresent a 10k-token trace).
  - cot_correct   : fraction of the 16 CoTs a vision judge ruled reasoning-SOUND
                    (from verdicts_cot.jsonl; None if judging not yet run).
  - answer_acc    : fraction of the 16 samples whose parsed 'Answer: X' == gold.
  - majority_correct: does the plurality answer over the 16 samples == gold (self-consistency).

Then the EVOLUTION over top_p (mean of each metric per top_p) and top_p / diversity /
correctness / accuracy correlations across the (id, top_p) cells.
"""
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
    xy = [(a, b) for a, b in zip(x, y) if a is not None and b is not None
          and not (isinstance(b, float) and math.isnan(b))]
    if len(xy) < 3: return float("nan"), float("nan")
    x = np.array([a for a, _ in xy], float); y = np.array([b for _, b in xy], float)
    if np.std(x) == 0 or np.std(y) == 0: return float("nan"), float("nan")
    pe = float(np.corrcoef(x, y)[0, 1]); rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return pe, float(np.corrcoef(rx, ry)[0, 1])

def chunk_words(text, size=180):
    w = text.split()
    if not w: return [""]
    return [" ".join(w[i:i + size]) for i in range(0, len(w), size)]

def _minilm_encode(chunks, model_name, batch=128):
    """Encode text chunks with a MiniLM sentence-transformer, loaded via plain
    transformers (mean-pooling) to avoid the sentence-transformers->torchcodec->FFmpeg
    import chain. Returns L2-normalized [N, H] embeddings."""
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()
    out = []
    for i in range(0, len(chunks), batch):
        enc = tok(chunks[i:i + batch], padding=True, truncation=True, max_length=256,
                  return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = model(**enc).last_hidden_state            # [B, T, H]
        m = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)     # masked mean pool
        v = torch.nn.functional.normalize(v, dim=1)
        out.append(v.cpu().numpy())
    return np.concatenate(out, 0)

def embed_cots(cots, model_name):
    """Chunk + mean-pool each CoT into a single normalized vector."""
    chunks, owner = [], []
    for i, c in enumerate(cots):
        for ch in chunk_words(c):
            chunks.append(ch); owner.append(i)
    E = _minilm_encode(chunks, model_name)
    dim = E.shape[1]; out = np.zeros((len(cots), dim), np.float32); cnt = np.zeros(len(cots))
    for v, o in zip(E, owner):
        out[o] += v; cnt[o] += 1
    out /= np.maximum(cnt, 1)[:, None]
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="outputs/cot_extracted.jsonl")
    ap.add_argument("--verdicts", default="outputs/verdicts_cot.jsonl", help="judge verdicts (optional)")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default="outputs/cot_report.json")
    ap.add_argument("--include-truncated", action="store_true",
                    help="keep finish_reason!='stop' traces. DEFAULT drops them: truncated CoTs "
                         "are repetition-loop outliers whose prevalence is strongly top_p-dependent "
                         "(11%% at 0.5 -> 0.6%% at 1.0), which otherwise CONFOUNDS every top_p / "
                         "diversity correlation (manufactured the spurious -0.35).")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.extracted)]
    if not args.include_truncated:
        n0 = len(rows)
        rows = [r for r in rows if r.get("finish_reason") == "stop"]
        print(f"[filter] completed-only: kept {len(rows)}/{n0} traces "
              f"(dropped {n0 - len(rows)} truncated; pass --include-truncated to keep)")
    # judge verdicts: (id, top_p, sample_idx) -> sound(bool)
    vmap = {}
    try:
        for v in (json.loads(l) for l in open(args.verdicts)):
            vmap[(v["id"], v["top_p"], v["sample_idx"])] = v.get("sound")
    except FileNotFoundError:
        print(f"[warn] {args.verdicts} not found -> cot_correct will be null")

    E = embed_cots([r["cot"] for r in rows], args.model)
    eby = {id(r): E[i] for i, r in enumerate(rows)}

    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["id"], r["top_p"])].append(r)

    per_cell = []
    for (qid, p), samp in sorted(cells.items()):
        emb = np.array([eby[id(r)] for r in samp])
        preds = [r["pred"] for r in samp]
        acc = [int(r["answer_correct"]) for r in samp if r["pred"] is not None]
        gold = samp[0]["gold"]
        valid_preds = [x for x in preds if x is not None]
        maj = collections.Counter(valid_preds).most_common(1)[0][0] if valid_preds else None
        sound = [vmap.get((qid, p, r["sample_idx"])) for r in samp]
        sound = [s for s in sound if s is not None]
        per_cell.append({
            "id": qid, "subject": samp[0]["subject"], "top_p": p, "n": len(samp), "gold": gold,
            "vendi": round(vendi(emb), 4), "cos_dist": round(cosd(emb), 4),
            "answer_acc": round(np.mean(acc), 4) if acc else None,
            "n_answered": len(acc),
            "majority_correct": (int(maj == gold) if maj is not None else None),
            "cot_correct": round(np.mean(sound), 4) if sound else None,
            "n_judged": len(sound),
        })

    ps = sorted({c["top_p"] for c in per_cell})
    P = [c["top_p"] for c in per_cell]
    metrics = ["vendi", "cos_dist", "answer_acc", "cot_correct", "majority_correct"]
    correlations = {}
    for nm in metrics:
        pe, sp = corr(P, [c[nm] for c in per_cell]); correlations[f"top_p__{nm}"] = {"pearson": round(pe, 3), "spearman": round(sp, 3)}
    for div in ["vendi", "cos_dist"]:
        for tgt in ["answer_acc", "cot_correct"]:
            pe, sp = corr([c[div] for c in per_cell], [c[tgt] for c in per_cell])
            correlations[f"{div}__{tgt}"] = {"pearson": round(pe, 3), "spearman": round(sp, 3)}

    # EVOLUTION over top_p: mean of each metric at each top_p
    byp = collections.defaultdict(list)
    for c in per_cell: byp[c["top_p"]].append(c)
    def m(cs, k):
        vals = [c[k] for c in cs if c[k] is not None]
        return round(float(np.mean(vals)), 4) if vals else None
    evolution = [{"top_p": p, "n_cells": len(byp[p]),
                  "vendi": m(byp[p], "vendi"), "cos_dist": m(byp[p], "cos_dist"),
                  "answer_acc": m(byp[p], "answer_acc"), "cot_correct": m(byp[p], "cot_correct"),
                  "majority_acc": m(byp[p], "majority_correct")} for p in ps]

    json.dump({"unit": "one <think> CoT per sample", "embed_model": args.model,
               "n_cells": len(per_cell), "top_ps": ps, "evolution": evolution,
               "correlations": correlations, "per_cell": per_cell}, open(args.out, "w"), indent=2)

    print(f"{'top_p':>6}{'vendi':>8}{'cos_d':>8}{'ans_acc':>9}{'maj_acc':>9}{'cot_ok':>9}")
    for e in evolution:
        cot = f"{e['cot_correct']:.3f}" if e["cot_correct"] is not None else "  n/a"
        print(f"{e['top_p']:>6}{e['vendi']:>8.2f}{e['cos_dist']:>8.3f}{e['answer_acc']:>9.3f}"
              f"{e['majority_acc']:>9.3f}{cot:>9}")
    print("\ncorrelations (across id x top_p cells):")
    for k, v in correlations.items():
        print(f"  {k:>26}: pearson={v['pearson']:+.3f} spearman={v['spearman']:+.3f}")

if __name__ == "__main__":
    main()
