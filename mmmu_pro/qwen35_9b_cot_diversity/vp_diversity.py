#!/usr/bin/env python
"""Stage 4b: diversity of the extracted visual premises vs top_p (T=1.0).

THE CONFOUND THAT DECIDES WHETHER THIS MEANS ANYTHING. Premise count rises monotonically
along the swept axis -- 6.40 per trace at top_p=0.5 to 7.33 at 1.0, +15% -- and Vendi is the
effective NUMBER of distinct items, so it inherits that gradient directly. Unmatched, the
cell curve rises +15.4% with quadratic a=+0.30; count-matched it rises +9.0% with a=-0.44.
The confound is worth ~40% of the effect and flips the curvature sign.

So every measure is COUNT-MATCHED: each cell is subsampled to exactly K items, R times, and
averaged. Cells with fewer than K items are dropped and the survivors are reported per
top_p. Raw unmatched numbers are printed alongside so the size of the confound is visible.

TWO LEVELS:
  within-trace   how varied the reads are inside ONE chain of thought
  across-sample  how differently the model frames its reading of the SAME image on a rerun.
                 This is what a top_p sweep is about, and it is matched on trace count too,
                 because the empty-trace rate also drifts (23.3% at 0.5 to 19.6% at 1.0).

MEASURES. Vendi = exp(Shannon entropy of the similarity-matrix eigenvalues) = the effective
number of distinct items, bounded [1, K]. Mean pairwise cosine distance is the scale-free
companion. A MODEL-FREE numeric measure covers the encoder's blind spot: every encoder tested
scores a misread number at <=0.21 of a different claim (vp_embed_probe.py), and a number IS
the observation in much of this data.
"""
import argparse, collections, json, random, re
import numpy as np

NUM = re.compile(r"-?\d+(?:\.\d+)?")


def vendi(E):
    """Effective number of distinct items. E: (n, d) L2-normalised embeddings.
    Bounded [1, n]: 1 when all items are identical (rank-1 Gram -> zero entropy), n when all
    are mutually orthogonal. Never 0, never unbounded."""
    n = len(E)
    if n < 2:
        return 1.0
    K = (E @ E.T) / n
    w = np.linalg.eigvalsh(K)
    w = w[w > 1e-12]
    if len(w) == 0:
        return 1.0
    return float(np.exp(-(w * np.log(w)).sum()))


def cosd(E):
    n = len(E)
    if n < 2:
        return 0.0
    S = E @ E.T
    iu = np.triu_indices(n, 1)
    return float(1.0 - S[iu].mean())


def matched(E, K, R, rng):
    """Count-matched Vendi and cosine distance: mean over R subsamples of exactly K rows."""
    n = len(E)
    if n < K:
        return None, None
    if n == K:
        return vendi(E), cosd(E)
    vs, cs = [], []
    for _ in range(R):
        sub = E[rng.sample(range(n), K)]
        vs.append(vendi(sub))
        cs.append(cosd(sub))
    return float(np.mean(vs)), float(np.mean(cs))


def numbers_in(premises):
    """Every numeral a trace asserts, normalised so 40,000 == 40000."""
    out = []
    for p in premises:
        for tok in NUM.findall(p.replace(",", "")):
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def eff_distinct(values):
    """exp(Shannon entropy) over the value distribution -- model-free, and blind to nothing:
    40,000 and 45,000 are simply different values."""
    if not values:
        return None
    c = collections.Counter(values)
    n = sum(c.values())
    p = np.array([v / n for v in c.values()])
    return float(np.exp(-(p * np.log(p)).sum()))


def jaccard_dist(sets):
    ss = [s for s in sets if s]
    if len(ss) < 2:
        return None
    d = [1.0 - (len(ss[i] & ss[j]) / len(ss[i] | ss[j])) if len(ss[i] | ss[j]) else 0.0
         for i in range(len(ss)) for j in range(i + 1, len(ss))]
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", default="outputs/vp_premises.jsonl")
    # bge-large won vp_embed_probe.py on misread sensitivity (0.211). The 8B encoder scored
    # WORSE (0.185): scale buys topical nuance, not numeral fidelity. fp32, because Qwen3 in
    # bf16 returned 1.0025 for identical pairs and Vendi is an eigenvalue quantity.
    ap.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--grid", default="0.5,0.6,0.7,0.8,0.9,0.925,0.95,0.975,0.99,1.0")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--k-premises", type=int, default=4)
    ap.add_argument("--k-cell", type=int, default=20)
    ap.add_argument("--k-traces", type=int, default=5)
    ap.add_argument("--reps", type=int, default=24)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/RESULT_VP_DIVERSITY.md")
    a = ap.parse_args()
    grid = [float(x) for x in a.grid.split(",")]
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in open(a.premises)]
    rows = [r for r in rows if r["sample_idx"] < a.layers and r["n_premises"] > 0
            and r["top_p"] in grid]
    texts, owner = [], []
    for r in rows:
        for p in r["premises"]:
            texts.append(p)
            owner.append((r["id"], r["top_p"], r["sample_idx"]))
    print(f"[div] {len(texts)} premises from {len(rows)} traces | encoder {a.model}", flush=True)

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(a.model, device="cuda", trust_remote_code=True)
    E = m.encode(texts, batch_size=a.batch, normalize_embeddings=True,
                 show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

    by_trace = collections.defaultdict(list)
    for e, k in zip(E, owner):
        by_trace[k].append(e)
    by_trace = {k: np.array(v) for k, v in by_trace.items()}
    prem_by_trace = {(r["id"], r["top_p"], r["sample_idx"]): r["premises"] for r in rows}

    cells = collections.defaultdict(list)
    for (qid, tp, s) in by_trace:
        cells[(qid, tp)].append(s)

    per = []
    for (qid, tp), sidx in cells.items():
        keys = [(qid, tp, s) for s in sorted(sidx)]
        prem = np.concatenate([by_trace[k] for k in keys])
        wt = [matched(by_trace[k], a.k_premises, a.reps, rng) for k in keys]
        wt = [x for x in wt if x[0] is not None]
        v_cell, c_cell = matched(prem, a.k_cell, a.reps, rng)
        setv = np.array([by_trace[k].mean(0) for k in keys])
        setv /= np.linalg.norm(setv, axis=1, keepdims=True) + 1e-12
        v_set, c_set = matched(setv, a.k_traces, a.reps, rng)
        num_eff = num_jac = None
        if len(keys) >= a.k_traces:
            evs, jds = [], []
            for _ in range(a.reps):
                pick = rng.sample(keys, a.k_traces)
                vals, sets_ = [], []
                for k in pick:
                    nv = numbers_in(prem_by_trace[k])
                    vals += nv
                    sets_.append(set(nv))
                e, j = eff_distinct(vals), jaccard_dist(sets_)
                if e is not None:
                    evs.append(e)
                if j is not None:
                    jds.append(j)
            num_eff = float(np.mean(evs)) if evs else None
            num_jac = float(np.mean(jds)) if jds else None
        per.append({"id": qid, "top_p": tp, "n_traces": len(keys), "n_premises": len(prem),
                    "num_eff_distinct": num_eff, "num_jaccard": num_jac,
                    "vendi_within": float(np.mean([x[0] for x in wt])) if wt else None,
                    "cosd_within": float(np.mean([x[1] for x in wt])) if wt else None,
                    "vendi_cell": v_cell, "cosd_cell": c_cell,
                    "vendi_set": v_set, "cosd_set": c_set,
                    "vendi_cell_raw": vendi(prem), "cosd_cell_raw": cosd(prem),
                    "vendi_set_raw": vendi(setv)})

    def col(tp, key):
        v = [c[key] for c in per if c["top_p"] == tp and c[key] is not None]
        return ((float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(len(v))), len(v))
                if v else (None, None, 0))

    out = ["# Diversity of the extracted visual premises vs top_p (T=1.0)", "",
           f"Encoder: `{a.model}` (chosen by vp_embed_probe.py).",
           f"{len(texts)} premises, {len(rows)} traces, {len(cells)} cells, "
           f"{a.layers} samples per cell.", "",
           f"Count-matched: within-trace to {a.k_premises} premises/trace, cell to {a.k_cell} "
           f"premises, across-sample to {a.k_traces} traces; {a.reps} subsamples averaged. "
           "Premise count rises with top_p, so the unmatched rows are the confound, not the "
           "result.", "",
           "| measure | " + " | ".join(str(p) for p in grid) + " |",
           "|---|" + "---|" * len(grid)]
    for k, label in [("vendi_set", "**across-sample Vendi** (matched traces)"),
                     ("cosd_set", "across-sample cosine distance"),
                     ("vendi_cell", "**cell Vendi** (matched premises)"),
                     ("cosd_cell", "cell cosine distance"),
                     ("vendi_within", "within-trace Vendi (matched)"),
                     ("cosd_within", "within-trace cosine distance"),
                     ("vendi_cell_raw", "cell Vendi, UNMATCHED (confounded)"),
                     ("vendi_set_raw", "across-sample Vendi, UNMATCHED"),
                     ("num_eff_distinct", "**numeric reads: effective distinct values** (model-free)"),
                     ("num_jaccard", "numeric reads: mean Jaccard distance between traces"),
                     ("n_premises", "premises per cell (the confound)")]:
        vals = [col(p, k)[0] for p in grid]
        out.append(f"| {label} | " + " | ".join(
            f"{v:.4f}" if v is not None else "-" for v in vals) + " |")
    out.append("| +/-1 SEM (across-sample Vendi) | " + " | ".join(
        f"{col(p,'vendi_set')[1]:.4f}" for p in grid) + " |")
    out.append("| cells surviving the trace match | " + " | ".join(
        str(col(p, "vendi_set")[2]) for p in grid) + " |")

    import sys
    sys.path.insert(0, ".")
    from analyze import rm_anova, shape_test
    stats_out = {}
    for k in ("vendi_set", "vendi_cell", "cosd_cell", "num_eff_distinct", "num_jaccard",
              "vendi_cell_raw"):
        d = {(c["id"], c["top_p"]): c[k] for c in per if c[k] is not None}
        qs = [q for q in sorted({q for q, _ in d}) if all((q, p) in d for p in grid)]
        if len(qs) < 10:
            continue
        M = np.array([[d[(q, p)] for p in grid] for q in qs])
        F, pv = rm_anova(M)
        st = shape_test(grid, M, 4000, a.seed)
        stats_out[k] = {"n": len(qs), "F": F, "p": pv, "argmax": st["argmax"],
                        "quad_a": st["quad_a"], "means": M.mean(0).tolist(),
                        "sem": (M.std(0, ddof=1) / np.sqrt(len(M))).tolist()}
        out.append(f"\n- `{k}`: n={len(qs)} paired questions, RM-ANOVA F={F:.2f}, p={pv:.4g}, "
                   f"argmax {st['argmax']}, quad a={st['quad_a']:+.4f}")
    open(a.out, "w").write("\n".join(out) + "\n")
    json.dump({"model": a.model, "grid": grid, "stats": stats_out, "per_cell_n": len(per)},
              open(a.out.replace(".md", ".json"), "w"), indent=1)
    json.dump(per, open("outputs/vp_diversity_cells.json", "w"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
