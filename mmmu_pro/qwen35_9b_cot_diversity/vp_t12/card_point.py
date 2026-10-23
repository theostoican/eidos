#!/usr/bin/env python
"""The model-card defaults (T=1.0, top_p .95, top_k 20, presence 1.5) as a reference point for
this arm's charts: maj@1 / maj@8 of the committed card baseline (../cots/card_t10*), under the
ballot rule, on EXACTLY the questions the visual-premise curves are paired on (a judged premise
at every top_p -- the same rule as vp_result.py's `common`). CPU only. Reads the committed
outputs/vp_verdicts.jsonl.gz and outputs/vp_layers/ (gitignored; rebuilt from ../cots by
../../common/vp_prep.py, the first step of run_vp_arm.sh).

  python card_point.py [--card ../cots/card_t10.shard*.jsonl.gz] [--out outputs/card_point.json]
"""
import argparse, collections, glob, gzip, json, random, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from analyze import parse_answer, majk_cell
from vp_result import cell_table, curve, accuracy_matrix

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vp", default="outputs/RESULT_VP.json")
    ap.add_argument("--verdicts", default="outputs/vp_verdicts.jsonl.gz")
    ap.add_argument("--layers", default="outputs/vp_layers/layer*.jsonl.gz")
    ap.add_argument("--card", default="../cots/card_t10.shard*.jsonl.gz")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="outputs/card_point.json")
    a = ap.parse_args()
    vp = json.load(open(a.vp))
    top_ps = vp["grid"]

    rows = [json.loads(l) for l in (gzip.open(a.verdicts, "rt") if a.verdicts.endswith(".gz") else open(a.verdicts))]
    _, qs_ok, _, _ = curve(cell_table(rows, True), top_ps)
    _, qs_acc = accuracy_matrix(a.layers, top_ps, layers=set(vp["layers"]))
    common = sorted(set(qs_ok) & set(qs_acc))
    assert len(common) == vp["n_questions"], (len(common), vp["n_questions"])

    cells, gold = collections.defaultdict(list), {}
    for f in sorted(glob.glob(a.card)):
        for l in gzip.open(f, "rt"):
            r = json.loads(l)
            if r["id"] not in set(common):
                continue
            ans = parse_answer(r["text"], r["n_options"]) if r["finish_reason"] == "stop" else None
            cells[r["id"]].append(ans); gold[r["id"]] = r["gold"]
    qs = [q for q in common if len(cells.get(q, [])) == a.k]
    rng = random.Random(0)
    m1 = np.array([majk_cell(cells[q], gold[q], 1, 400, rng) for q in qs])
    mk = np.array([majk_cell(cells[q], gold[q], a.k, 400, rng) for q in qs])
    out = {"label": "model card: T=1.0, top_p 0.95, top_k 20, presence 1.5", "top_p": 0.95,
           "n_questions": len(qs), "k": a.k,
           "maj1": float(m1.mean()), "maj1_sem": float(m1.std(ddof=1) / np.sqrt(len(qs))),
           "majk": float(mk.mean()), "majk_sem": float(mk.std(ddof=1) / np.sqrt(len(qs)))}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[card] n={len(qs)} of {len(common)} paired | maj@1 {out['maj1']:.4f} ±{out['maj1_sem']:.4f} | "
          f"maj@{a.k} {out['majk']:.4f} ±{out['majk_sem']:.4f} -> {a.out}")

if __name__ == "__main__":
    main()
