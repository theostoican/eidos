#!/usr/bin/env python
"""maj@k: majority-vote accuracy as a function of BOTH top_p and the vote budget k.

Why this exists. With n=16 samples per cell, self-consistency SATURATES: the majority
answer converges to the model's modal answer and top_p stops mattering. Measured on the
first two cells, the gap between top_p=0.1 and top_p=1.0 collapses from 0.0245 at k=1 to
0.0010 at k=16 -- a 25x shrink. So a single maj@16 curve can look flat even if top_p has
a real effect at smaller vote budgets.

k <= n_samples is subsampled from the samples ALREADY generated, so the whole k-family is
free -- no extra generation. If the inverted-U exists it should be visible at low k and
wash out at high k.

Estimator: for each (id, top_p) cell, draw B random size-k subsets WITHOUT replacement
from that cell's parsed answers, take the plurality of each, and average the indicator
(plurality == gold). Ties broken by Counter.most_common order (first-seen wins), matching
analyze_cot.py's majority rule.
"""
import argparse, collections, glob, json, math, random
import numpy as np

from extract_cot import parse_answer

DEFAULT_KS = [1, 3, 5, 7, 9, 12, 16]


def load_cells(pattern="outputs/vp_gen.shard*.jsonl", include_truncated=False):
    """-> {(id, top_p): (preds:list[str], gold:str)}"""
    raw = collections.defaultdict(list)
    for f in glob.glob(pattern):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if not include_truncated and r.get("finish_reason") != "stop":
                continue
            raw[(r["id"], r["top_p"])].append(r)
    out = {}
    for key, rs in raw.items():
        preds = [a for a in (parse_answer(r["text"], r.get("n_options", 10)) for r in rs) if a]
        if preds:
            out[key] = (preds, rs[0]["gold"])
    return out


def majk_cell(preds, gold, k, B, rng):
    """P(plurality of a random size-k subset == gold)."""
    if len(preds) < k:
        return None
    if k == 1:
        return sum(p == gold for p in preds) / len(preds)
    if k >= len(preds):
        m = collections.Counter(preds).most_common(1)[0][0]
        return 1.0 if m == gold else 0.0
    hits = 0
    for _ in range(B):
        s = rng.sample(preds, k)
        if collections.Counter(s).most_common(1)[0][0] == gold:
            hits += 1
    return hits / B


def majk_table(cells, ks=DEFAULT_KS, B=400, seed=0, balanced=True):
    """-> (top_ps, {k: {top_p: (mean, se, n_cells)}}).

    balanced=True restricts to questions present at EVERY top_p, so curves are compared on
    the same questions (top_p cells complete at different times; unbalanced means would mix
    different question difficulties and can flip the sign of the result).
    """
    rng = random.Random(seed)
    top_ps = sorted({p for _, p in cells})
    qs_by_p = {p: {q for q, pp in cells if pp == p} for p in top_ps}
    qs = set.intersection(*qs_by_p.values()) if balanced and top_ps else set(q for q, _ in cells)
    qs = sorted(qs)

    table = {}
    for k in ks:
        # PER-k BALANCING. A cell whose trace was truncated has <16 valid answers, so it
        # silently drops out at high k -- and drops out UNEQUALLY across top_p (top_p=0.5
        # had 22 truncations, top_p=1.0 had 0). The dropped questions are the hard ones, so
        # naive per-k means compare different question sets and inflate the top_p values
        # that truncate most. Require EVERY top_p to have >=k valid answers for a question
        # before that question contributes to this k.
        if balanced:
            qs_k = [q for q in qs
                    if all((q, p) in cells and len(cells[(q, p)][0]) >= k for p in top_ps)]
        else:
            qs_k = qs
        table[k] = {}
        for p in top_ps:
            vals = [v for v in (majk_cell(*cells[(q, p)], k, B, rng) for q in qs_k
                                if (q, p) in cells) if v is not None]
            if not vals:
                continue
            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            table[k][p] = (mean, se, len(vals))
    return top_ps, table, qs


def shape_verdict(ps, means, ses):
    """Interior peak? Returns (has_peak, xstar, amp, noise, msg)."""
    if len(ps) < 3:
        return None
    a, b, c = np.polyfit(np.array(ps, float), np.array(means, float), 2)
    xstar = -b / (2 * a) if a else float("nan")
    interior = a < 0 and min(ps) < xstar < max(ps)
    imax = int(np.argmax(means))
    edge = imax in (0, len(ps) - 1)
    amp = max(means) - min(means)
    noise = 2 * max(ses) if ses else 0.0
    ok = interior and not edge and amp > noise
    msg = ("INVERTED-U" if ok else
           ("interior vertex but peak at edge / within noise" if interior else "no interior peak"))
    return ok, float(xstar), amp, noise, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/vp_gen.shard*.jsonl")
    ap.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unbalanced", action="store_true",
                    help="do NOT restrict to questions common to every top_p (not recommended)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cells = load_cells(args.glob)
    if not cells:
        print("[majk] no generations yet")
        return
    ps, table, qs = majk_table(cells, ks, args.boot, args.seed, balanced=not args.unbalanced)
    print(f"[majk] top_p present: {ps}")
    print(f"[majk] {len(qs)} questions {'common to ALL top_p (balanced)' if not args.unbalanced else '(unbalanced)'}")
    if len(ps) < 2:
        print("[majk] need >=2 top_p")
        return

    print(f"\n{'k':>3} {'nQ':>4}  " + " ".join(f"{('p='+str(p)):>9}" for p in ps) + "   shape")
    print("-" * (4 + 10 * len(ps) + 30))
    for k in ks:
        row = table.get(k, {})
        if len(row) < len(ps):
            continue
        means = [row[p][0] for p in ps]
        ses = [row[p][1] for p in ps]
        cellsn = row[ps[0]][2]
        sv = shape_verdict(ps, means, ses)
        tag = ""
        if sv:
            ok, xstar, amp, noise, msg = sv
            tag = f"{msg}" + (f" @p={xstar:.2f}" if ok else "")
            tag += f"  (amp={amp:.4f} vs 2SE={noise:.4f})"
        print(f"{k:>3} {cellsn:>4}  " + " ".join(f"{m:>9.4f}" for m in means) + f"   {tag}")
    print(f"\n[majk] n_cells per point = {table[ks[0]][ps[0]][2]}")

    if args.json_out:
        json.dump({"top_ps": ps, "n_questions": len(qs),
                   "table": {str(k): {str(p): table[k][p] for p in table[k]} for k in table}},
                  open(args.json_out, "w"), indent=1)
        print(f"[majk] -> {args.json_out}")


if __name__ == "__main__":
    main()
