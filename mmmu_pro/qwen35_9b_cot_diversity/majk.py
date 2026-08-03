#!/usr/bin/env python
"""maj@k vs top_p under the SPOILED-BALLOT counting rule (HANDOFF section 2.3-2.5, 3).

What changed vs the original majk.py, and why.

The original loaded only `finish_reason == "stop"` rows and then dropped unparseable
ones too, so a cell's vote pool shrank wherever generation truncated -- and truncation
is strongly top_p-dependent (9.8% at top_p=0.1 vs 0.1% at 1.0 in the v1 data). That is
a differential filter across the very axis being swept: it biases the low-top_p end
upward and, on its own, moved the apparent peak from the 1.0 edge to 0.5.

Ballot model instead. Every (question, top_p) cell has exactly n_samples BALLOTS, one
per generated sample. A ballot is VALID if the trace terminated and parse_answer found
a letter; otherwise it is SPOILED. Spoiled ballots consume vote budget but do not vote.
No sample is ever silently discarded, so every cell has the same denominator and the
per-k balancing that used to drop 39 of 86 questions is unnecessary.

Three counting rules (--counting):
  spoiled  (primary)   spoiled ballots consume budget, do not vote; a draw with zero
                       valid ballots is a failure.
  sentinel (harsher)   spoiled ballots all vote for one sentinel that can never equal
                       gold, so a heavily-truncated cell can lose to the sentinel.
  exclude  (the old)   spoiled ballots removed from the pool entirely. Kept ONLY so the
                       delta against the other two is visible -- that delta quantifies
                       how much of any observed shape is a filtering artifact.

Shape test is the pre-registered one from HANDOFF section 4: two-lines split at the
argmax, bootstrapped over QUESTIONS (the same resample applied to every top_p column so
pairing is preserved), reporting P(shape holds). The old `amp > 2*max(se)` heuristic
used the marginal SE across questions, ~2.6x larger than the paired SE that applies.
"""
import argparse, collections, glob, gzip, json, random, re
import numpy as np

# Answer parsing, inlined so this tree has no dependency on the superseded pipeline.
# Matches the official MMMU-Pro answer format the generation prompt asks for
# ("Answer: $LETTER"), falling back to the last bare option letter. n_options is per
# question -- despite the "standard (10 options)" config name only ~70% of questions
# actually have 10, so the valid-letter set must be derived per question, not assumed.
LETTERS = [chr(ord("A") + i) for i in range(26)]
ANS_RE = re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)


def parse_answer(text, n_options):
    valid = set(LETTERS[:n_options])
    after = text.split("</think>")[-1] if "</think>" in text else text
    for c in reversed(ANS_RE.findall(after) or ANS_RE.findall(text)):
        if c.upper() in valid:
            return c.upper()
    for ch in reversed(re.findall(r"\b([A-J])\b", after)):
        if ch in valid:
            return ch
    return None

DEFAULT_KS = [1, 2, 4, 8, 16]
SENTINEL = "<spoiled>"


def _open(path):
    """Transparent gzip (HANDOFF Phase 0: shards ship as .jsonl.gz)."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_cells(pattern, temperature=None):
    """-> ({(id, top_p): (ballots, gold)}, stats).

    ballots is one entry per GENERATED sample: a letter, or None for a spoiled ballot
    (truncated, or terminated but unparseable). Nothing is dropped.
    """
    raw = collections.defaultdict(list)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[majk] no files match {pattern!r}")
    cfgs, temps = set(), set()
    for f in files:
        for line in _open(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if temperature is not None and r.get("temperature", 1.0) != temperature:
                continue
            raw[(r["id"], r["top_p"])].append(r)
            temps.add(r.get("temperature", 1.0))
            if "cfg_profile" in r:
                cfgs.add(r["cfg_profile"])

    # Cells are keyed on (id, top_p) ONLY. If a glob spans more than one temperature the
    # arms silently merge into 32-ballot cells and every curve becomes a blend of two
    # experiments. Temperature is the swept variable in the T-sweep, so this is a live
    # footgun the moment T=1.6 data lands next to the T=1.0 data. Refuse instead.
    if len(temps) > 1:
        raise SystemExit(
            f"[majk] {pattern!r} spans temperatures {sorted(temps)}. Cells are keyed on "
            f"(id, top_p), so these would merge. Pass --temperature to pick one arm.")

    cells, stats = {}, collections.defaultdict(collections.Counter)
    for key, rs in raw.items():
        ballots = []
        for r in rs:
            stopped = r.get("finish_reason") == "stop"
            a = parse_answer(r["text"], r.get("n_options", 10)) if stopped else None
            ballots.append(a)
            s = stats[key[1]]
            s["n"] += 1
            s["trunc"] += (not stopped)
            s["unparsed"] += (stopped and a is None)
        cells[key] = (ballots, rs[0]["gold"])
    return cells, stats, (sorted(cfgs) or ["<unstamped>"]), files


def majk_cell(ballots, gold, k, B, rng, counting="spoiled"):
    """P(plurality of a random size-k draw of ballots == gold)."""
    if counting == "exclude":
        pool = [a for a in ballots if a is not None]
        if not pool:
            return 0.0
    elif counting == "sentinel":
        pool = [a if a is not None else SENTINEL for a in ballots]
    else:
        pool = list(ballots)

    if k == 1:
        # spoiled ballots count as wrong -- that is the point of the rule
        return sum(a == gold for a in pool) / len(pool)
    if k >= len(pool):
        valid = [a for a in pool if a is not None]
        if not valid:
            return 0.0
        return float(collections.Counter(valid).most_common(1)[0][0] == gold)

    hits = 0
    for _ in range(B):
        s = rng.sample(pool, k)
        valid = [a for a in s if a is not None]
        if valid and collections.Counter(valid).most_common(1)[0][0] == gold:
            hits += 1
    return hits / B


def build_matrix(cells, ks, B, seed, counting):
    """-> (top_ps, questions, {k: ndarray[n_questions, n_top_ps]}).

    Balanced by construction: only questions present at EVERY top_p contribute, and
    under the ballot rule every such question contributes at every k.
    """
    rng = random.Random(seed)
    top_ps = sorted({p for _, p in cells})
    qs_by_p = {p: {q for q, pp in cells if pp == p} for p in top_ps}
    qs = sorted(set.intersection(*qs_by_p.values()))
    mats = {}
    for k in ks:
        M = np.empty((len(qs), len(top_ps)))
        for i, q in enumerate(qs):
            for j, p in enumerate(top_ps):
                M[i, j] = majk_cell(*cells[(q, p)], k, B, rng, counting)
        mats[k] = M
    return top_ps, qs, mats


def _two_lines(ps, means):
    """Interior peak by the two-lines rule: split at argmax, left slope>0, right slope<0."""
    imax = int(np.argmax(means))
    if imax == 0 or imax == len(ps) - 1:
        return False, imax
    lp, lm = ps[:imax + 1], means[:imax + 1]
    rp, rm = ps[imax:], means[imax:]
    ls = np.polyfit(lp, lm, 1)[0]
    rs = np.polyfit(rp, rm, 1)[0]
    return bool(ls > 0 and rs < 0), imax


def shape_test(ps, M, B=20000, seed=0):
    """Bootstrap over questions, preserving pairing. -> dict of pre-registered stats."""
    ps = np.asarray(ps, float)
    rng = np.random.default_rng(seed)
    n = M.shape[0]
    obs_means = M.mean(0)
    obs_shape, obs_imax = _two_lines(ps, obs_means)
    a_obs = np.polyfit(ps, obs_means, 2)[0]

    idx = rng.integers(0, n, size=(B, n))
    boot = M[idx].mean(1)                       # [B, n_top_ps]
    a_boot = np.polyfit(ps, boot.T, 2)[0]
    tl = np.array([_two_lines(ps, boot[b])[0] for b in range(B)])
    n_shape = int(tl.sum())
    # JOINT criterion. Two-lines alone is too easy to pass on this grid: if the argmax
    # lands on the second-to-last point, the "right slope < 0" arm is fitted to just two
    # points, so it reduces to the sign of one noisy difference and any curve peaking
    # there passes automatically. Requiring the SAME bootstrap replicate to also produce a
    # down-opening quadratic makes the claim about the whole curve's shape, not about one
    # adjacent pair. Reported alongside, never instead of, the two components.
    p_joint = float((tl & (a_boot < 0)).mean())

    # paired SE of each top_p vs the observed argmax column
    d = M - M[:, [obs_imax]]
    with np.errstate(invalid="ignore"):
        se_paired = d.std(0, ddof=1) / np.sqrt(n)
    return {
        "means": obs_means.tolist(),
        "se_marginal": (M.std(0, ddof=1) / np.sqrt(n)).tolist(),
        "se_paired_vs_peak": se_paired.tolist(),
        "argmax_top_p": float(ps[obs_imax]),
        "argmax_is_interior": bool(0 < obs_imax < len(ps) - 1),
        "shape_holds_observed": bool(obs_shape),
        "p_shape": n_shape / B,
        "p_shape_and_concave": p_joint,
        "quad_a": float(a_obs),
        "p_quad_negative": float((a_boot < 0).mean()),
        "left_segment_points": int(obs_imax + 1),
        "right_segment_points": int(len(ps) - obs_imax),
        "thin_segment": bool(min(obs_imax + 1, len(ps) - obs_imax) < 3),
        "n_questions": int(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="cots/*_gen.shard*.jsonl*")
    ap.add_argument("--ks", default=",".join(map(str, DEFAULT_KS)))
    ap.add_argument("--boot", type=int, default=400, help="draws per cell for the maj@k estimate")
    ap.add_argument("--shape-boot", type=int, default=20000, help="bootstrap reps over questions")
    ap.add_argument("--counting", default="spoiled", choices=["spoiled", "sentinel", "exclude"])
    ap.add_argument("--top-ps", default="",
                    help="comma-sep top_p to analyze (default: all present). Use this to drop "
                         "partially-generated cells, which otherwise shrink the balanced set.")
    ap.add_argument("--min-top-p", type=float, default=0.0,
                    help="HANDOFF 2.2 recommends 0.4 for the neutral config, where top_p<0.4 is "
                         "a repetition-collapse regime. Irrelevant when penalties suppress it.")
    ap.add_argument("--informative-only", action="store_true",
                    help="drop questions answered all-correct or all-wrong at every top_p")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    cells, stats, cfgs, files = load_cells(args.glob)
    if args.top_ps:
        want = {float(x) for x in args.top_ps.split(",") if x.strip()}
        cells = {k: v for k, v in cells.items() if k[1] in want}
    if args.min_top_p:
        cells = {k: v for k, v in cells.items() if k[1] >= args.min_top_p}

    print(f"[majk] {len(files)} file(s), sampling config profile(s): {cfgs}")
    print(f"[majk] counting rule = {args.counting}")
    # HANDOFF section 3: no analysis may silently exclude a generated sample. Print the
    # per-top_p spoil rates in the artifact itself so the differential is always visible.
    print(f"\n{'top_p':>6} {'gens':>6} {'trunc':>7} {'unparsed':>9} {'spoiled%':>9}")
    for p in sorted(stats):
        s = stats[p]
        sp = 100 * (s["trunc"] + s["unparsed"]) / s["n"]
        print(f"{p:>6} {s['n']:>6} {s['trunc']:>7} {s['unparsed']:>9} {sp:>8.2f}%")

    top_ps, qs, mats = build_matrix(cells, ks, args.boot, args.seed, args.counting)

    keep = np.arange(len(qs))
    if args.informative_only:
        M1 = mats[1]
        keep = np.where(~(np.all(M1 == 1.0, axis=1) | np.all(M1 == 0.0, axis=1)))[0]
        print(f"\n[majk] informative subset: {len(keep)}/{len(qs)} questions "
              f"(dropped {len(qs)-len(keep)} always-right/always-wrong)")

    print(f"\n[majk] {len(keep)} questions balanced across top_p {top_ps}")
    print(f"\n{'k':>3}  " + " ".join(f"{('p='+str(p)):>8}" for p in top_ps)
          + f"  {'argmax':>7} {'P(2line)':>9} {'P(a<0)':>7} {'P(both)':>8} {'quad a':>8}")
    print("-" * (5 + 9 * len(top_ps) + 45))
    results = {}
    for k in ks:
        M = mats[k][keep]
        st = shape_test(top_ps, M, args.shape_boot, args.seed)
        results[k] = st
        flag = ""
        if st["p_shape_and_concave"] >= 0.95:
            flag = "  <-- INVERTED-U"
        elif st["p_shape"] >= 0.95:
            flag = "  (2-lines only" + ("; THIN segment" if st["thin_segment"] else "") + ")"
        print(f"{k:>3}  " + " ".join(f"{m:>8.4f}" for m in st["means"])
              + f"  {st['argmax_top_p']:>7} {st['p_shape']:>9.3f} {st['p_quad_negative']:>7.3f}"
              f" {st['p_shape_and_concave']:>8.3f} {st['quad_a']:>8.4f}{flag}")
    print("\n[majk] pre-registered criterion: interior peak claimed only if the JOINT")
    print("       P(two-lines AND down-opening quadratic) >= 0.95. Two-lines alone can pass")
    print("       on a 2-point right segment, which is one noisy difference, not a shape.")

    if args.json_out:
        json.dump({"counting": args.counting, "cfg_profiles": cfgs, "top_ps": top_ps,
                   "n_questions": int(len(keep)), "informative_only": args.informative_only,
                   "spoil_stats": {str(p): dict(stats[p]) for p in sorted(stats)},
                   "results": {str(k): results[k] for k in results}},
                  open(args.json_out, "w"), indent=1)
        print(f"[majk] -> {args.json_out}")


if __name__ == "__main__":
    main()
