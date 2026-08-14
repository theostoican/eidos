#!/usr/bin/env python
"""top_p sweep analysis: ballot-model maj@k + the pre-registered shape tests.

COUNTING. There is exactly one counting rule and it is not selectable. Every
(question, top_p) cell has exactly n_samples BALLOTS, one per generated sample. A ballot is
VALID if the trace terminated and an answer parsed; otherwise it is SPOILED: it consumes
vote budget and does not vote, and a draw with zero valid ballots is a failure. Truncated
generations are FAILURES -- they consumed inference and produced no answer -- and are never
excluded from a reported result. That is not a preference: truncation is strongly
top_p-dependent (9.2% at top_p=0.5 vs 0.07% at 1.0 for T=1.0), so dropping those traces
compares different subsets of samples across the very axis being swept, and on its own
manufactured the interior peak this work set out to check.

TESTS (pre-registered, applied identically to every arm):
  omnibus first    repeated-measures ANOVA over the top_p levels. A null omnibus is
                   reported prominently -- pairwise tests after one are not evidence.
  shape            two-lines split at the argmax, bootstrapped over QUESTIONS (B=20,000)
                   with the same resample applied to every top_p column so pairing holds.
                   alpha=0.05 -> the claim stands only at P >= 0.95.
  joint            two-lines alone is too easy to pass: with the argmax on the
                   second-to-last grid point the right arm is fitted to two points and
                   reduces to the sign of one noisy difference. The reported figure
                   requires the SAME bootstrap replicate to also yield a down-opening
                   quadratic.
  multiplicity     pairwise tests against the peak are Holm-corrected; raw and adjusted
                   p both reported, never only the winning comparison.

Every question is analysed, and there is no subsetting knob. Every test here is paired --
repeated-measures ANOVA, paired t-tests, and a bootstrap that applies the same question
resample to every top_p column -- so a question answered identically at every top_p already
contributes nothing to the treatment effect and nothing to the error term. Dropping such
questions cannot sharpen a within-subject test; it only spends degrees of freedom.
"""
import argparse, collections, glob, gzip, json, random, re
import numpy as np
from scipy import stats

KS = [1, 16]
LETTERS = [chr(ord("A") + i) for i in range(26)]
ANS_RE = re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)


def parse_answer(text, n_options):
    """Official MMMU-Pro format ('Answer: $LETTER'), else the last bare option letter.
    n_options is per question: despite the 'standard (10 options)' config name only ~70%
    of questions have 10, so the valid-letter set cannot be assumed."""
    valid = set(LETTERS[:n_options])
    after = text.split("</think>")[-1] if "</think>" in text else text
    for c in reversed(ANS_RE.findall(after) or ANS_RE.findall(text)):
        if c.upper() in valid:
            return c.upper()
    for ch in reversed(re.findall(r"\b([A-J])\b", after)):
        if ch in valid:
            return ch
    return None


def load_cells(pattern, temperature=None):
    """-> ({(id, top_p): (ballots, gold)}, spoil stats, cfg profiles, files).
    ballots has one entry per generated sample; None marks a spoiled ballot."""
    raw, files = collections.defaultdict(list), sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"[analyze] no files match {pattern!r}")
    cfgs, temps = set(), set()
    for f in files:
        for line in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if temperature is not None and r.get("temperature", 1.0) != temperature:
                continue
            raw[(r["id"], r["top_p"])].append(r)
            temps.add(r.get("temperature", 1.0))
            cfgs.add(r.get("cfg_profile", "<unstamped>"))
    # Cells are keyed on (id, top_p) only, so a glob spanning two temperatures would merge
    # them into 32-ballot cells and blend two experiments into one curve. Refuse instead.
    if len(temps) > 1:
        raise SystemExit(f"[analyze] {pattern!r} spans temperatures {sorted(temps)}; "
                         f"pass --temperature to select one arm.")
    cells, spoil = {}, collections.defaultdict(collections.Counter)
    for key, rs in raw.items():
        ballots = []
        for r in rs:
            stopped = r.get("finish_reason") == "stop"
            a = parse_answer(r["text"], r.get("n_options", 10)) if stopped else None
            ballots.append(a)
            s = spoil[key[1]]
            s["n"] += 1
            s["trunc"] += (not stopped)
            s["unparsed"] += (stopped and a is None)
        cells[key] = (ballots, rs[0]["gold"])
    # Cells are accumulated across every file in the glob, and nothing upstream checks how many
    # ballots landed in one. A cell present in TWO sources -- a committed trace set and a
    # re-generated shard covering the same (id, top_p) -- silently becomes a 32-ballot cell and
    # blends two runs into a single curve, the same failure the temperature guard above refuses.
    counts = collections.Counter(len(b) for b, _ in cells.values())
    modal = counts.most_common(1)[0][0]
    over = {k: len(v[0]) for k, v in cells.items() if len(v[0]) > modal}
    under = {k: len(v[0]) for k, v in cells.items() if len(v[0]) < modal}
    if over:
        ex = ", ".join(f"{k}->{n}" for k, n in sorted(over.items())[:8])
        raise SystemExit(f"[analyze] {len(over)} cell(s) exceed the modal {modal} ballots: {ex}"
                         f"{' ...' if len(over) > 8 else ''}\nDe-duplicate the sources or narrow "
                         f"--glob; merging duplicated cells would blend two runs into one curve.")
    if under:
        print(f"[analyze] WARNING: {len(under)} cell(s) below the modal {modal} ballots "
              f"(incomplete generation). They are analysed as-is, on a smaller denominator.")
    return cells, spoil, sorted(cfgs), files


def majk_cell(ballots, gold, k, B, rng):
    pool = list(ballots)
    if k == 1:                                   # spoiled ballots count as wrong
        return sum(a == gold for a in pool) / len(pool)
    if k >= len(pool):
        valid = [a for a in pool if a is not None]
        return float(bool(valid) and collections.Counter(valid).most_common(1)[0][0] == gold)
    hits = 0
    for _ in range(B):
        valid = [a for a in rng.sample(pool, k) if a is not None]
        hits += bool(valid) and collections.Counter(valid).most_common(1)[0][0] == gold
    return hits / B


def build_matrix(cells, ks, B, seed):
    """-> (top_ps, questions, {k: ndarray[n_questions, n_top_ps]}). Balanced by
    construction: only questions present at every top_p contribute, and under the ballot
    rule every such question contributes at every k."""
    rng = random.Random(seed)
    top_ps = sorted({p for _, p in cells})
    qs = sorted(set.intersection(*({q for q, pp in cells if pp == p} for p in top_ps)))
    return top_ps, qs, {k: np.array([[majk_cell(*cells[(q, p)], k, B, rng)
                                      for p in top_ps] for q in qs]) for k in ks}


def _two_lines(ps, means):
    imax = int(np.argmax(means))
    if imax in (0, len(ps) - 1):
        return False, imax
    return bool(np.polyfit(ps[:imax + 1], means[:imax + 1], 1)[0] > 0
                and np.polyfit(ps[imax:], means[imax:], 1)[0] < 0), imax


def shape_test(ps, M, B, seed):
    ps = np.asarray(ps, float)
    obs = M.mean(0)
    _, imax = _two_lines(ps, obs)
    boot = M[np.random.default_rng(seed).integers(0, M.shape[0], (B, M.shape[0]))].mean(1)
    a_boot = np.polyfit(ps, boot.T, 2)[0]
    tl = np.array([_two_lines(ps, boot[b])[0] for b in range(B)])
    return {"means": obs.tolist(), "sem": (M.std(0, ddof=1) / np.sqrt(M.shape[0])).tolist(),
            "argmax": float(ps[imax]),
            "p_shape": float(tl.mean()), "p_joint": float((tl & (a_boot < 0)).mean()),
            "quad_a": float(np.polyfit(ps, obs, 2)[0]),
            "p_quad_negative": float((a_boot < 0).mean()),
            "thin_segment": bool(min(imax + 1, len(ps) - imax) < 3)}


def rm_anova(M):
    n, k = M.shape
    g = M.mean()
    ss_l = n * ((M.mean(0) - g) ** 2).sum()
    ss_e = ((M - g) ** 2).sum() - ss_l - k * ((M.mean(1) - g) ** 2).sum()
    df_l, df_e = k - 1, (k - 1) * (n - 1)
    if ss_e <= 0 or df_e <= 0:
        return float("nan"), float("nan")
    F = (ss_l / df_l) / (ss_e / df_e)
    return float(F), float(stats.f.sf(F, df_l, df_e))


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty(len(p)), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def analyse(cells, grid, ks, boot, shape_boot, seed, out):
    top_ps, qs, mats = build_matrix({k: v for k, v in cells.items() if k[1] in set(grid)},
                                    ks, boot, seed)
    out.append(f"\n### n = {len(qs)} questions | grid = {top_ps}\n")
    out.append("| k | " + " | ".join(f"p={p}" for p in top_ps)
               + " | argmax | F | p(omni) | P(shape) | P(joint) | quad a | best raw/Holm p |")
    out.append("|" + "---|" * (len(top_ps) + 8))
    rows, optima = [], {}
    for k in ks:
        M = mats[k]
        F, p_om = rm_anova(M)
        st = shape_test(top_ps, M, shape_boot, seed)
        imax = int(np.argmax(st["means"]))
        optima[k] = top_ps[imax]
        raw = [stats.ttest_rel(M[:, imax], M[:, j]).pvalue
               if (M[:, imax] - M[:, j]).std() > 0 else 1.0
               for j in range(len(top_ps)) if j != imax]
        adj = holm(raw)
        rows.append({"k": k, **st, "F": F, "p_omnibus": p_om,
                     "best_raw_p": float(min(raw)), "best_holm_p": float(min(adj))})
        out.append(f"| {k} | " + " | ".join(f"{m:.4f}" for m in st["means"])
                   + f" | {st['argmax']} | {F:.2f} | {p_om:.3f} | {st['p_shape']:.3f}"
                   f" | {st['p_joint']:.3f} | {st['quad_a']:+.4f}"
                   f" | {min(raw):.4f} / {min(adj):.4f} |")
    mono = all(optima[ks[i]] <= optima[ks[i + 1]] for i in range(len(ks) - 1))
    out.append(f"\noptimum vs k: {', '.join(f'k={k}->{optima[k]}' for k in ks)} "
               f"-- pre-declared non-decreasing in k: **{'HOLDS' if mono else 'VIOLATED'}**")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--temperature", type=float, default=None,
                    help="select one temperature arm; required when arms share a directory")
    ap.add_argument("--grid", required=True, help="comma-separated top_p to analyse")
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--shape-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    grid = [float(x) for x in a.grid.split(",")]
    ks = [int(x) for x in a.ks.split(",")]

    cells, spoil, cfgs, files = load_cells(a.glob, a.temperature)
    out = ["# top_p sweep -- pre-registered analysis", "",
           f"Sources: {len(files)} file(s). Config profile(s) in data: `{cfgs}`.",
           f"Temperature arm: {a.temperature if a.temperature else 1.0} | grid: {grid}", "",
           "## Spoil rates per top_p", "",
           "| top_p | generations | truncated | unparseable | spoiled % |", "|---|---|---|---|---|"]
    for p in [p for p in sorted(spoil) if p in grid]:
        s = spoil[p]
        out.append(f"| {p} | {s['n']} | {s['trunc']} | {s['unparsed']} | "
                   f"{100*(s['trunc']+s['unparsed'])/s['n']:.2f}% |")
    res = {"grid": grid, "ks": ks, "temperature": a.temperature or 1.0, "cfg_profiles": cfgs,
           "spoil": {str(p): dict(spoil[p]) for p in sorted(spoil) if p in grid}}
    out.append("\n## Results")
    res["results"] = analyse(cells, grid, ks, a.boot, a.shape_boot, a.seed, out)

    open(a.out, "w").write("\n".join(out) + "\n")
    json.dump(res, open(a.out.replace(".md", ".json"), "w"), indent=1, default=float)
    print("\n".join(out))
    print(f"\n[analyze] -> {a.out}")


if __name__ == "__main__":
    main()
