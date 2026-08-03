#!/usr/bin/env python
"""HANDOFF Phase 1 -- the pre-registered re-analysis (section 4). NO GPU REQUIRED.

Runs the whole of HANDOFF section 4 on the existing neutral-config traces, in the order the
handoff specifies, and writes the numbers down before anyone looks at a chart:

  omnibus first   repeated-measures ANOVA over the top_p levels. If it does not reject,
                  that is said prominently -- pairwise tests after a null omnibus are not
                  evidence (HANDOFF 4, and the reason the original t=2.37 / t=2.09 do not
                  survive).
  shape test      two-lines split at the argmax, bootstrapped over QUESTIONS with the same
                  resample applied to every top_p column so pairing is preserved, B=20,000.
                  Pre-declared alpha = 0.05: the claim stands only if P(shape) >= 0.95.
  secondary       sign of the quadratic coefficient under the same bootstrap; and the
                  optimum-vs-k relationship, pre-declared to be NON-DECREASING in k if a
                  coverage-vs-precision tradeoff is real.
  multiplicity    pairwise paired t-tests against the peak are Holm-corrected across all
                  comparisons, raw and adjusted p both reported. Never only the winner.
  power           reported on the full set AND the informative subset (questions not
                  answered identically by all 16 samples at every top_p).

Grid is HANDOFF 2.2: top_p > 0.4 only. Counting is HANDOFF 2.4: spoiled ballot primary,
sentinel and exclude reported alongside so the filtering delta is visible.
"""
import argparse, json
import numpy as np
from scipy import stats

from majk import load_cells, build_matrix, shape_test, _two_lines

KS = [1, 2, 4, 8, 16]
PRIMARY_GRID = [0.5, 0.7, 0.9, 0.95, 1.0]


def rm_anova(M):
    """Repeated-measures ANOVA over the columns (top_p levels) of M[n_questions, n_levels]."""
    n, k = M.shape
    grand = M.mean()
    ss_levels = n * ((M.mean(0) - grand) ** 2).sum()
    ss_subj = k * ((M.mean(1) - grand) ** 2).sum()
    ss_total = ((M - grand) ** 2).sum()
    ss_err = ss_total - ss_levels - ss_subj
    df_l, df_e = k - 1, (k - 1) * (n - 1)
    if ss_err <= 0 or df_e <= 0:
        return float("nan"), float("nan"), df_l, df_e
    F = (ss_levels / df_l) / (ss_err / df_e)
    return float(F), float(stats.f.sf(F, df_l, df_e)), df_l, df_e


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def analyse(cells, grid, ks, counting, boot, shape_boot, seed, informative_only, out):
    top_ps, qs, mats = build_matrix(
        {kk: v for kk, v in cells.items() if kk[1] in set(grid)}, ks, boot, seed, counting)

    keep = np.arange(len(qs))
    if informative_only:
        M1 = mats[1]
        keep = np.where(~(np.all(M1 == 1.0, axis=1) | np.all(M1 == 0.0, axis=1)))[0]

    out.append(f"\n### counting = `{counting}`"
               f"{' | informative subset' if informative_only else ' | full set'}"
               f" | n = {len(keep)} questions | grid = {top_ps}\n")

    rows, optima = [], {}
    for k in ks:
        M = mats[k][keep]
        F, p_om, df_l, df_e = rm_anova(M)
        st = shape_test(top_ps, M, shape_boot, seed)
        imax = int(np.argmax(st["means"]))
        optima[k] = top_ps[imax]

        # Holm-corrected pairwise paired t-tests of the peak against every other level
        raw = []
        for j in range(len(top_ps)):
            if j == imax:
                continue
            d = M[:, imax] - M[:, j]
            raw.append(stats.ttest_rel(M[:, imax], M[:, j]).pvalue if d.std() > 0 else 1.0)
        adj = holm(raw)
        worst = f"{min(raw):.4f} / {min(adj):.4f}" if raw else "-"

        rows.append({
            "k": k, "means": st["means"], "argmax": top_ps[imax],
            "interior": st["argmax_is_interior"], "p_shape": st["p_shape"],
            "p_joint": st["p_shape_and_concave"], "quad_a": st["quad_a"],
            "p_quad_neg": st["p_quad_negative"], "thin": st["thin_segment"],
            "F": F, "p_omnibus": p_om, "df": [df_l, df_e],
            "best_raw_p": float(min(raw)) if raw else None,
            "best_holm_p": float(min(adj)) if raw else None,
            "se_paired": st["se_paired_vs_peak"],
        })
        out.append(f"| {k} | " + " | ".join(f"{m:.4f}" for m in st["means"])
                   + f" | {top_ps[imax]} | {F:.2f} | {p_om:.3f} | {st['p_shape']:.3f}"
                   f" | {st['p_shape_and_concave']:.3f} | {st['quad_a']:+.4f} | {worst} |")

    hdr = ("| k | " + " | ".join(f"p={p}" for p in top_ps)
           + " | argmax | F | p(omni) | P(2line) | P(joint) | quad a | best raw/Holm p |")
    sep = "|" + "---|" * (len(top_ps) + 8)
    out.insert(len(out) - len(rows), hdr + "\n" + sep)

    mono = all(optima[ks[i]] <= optima[ks[i + 1]] for i in range(len(ks) - 1))
    out.append(f"\noptimum vs k: {', '.join(f'k={k}->{optima[k]}' for k in ks)} "
               f"-- pre-declared non-decreasing in k: **{'HOLDS' if mono else 'VIOLATED'}**")
    return rows, optima


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="cots/u_gen.shard*.jsonl.gz")
    ap.add_argument("--extra-glob", default="", help="Phase-2 top-up files to merge in")
    ap.add_argument("--grid", default=",".join(map(str, PRIMARY_GRID)))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--boot", type=int, default=400)
    ap.add_argument("--shape-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=None,
                    help="select one temperature arm. REQUIRED once T=1.6 data exists next to "
                         "T=1.0: cells are keyed on (id, top_p), so two arms in one glob would "
                         "merge into 32-ballot cells and blend two experiments into one curve.")
    ap.add_argument("--out", default="outputs/PHASE1_RESULT.md")
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",")]
    ks = [int(x) for x in args.ks.split(",")]

    cells, stats_, cfgs, files = load_cells(args.glob, args.temperature)
    if args.extra_glob:
        extra, estats, ecfgs, efiles = load_cells(args.extra_glob, args.temperature)
        for kk, v in extra.items():
            if kk not in cells:
                cells[kk] = v
        for p, c in estats.items():
            for key, n in c.items():
                stats_[p][key] += n
        files += efiles
        cfgs = sorted(set(cfgs) | set(ecfgs))

    out = ["# HANDOFF Phase 1 -- pre-registered re-analysis (section 4)", "",
           f"Sources: {len(files)} file(s). Sampling config profile(s) found in data: `{cfgs}`.",
           f"Grid (HANDOFF 2.2, top_p > 0.4): {grid}", f"Temperature arm: {args.temperature if args.temperature else 1.0}", "",
           "## Spoil / truncation rates per top_p", "",
           "| top_p | generations | truncated | unparseable | spoiled % |", "|---|---|---|---|---|"]
    for p in sorted(stats_):
        if p not in grid:
            continue
        s = stats_[p]
        out.append(f"| {p} | {s['n']} | {s['trunc']} | {s['unparsed']} | "
                   f"{100*(s['trunc']+s['unparsed'])/s['n']:.2f}% |")

    all_rows = {}
    out.append("\n## Primary: spoiled-ballot counting")
    all_rows["spoiled_full"] = analyse(cells, grid, ks, "spoiled", args.boot,
                                       args.shape_boot, args.seed, False, out)
    out.append("\n## Power: informative subset (HANDOFF 4)")
    all_rows["spoiled_info"] = analyse(cells, grid, ks, "spoiled", args.boot,
                                       args.shape_boot, args.seed, True, out)
    out.append("\n## Robustness: sentinel counting (strictly harsher)")
    all_rows["sentinel_full"] = analyse(cells, grid, ks, "sentinel", args.boot,
                                        args.shape_boot, args.seed, False, out)
    out.append("\n## The old rule, for the delta: exclude-truncated")
    all_rows["exclude_full"] = analyse(cells, grid, ks, "exclude", args.boot,
                                       args.shape_boot, args.seed, False, out)

    txt = "\n".join(out) + "\n"
    open(args.out, "w").write(txt)
    print(txt)
    json.dump({"grid": grid, "ks": ks,
               "temperature": args.temperature if args.temperature else 1.0,
               "cfg_profiles": cfgs,
               "spoil": {str(p): {"n": stats_[p]["n"], "trunc": stats_[p]["trunc"],
                                  "unparsed": stats_[p]["unparsed"]}
                         for p in sorted(stats_) if p in grid},
               **{kk: rows for kk, (rows, _) in all_rows.items()}},
              open(args.out.replace(".md", ".json"), "w"), indent=1, default=float)
    print(f"[phase1] -> {args.out}")


if __name__ == "__main__":
    main()
