#!/usr/bin/env python
"""Stage 4a: visual-premise soundness vs top_p at T=1.0, and per-sample accuracy.

METRIC. Per (question, top_p) cell, pool every premise judged across that cell's samples and
take sound / (sound + unsound). UNVERIFIABLE is excluded from the denominator (extraction
slips, not visual errors); its rate is reported per top_p. The headline is the mean over
QUESTIONS of the cell rate, which weights every question equally and keeps the design paired,
exactly like the accuracy analysis it is plotted against.

TRUNCATED TRACES ARE IN. A truncated trace made visual reads before running out of budget;
excluding them subsets the data along the swept axis (11.09% truncated at top_p=0.5 vs 0.42%
at 1.0). The drop-truncated variant is drawn as a dashed line so the size of that bias is on
the chart rather than in a footnote.

ACCURACY. maj@1 under the ballot rule -- a spoiled ballot counts as wrong -- recomputed here
from the same traces, and CHECKED against outputs/RESULT_T10.json, aborting on a mismatch so
this arm can never silently plot a different accuracy definition than the repo's own.
"""
import argparse, collections, glob, gzip, json, sys
from pathlib import Path
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from analyze import rm_anova, shape_test                     # same tests, same code path

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"


def cell_table(rows, keep_truncated=True):
    """-> {(qid, top_p): [n_sound, n_unsound, n_unverifiable, n_traces]}"""
    t = collections.defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        if not keep_truncated and r.get("finish_reason") != "stop":
            continue
        c = t[(r["id"], r["top_p"])]
        c[0] += r["n_sound"]; c[1] += r["n_unsound"]; c[2] += r["n_unverifiable"]; c[3] += 1
    return t


def curve(cells, top_ps):
    """Paired question x top_p matrix of cell soundness, plus the premise-pooled rate. Only
    questions with >=1 judged premise at EVERY top_p enter, so the tests compare like with
    like -- at the cost of dropping ~40 questions, which is why the plotted accuracy line
    sits ~2 pp below the repo's full-set curve."""
    qs = sorted({q for q, _ in cells})
    ok = [q for q in qs
          if all((cells.get((q, p), [0, 0, 0, 0])[0] + cells.get((q, p), [0, 0, 0, 0])[1]) > 0
                 for p in top_ps)]
    M = np.array([[cells[(q, p)][0] / (cells[(q, p)][0] + cells[(q, p)][1]) for p in top_ps]
                  for q in ok]) if ok else np.zeros((0, len(top_ps)))
    pooled, unver = [], []
    for p in top_ps:
        s = sum(cells[(q, p)][0] for q in qs if (q, p) in cells)
        u = sum(cells[(q, p)][1] for q in qs if (q, p) in cells)
        v = sum(cells[(q, p)][2] for q in qs if (q, p) in cells)
        pooled.append(s / max(s + u, 1))
        unver.append(v / max(s + u + v, 1))
    return M, ok, np.array(pooled), np.array(unver)


def accuracy_matrix(layer_glob, top_ps, layers=None):
    """maj@1 under the ballot rule, from the layer files: spoiled counts as wrong."""
    acc = collections.defaultdict(lambda: [0, 0])
    for f in sorted(glob.glob(layer_glob)):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if layers is not None and r["sample_idx"] not in layers:
                continue
            a = acc[(r["id"], r["top_p"])]
            a[0] += bool(r["answer_correct"]); a[1] += 1
    qs = sorted({q for q, _ in acc})
    return np.array([[acc[(q, p)][0] / acc[(q, p)][1] for p in top_ps] for q in qs]), qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdicts", default="outputs/vp_verdicts.jsonl")
    ap.add_argument("--premises", default="outputs/vp_premises.jsonl")
    ap.add_argument("--layers", default="outputs/vp_layers/layer*.jsonl.gz")
    ap.add_argument("--reference", default="outputs/RESULT_T10.json")
    ap.add_argument("--grid", default="0.5,0.6,0.7,0.8,0.9,0.925,0.95,0.975,0.99,1.0")
    ap.add_argument("--out", default="outputs/RESULT_VP.md")
    ap.add_argument("--png", default="outputs/premise_soundness_vs_topp.png")
    ap.add_argument("--shape-boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    top_ps = [float(x) for x in a.grid.split(",")]

    rows = [json.loads(l) for l in open(a.verdicts)]
    if not rows:
        raise SystemExit("[vp] no verdicts yet")
    layers = sorted({r["sample_idx"] for r in rows})
    # Completeness is measured against the EXTRACTION stage, not the cell count: a trace whose
    # extractor found no premises never reaches the judge, so a finished layer always holds
    # fewer verdicts than it has cells. Comparing verdicts to cells declares every layer
    # incomplete and silently empties the analysis.
    per_layer = collections.Counter(r["sample_idx"] for r in rows)
    extracted, judgeable = collections.Counter(), collections.Counter()
    for line in open(a.premises):
        r = json.loads(line)
        extracted[r["sample_idx"]] += 1
        judgeable[r["sample_idx"]] += int(r["n_premises"] > 0)
    ncell = max(extracted.values()) if extracted else 0
    complete = sorted(s for s in layers
                      if extracted[s] >= 0.999 * ncell and per_layer[s] >= judgeable[s])
    if complete != layers:
        print(f"[vp] layers judged {layers}; complete {complete} -- restricting")
        rows = [r for r in rows if r["sample_idx"] in complete]
    if not rows:
        raise SystemExit("[vp] no fully processed layer yet -- nothing to chart")

    M, qs_ok, pooled, unver = curve(cell_table(rows, True), top_ps)
    M_st, _, pooled_st, _ = curve(cell_table(rows, False), top_ps)
    A, qs_acc = accuracy_matrix(a.layers, top_ps, layers=set(complete))
    common = sorted(set(qs_ok) & set(qs_acc))
    Ai = np.array([A[qs_acc.index(q)] for q in common])
    Mi = np.array([M[qs_ok.index(q)] for q in common])

    F_s, p_s = rm_anova(Mi); F_a, p_a = rm_anova(Ai)
    st_s = shape_test(top_ps, Mi, a.shape_boot, a.seed)
    st_a = shape_test(top_ps, Ai, a.shape_boot, a.seed)
    r_pear, p_pear = stats.pearsonr(Mi.mean(0), Ai.mean(0))
    within = [stats.pearsonr(Mi[i], Ai[i])[0] for i in range(len(common))
              if Mi[i].std() > 0 and Ai[i].std() > 0]

    ref_note, ref_means = "", None
    if Path(a.reference).exists():
        ref = json.load(open(a.reference))
        rrow = next((r for r in ref["results"] if r["k"] == 1), None)
        if rrow and ref.get("grid") == top_ps:
            ref_means = rrow["means"]
            d = np.abs(np.array(rrow["means"]) - A.mean(0)).max()
            ref_note = (f"recomputed maj@1 vs published RESULT_T10 k=1: max |delta| = {d:.4f} "
                        f"over {len(complete)}/16 samples")
            if len(complete) == 16 and d > 1e-9:
                raise SystemExit(f"[vp] recomputed accuracy disagrees with {a.reference} ({d:.5f})")

    clip = collections.defaultdict(lambda: [0, 0])
    for line in open(a.premises):
        r = json.loads(line)
        if r["sample_idx"] in complete:
            c = clip[r["top_p"]]
            c[0] += int((r.get("cot_tokens_kept") or 0) >= (r.get("cot_token_cap") or 10**9))
            c[1] += 1

    o = ["# Visual-premise soundness vs top_p (T=1.0)", "",
         f"Samples judged per cell: {len(complete)}/16 (`sample_idx` {complete}).",
         f"Questions: {len(common)} paired at every top_p.",
         f"Traces judged: {len(rows)} | premises: "
         f"{sum(r['n_sound']+r['n_unsound']+r['n_unverifiable'] for r in rows)}.", ref_note, "",
         "Judge: InternVL3-8B-AWQ, no gold answer shown. Extractor: Qwen3.5-9B, temp 0.", "",
         "| top_p | " + " | ".join(str(p) for p in top_ps) + " |",
         "|---|" + "---|" * len(top_ps),
         "| **premise soundness** (per-question mean) | "
         + " | ".join(f"{v:.4f}" for v in Mi.mean(0)) + " |",
         "| +/- 1 SEM | " + " | ".join(
             f"{v:.4f}" for v in Mi.std(0, ddof=1) / np.sqrt(len(Mi))) + " |",
         "| premise-pooled soundness | " + " | ".join(f"{v:.4f}" for v in pooled) + " |",
         "| soundness, truncated dropped | " + " | ".join(f"{v:.4f}" for v in pooled_st) + " |",
         "| **per-sample accuracy** (maj@1, ballot rule) | "
         + " | ".join(f"{v:.4f}" for v in Ai.mean(0)) + " |",
         "| unverifiable share of verdicts | " + " | ".join(f"{v:.3f}" for v in unver) + " |",
         "| traces clipped at the extractor token cap | " + " | ".join(
             f"{clip[p][0]/max(clip[p][1],1):.3f}" for p in top_ps) + " |", "",
         f"- premise soundness: RM-ANOVA F={F_s:.2f}, p={p_s:.4g}; argmax {st_s['argmax']}; "
         f"quad a={st_s['quad_a']:+.4f}",
         f"- per-sample accuracy: RM-ANOVA F={F_a:.2f}, p={p_a:.4g}; argmax {st_a['argmax']}; "
         f"quad a={st_a['quad_a']:+.4f}",
         f"- across the grid the two curves correlate r={r_pear:+.3f} (p={p_pear:.3g})",
         f"- within a question, mean r(soundness, accuracy) = {np.mean(within):+.3f} "
         f"(n={len(within)})"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    open(a.out, "w").write("\n".join(o) + "\n")
    json.dump({"grid": top_ps, "n_questions": len(common), "layers": complete,
               "soundness_mean": Mi.mean(0).tolist(),
               "soundness_sem": (Mi.std(0, ddof=1) / np.sqrt(len(Mi))).tolist(),
               "soundness_pooled": pooled.tolist(),
               "soundness_pooled_drop_truncated": pooled_st.tolist(),
               "unverifiable_share": unver.tolist(),
               "accuracy_mean": Ai.mean(0).tolist(),
               "accuracy_sem": (Ai.std(0, ddof=1) / np.sqrt(len(Ai))).tolist(),
               "anova": {"soundness": [F_s, p_s], "accuracy": [F_a, p_a]},
               "shape": {"soundness": st_s, "accuracy": st_a},
               "r_across_grid": [r_pear, p_pear], "r_within_question": float(np.mean(within))},
              open(a.out.replace(".md", ".json"), "w"), indent=1)
    print("\n".join(o))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for ys, es, c, m, lab in (
            (Mi.mean(0), Mi.std(0, ddof=1) / np.sqrt(len(Mi)), ORANGE, "s",
             "visual-premise soundness (fraction of premises the judge confirms)"),
            (Ai.mean(0), Ai.std(0, ddof=1) / np.sqrt(len(Ai)), BLUE, "o",
             "per-sample accuracy, maj@1 (spoiled ballot = wrong)")):
        ax.plot(top_ps, ys, color=c, lw=2, marker=m, ms=8, label=lab,
                markeredgecolor=SURF, markeredgewidth=2, zorder=3)
        ax.errorbar(top_ps, ys, yerr=es, fmt="none", ecolor=c, elinewidth=1.4,
                    capsize=3, capthick=1.4, alpha=0.75, zorder=2)
    ax.plot(top_ps, pooled_st, color=ORANGE, lw=1.4, ls="--", alpha=0.65, zorder=2,
            label="soundness, truncated traces dropped (the biased rule)")
    if ref_means is not None:
        ax.plot(top_ps, ref_means, color=BLUE, lw=1.4, ls=":", alpha=0.7, zorder=2,
                label="maj@1, all 345 questions x 16 samples (RESULT_T10, for reference)")
    ax.set_title("Visual-premise soundness and answer accuracy vs top_p\n"
                 "Qwen3.5-9B, MMMU-Pro standard (10 options), T=1.0",
                 color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel("top_p", color=INK2, fontsize=9)
    ax.set_ylabel("fraction correct / sound", color=INK2, fontsize=9)
    ax.grid(True, color=GRID, lw=0.8, alpha=0.7); ax.set_axisbelow(True)
    for sp_ in ax.spines.values():
        sp_.set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.margins(x=0.06, y=0.18)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="best")
    fig.text(0.012, 0.03, f"Solid lines: the {len(common)} of 345 questions with a judged "
             f"premise at every top_p, x {len(complete)} samples; +/-1 SEM over questions.",
             color=INK2, fontsize=7)
    fig.text(0.012, 0.008, "Judge: InternVL3-8B-AWQ, gold answer withheld. Extractor: "
             "Qwen3.5-9B, temp 0, first 6,000 tokens per trace.", color=INK2, fontsize=7)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(a.png, dpi=170, facecolor=SURF)
    print(f"[vp] -> {a.out} , {a.png}")


if __name__ == "__main__":
    main()
