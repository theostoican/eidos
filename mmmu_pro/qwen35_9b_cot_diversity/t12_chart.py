#!/usr/bin/env python
"""T=1.2 / top_k=-1 / presence 1.5 top_p result as one figure, in the style of result_chart.py:
maj@1 (+-1 SEM) and maj@8 vs top_p, read from analyze.py's RESULT_T12_SWEEP_FULL.json, with the
model-card defaults (T=1.0, top_p .95, top_k 20, presence 1.5) as dashed reference lines computed
from the baseline traces with analyze.py's own majk_cell (every question with 8 ballots).

  python t12_chart.py [--sweep outputs/RESULT_T12_SWEEP_FULL.json] [--card outputs/baseline_card.jsonl]
                      [--out outputs/t12_topp_result.png]
"""
import argparse, collections, glob, gzip, json, random, sys
import numpy as np
sys.path.insert(0, "../common")
from analyze import parse_answer, majk_cell
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
SAMPLES, K = 8, 8

def row(d, k):
    return next(r for r in d["results"] if r["k"] == k)

def balanced_questions(grid, files=("cots/t12_*.jsonl.gz",)):
    """The questions analyze.py's matrix is built on: those with a cell at every grid top_p."""
    seen = collections.defaultdict(set)
    for f in (x for pat in files for x in (sorted(glob.glob(pat)) or [pat])):
        for l in (gzip.open(f, "rt") if f.endswith(".gz") else open(f)):
            try: r = json.loads(l)
            except Exception: continue
            seen[r["id"]].add(r["top_p"])
    return {q for q, ps in seen.items() if set(grid) <= ps}

def card_reference(path, qs):
    """maj@1 / maj@8 of the card baseline (ballot rule) on exactly the sweep's balanced questions,
    so the reference lines and the curves are paired on the same questions at any stage."""
    cells, gold = collections.defaultdict(list), {}
    files = sorted(glob.glob(path)) or [path]
    for l in (l for f in files for l in (gzip.open(f, "rt") if f.endswith(".gz") else open(f))):
        try: r = json.loads(l)
        except Exception: continue
        if r["id"] not in qs: continue
        a = parse_answer(r["text"], r["n_options"]) if r["finish_reason"] == "stop" else None
        cells[r["id"]].append(a); gold[r["id"]] = r["gold"]
    qs = [q for q, b in cells.items() if len(b) == SAMPLES]
    rng = random.Random(0)
    return (np.mean([majk_cell(cells[q], gold[q], 1, 400, rng) for q in qs]),
            np.mean([majk_cell(cells[q], gold[q], K, 400, rng) for q in qs]), len(qs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="outputs/RESULT_T12_SWEEP_FULL.json")
    ap.add_argument("--card", default="cots/card_t10.shard*.jsonl.gz")
    ap.add_argument("--out", default="outputs/t12_topp_result.png")
    a = ap.parse_args()
    d = json.load(open(a.sweep))
    grid = d["grid"]
    qs = balanced_questions(grid)
    n_q = len(qs)
    r1, rk = row(d, 1), row(d, K)
    m1, s1, mk = np.array(r1["means"]), np.array(r1["sem"]), np.array(rk["means"])
    c1, ck, n_c = card_reference(a.card, qs)

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150); fig.patch.set_facecolor(SURF)
    ax.set_facecolor(SURF); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.fill_between(grid, m1 - s1, m1 + s1, color=BLUE, alpha=0.12, lw=0)
    ax.plot(grid, m1, color=BLUE, lw=1.8, marker="o", ms=5.5, label="maj@1 (±1 SEM)")
    ax.plot(grid, mk, color=ORANGE, lw=1.8, marker="o", ms=5.5, label=f"maj@{K}")
    ax.axhline(c1, color=BLUE, lw=1.1, ls="--", alpha=0.8, label=f"model card maj@1 ({c1:.3f})")
    ax.axhline(ck, color=ORANGE, lw=1.1, ls="--", alpha=0.8, label=f"model card maj@{K} ({ck:.3f})")
    for j, (r, m, col) in enumerate(((r1, m1, BLUE), (rk, mk, ORANGE))):
        i = int(np.argmax(m))
        ax.axvline(grid[i], color=col, lw=0.9, ls=":", alpha=0.9, zorder=1)
        ax.annotate(f"maj@{r['k']}  argmax {grid[i]} · {m[i]:.4f}  ({(m[i] - (c1 if r['k'] == 1 else ck)) * 100:+.2f} pp vs card)",
                    (0, 1), xycoords="axes fraction", textcoords="offset points",
                    xytext=(0, 15 - 12 * j), ha="left", va="bottom", color=col, fontsize=7.5)
    ax.set_title(f"Qwen3.5-9B · T=1.2, top_k off, presence 1.5 · n={n_q} questions, paired with the card",
                 loc="left", color=INK, fontsize=9, pad=30)
    ax.set_xlabel("top_p"); ax.set_ylabel("accuracy")
    ax.margins(x=0.06, y=0.14)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    fig.suptitle(f"maj@1 and maj@{K} vs top_p at T=1.2, against the model-card defaults",
                 x=0.01, ha="left", color=INK, fontsize=10.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(a.out, facecolor=SURF)
    print(f"[chart] -> {a.out}  (n={n_q} balanced questions; card reference on {n_c} of them: "
          f"maj@1 {c1:.4f}, maj@{K} {ck:.4f})")

if __name__ == "__main__":
    main()
