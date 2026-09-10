#!/usr/bin/env python
"""Figure for the InternVL3.5-8B T=1.2 arm.  python result_chart_ivl.py [--json outputs/RESULT_IVL35_T12.json] [--out ...]
A: maj@1 (+/-1 SEM) and maj@8 vs top_p.   B: spoiled ballots by top_p.
Single-arm figure: this directory commits the T=1.2 arm only, so nothing here reads another arm's
traces. The accuracy-among-valid decomposition that explains the A/B contrast is in README section 2."""
import argparse, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("--json", default="outputs/RESULT_IVL35_T12.json")
ap.add_argument("--out", default="outputs/ivl35_t12_result.png")
a = ap.parse_args()
BLUE, ORANGE = "#2a78d6", "#eb6834"; INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
d = json.load(open(a.json)); grid = d["grid"]; res = {r["k"]: r for r in d["results"]}
sp = d["spoil"]
spoiled = [100 * (sp[str(p)]["trunc"] + sp[str(p)]["unparsed"]) / sp[str(p)]["n"] for p in grid]
trunc = [100 * sp[str(p)]["trunc"] / sp[str(p)]["n"] for p in grid]
unp = [100 * sp[str(p)]["unparsed"] / sp[str(p)]["n"] for p in grid]
n_q = round(sp[str(grid[0])]["n"] / 8)

plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
fig, ax = plt.subplots(1, 2, figsize=(10, 4.6), dpi=150); fig.patch.set_facecolor("#fcfcfb")
for x in ax: x.set_facecolor("#fcfcfb"); x.grid(True, color=GRID, lw=0.8); x.set_axisbelow(True); [s.set_visible(False) for s in (x.spines["top"], x.spines["right"])]

# A -- the two ballot-rule curves, nothing else
m1, s1, m8 = np.array(res[1]["means"]), np.array(res[1]["sem"]), np.array(res[8]["means"])
ax[0].fill_between(grid, m1 - s1, m1 + s1, color=BLUE, alpha=0.12, lw=0)
ax[0].plot(grid, m1, color=BLUE, lw=1.8, marker="o", ms=5.5)
ax[0].plot(grid, m8, color=ORANGE, lw=1.8, marker="o", ms=5.5)
for y, t in [(m1[-1], "maj@1 (±1 SEM)"), (m8[-1], "maj@8")]:
    ax[0].annotate(t, (1.0, y), xytext=(6, 0), textcoords="offset points", color=INK, fontsize=8.5, va="center")
i = int(np.argmax(m1))
ax[0].annotate(f"argmax {grid[i]}\nP(joint)={res[1]['p_joint']:.3f}", (grid[i], m1[i]), xytext=(-46, -52),
               textcoords="offset points", ha="center", color=INK2, fontsize=8,
               arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax[0].set_title(f"A  InternVL3.5-8B, T=1.2, n={n_q} questions", loc="left", color=INK, fontsize=10)
ax[0].set_xlabel("top_p"); ax[0].set_ylabel("accuracy"); ax[0].set_xlim(0.05, 1.24); ax[0].set_ylim(0.45, 0.72)

# B
ax[1].bar(grid, trunc, width=0.07, color=BLUE, edgecolor="#fcfcfb", lw=1.5)
ax[1].bar(grid, unp, width=0.07, bottom=trunc, color=ORANGE, edgecolor="#fcfcfb", lw=1.5)
for p, s in zip(grid, spoiled):
    ax[1].annotate(f"{s:.0f}%" if s >= 1 else f"{s:.1f}%", (p, s), xytext=(0, 3), textcoords="offset points",
                   ha="center", color=INK2, fontsize=7.5)
ax[1].annotate("truncated (repetition loop / cap)", (0.50, 27), color=INK, fontsize=8.5)
ax[1].plot([0.45, 0.48], [27.4, 27.4], color=BLUE, lw=6, solid_capstyle="butt")
ax[1].annotate(f"unparseable ({sum(sp[str(q)]['unparsed'] for q in grid)} of "
               f"{sum(sp[str(q)]['n'] for q in grid):,})", (0.50, 25), color=INK, fontsize=8.5)
ax[1].plot([0.45, 0.48], [25.4, 25.4], color=ORANGE, lw=6, solid_capstyle="butt")
ax[1].set_title("B  Spoiled ballots by top_p", loc="left", color=INK, fontsize=10)
ax[1].set_xlabel("top_p"); ax[1].set_ylabel("% of 8 ballots per cell"); ax[1].set_ylim(0, 30)

fig.suptitle("maj@1 has an interior optimum at top_p=0.8; maj@8 is flat — the gap between them is spoiled ballots",
             x=0.01, ha="left", color=INK, fontsize=10.5, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(a.out, facecolor=fig.get_facecolor()); print("->", a.out)
