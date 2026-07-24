#!/usr/bin/env python
"""Single-panel chart: visual-premise soundness, visual-premise diversity, per-sample accuracy.

The three headline measures on ONE axis. They live on different scales (soundness 0.45-0.61,
Vendi 1.07-1.33, accuracy 0.78-0.81) but all read against the same [0,1.4] axis: Vendi is a
diversity index that naturally sits above 1.0, the other two are fractions below it. No dual
axis (the classic charting error) -- the plot is read as absolute value, and every point is
directly labelled with its raw number. Distinct marker shapes carry identity alongside colour.

Restricted to the 7 fully-generated top_p; soundness is balanced across the shared question set.
Reads the committed gzipped data directly (no uncompressed intermediates required).
"""
import collections, gzip, json, glob, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPLETE=[0.1,0.3,0.5,0.7,0.9,0.95,1.0]
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; GRID="#d9d8d4"
C_SOUND="#2a78d6"; C_DIV="#eb6834"; C_ACC="#1baf7a"   # validated categorical slots 1,2,3

LETTERS=[chr(ord("A")+i) for i in range(26)]
ANS_RE=re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)
def parse_answer(text, n_options):
    valid=set(LETTERS[:n_options])
    after=text.split("</think>")[-1] if "</think>" in text else text
    for c in reversed(ANS_RE.findall(after) or ANS_RE.findall(text)):
        if c.upper() in valid: return c.upper()
    for ch in reversed(re.findall(r"\b([A-J])\b", after)):
        if ch in valid: return ch
    return None

# --- soundness (holistic, balanced on shared question ids) ---
V=[json.loads(l) for l in gzip.open("outputs/u_verdicts_holistic.jsonl.gz","rt")]
V=[r for r in V if r["correct"] is not None and r["top_p"] in COMPLETE]
tp=collections.defaultdict(set)
for r in V: tp[r["top_p"]].add(r["id"])
sh=set.intersection(*[tp[p] for p in COMPLETE])
agg=collections.defaultdict(lambda:[0,0])
for r in V:
    if r["id"] in sh: agg[r["top_p"]][0]+=int(r["correct"]); agg[r["top_p"]][1]+=1
sound=[agg[p][0]/agg[p][1] for p in COMPLETE]
print(f"[soundness] balanced on {len(sh)} shared questions, n={agg[COMPLETE[0]][1]}/point")

# --- diversity (Vendi over premise-set embeddings) ---
D={e["top_p"]:e["vendi_set"] for e in json.load(open("outputs/u_premise_diversity.json"))["evolution"]}
div=[D[p] for p in COMPLETE]

# --- per-sample answer accuracy (balanced; truncated traces excluded, matching the report) ---
cell=collections.defaultdict(list)
for f in glob.glob("cots/u_gen.shard*.jsonl.gz"):
    for r in (json.loads(l) for l in gzip.open(f,"rt")):
        p=float(r["top_p"])
        if p not in COMPLETE or r.get("finish_reason")!="stop": continue
        pred=parse_answer(r["text"], int(r["n_options"]))
        if pred is None: continue
        cell[(r["id"],p)].append(int(pred==r["gold"]))
qfrac=collections.defaultdict(dict)
for (q,p),v in cell.items(): qfrac[q][p]=float(np.mean(v))
qs=[q for q,d in qfrac.items() if all(p in d for p in COMPLETE)]
acc=[float(np.mean([qfrac[q][p] for q in qs])) for p in COMPLETE]
print(f"[accuracy] balanced on {len(qs)} shared questions")

# --- plot ---
fig,ax=plt.subplots(figsize=(12.0,7.0))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.axhline(1.0, lw=1, color=GRID, zorder=1, alpha=0.6)
series=[("Visual-premise soundness (fraction sound)", sound, C_SOUND, "o", -14),
        ("Visual-premise diversity (Vendi index, not a fraction)", div, C_DIV, "s", +11),
        ("Per-sample answer accuracy (fraction correct)", acc, C_ACC, "^", -14)]
for name,y,c,mk,dy in series:
    ax.plot(COMPLETE,y,lw=2,marker=mk,ms=8,color=c,label=name,zorder=3,
            markeredgecolor=SURF,markeredgewidth=2)
    for px,py in zip(COMPLETE,y):
        dx=-7 if px==0.9 else (7 if px==0.95 else 0)
        ax.annotate(f"{py:.3f}", xy=(px,py), xytext=(dx,dy), textcoords="offset points",
                    ha="center", va="center", fontsize=7.5, color=INK2, zorder=5)
ax.set_xlabel("top_p", fontsize=11, color=INK)
ax.set_ylabel("value", fontsize=11, color=INK)
ax.set_ylim(0, 1.40); ax.set_xlim(0.03, 1.10)
ax.set_title("Raising top_p: visual premises get MORE DIVERSE and LESS SOUND; per-sample accuracy stays ~flat\n"
             "MMMU-Pro / Qwen3.5-9B thinking · 86 questions × 16 samples · original sampling (top_k=-1, presence_penalty=0)",
             fontsize=12, color=INK, pad=14, loc="left")
ax.grid(axis="y", color=GRID, lw=0.8, alpha=0.7, zorder=0); ax.set_axisbelow(True)
for s in ("top","right"): ax.spines[s].set_visible(False)
for s in ("left","bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=10)
ax.legend(fontsize=10, frameon=False, loc="lower left", labelcolor=INK)
fig.tight_layout()
fig.savefig("outputs/u_combined_chart.png", dpi=160, facecolor=SURF, bbox_inches="tight")
print("[combined_chart] -> outputs/u_combined_chart.png")
for name,y,_,_,_ in series:
    print(f"  {name:52} {y[0]:.3f} -> {y[-1]:.3f}")
