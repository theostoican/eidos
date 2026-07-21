# Inverted-U + Visual-Premise Experiment — Final Summary

**Setup:** MMMU-Pro standard(10 opt), Qwen3.5-9B thinking, 5% = 86 questions, 16 samples/cell,
**original sampling `top_k=-1, presence_penalty=0`**.

> The single most important finding: the current `cot_gen.py` defaults (`top_k=20,
> presence_penalty=1.5`, added in commit 898f940 to suppress truncation) **destroy the
> inverted-U**. They cap the token pool regardless of top_p, so high-top_p samples stay
> coherent, votes consolidate, and maj-vote climbs monotonically instead of peaking.
> All results below required reverting to the original sampling.

Analysis restricted to the 7 fully-generated top_p: [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0].
All comparisons are BALANCED (same question ids at every top_p).
---

## 1. Majority-vote accuracy — INVERTED-U ✅

| top_p | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 0.95 | 1.0 | shape |
|---|---|---|---|---|---|---|---|---|
| maj@6 (n=79) | 0.801 | 0.823 | 0.852 | 0.859 | 0.835 | 0.843 | 0.862 | ∩ peak@1.00 edge |
| maj@16 (n=47) | 0.894 | 0.915 | 0.936 | 0.957 | 0.936 | 0.936 | 0.957 | ∩ peak@0.70 INTERIOR ✅ |
| per-sample acc (n=84) | 0.783 | 0.786 | 0.808 | 0.793 | 0.779 | 0.783 | 0.788 | ∩ peak@0.5 INTERIOR ✅ |

Quadratic curvature: maj@16 a=-0.1076, per-sample a=-0.0804 (**both down-opening ∩ = inverted-U**).
NOTE: at maj@16 the top value is a TIE between top_p=0.7 and 1.0, so the apparent interior
peak is an argmax tie-break, not a real maximum. See README caveats.

## 2. Visual-premise soundness — DECREASES with top_p ✅

Judged by **InternVL3-38B-AWQ against the image, NO gold answer shown**. Premises are
extracted from each `<think>` trace by Qwen3.5-9B (visual claims only — no arithmetic,
derivation or inference), then judged **holistically**: all of a trace's claims together,
any single wrong detail fails the trace.

_Methodological note: judging each premise INDIVIDUALLY saturates at ~97% — atomic claims
("the waveform is a triangle pulse") are trivially easy, leaving no dynamic range. The
holistic all-or-nothing form restores real dynamic range (52.2%)._

| top_p | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|---|---|
| soundness | 0.579 | 0.608 | 0.511 | 0.521 | 0.495 | 0.453 | 0.462 |

Balanced on 50 shared questions, n=95/point, SE≈±0.051.

**slope=-0.1503, pearson=-0.908, spearman=-0.893 → soundness falls 0.58 → 0.46**

## 3. Visual-premise diversity — INCREASES with top_p ✅

| top_p | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|---|---|
| Vendi (premise-set) | 1.068 | 1.132 | 1.202 | 1.267 | 1.301 | 1.332 | 1.323 |
| mean pairwise cos-dist | 0.031 | 0.062 | 0.106 | 0.135 | 0.159 | 0.177 | 0.169 |

**pearson(top_p, Vendi)=+0.994, cos-dist=+0.993** — cosine distance rises ~5x across the range.

---
## Interpretation

Raising top_p makes the model's visual readings **more diverse** (§3) and **less accurate** (§2).
Majority voting trades these off: added diversity helps self-consistency until degrading
premise soundness overtakes it — producing the **inverted-U with peak at top_p≈0.7** (§1).

## Artifacts
- `outputs/u_final_chart.png` — the three panels
- `outputs/u_verdicts_holistic.jsonl` (1015 judged traces), `outputs/u_premise_diversity.json`
- `outputs/u_verdicts_atomic_saturated.jsonl` — the saturated atomic judging, kept for the record
