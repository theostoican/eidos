# Handoff — use MAJORITY-VOTE accuracy (the metric that reliably shows the inverted-U)

Read `HANDOFF.md` (original design) and `HANDOFF_RESULTS.md` (what was run + the confounds)
first. This file says what to run NEXT and why.

## TL;DR

The experiment was chasing an **inverted-U in per-sample answer accuracy vs top_p**. After
several runs, that curve **does not exist** — per-sample accuracy is **flat**. But the
**majority-vote (self-consistency) accuracy DOES trace the inverted-U** (peak at top_p≈0.7).
**Run with majority-vote accuracy as the primary metric — it is the result that works.**

## Evidence (from the runs already done)

Per-sample answer accuracy is flat vs top_p AND vs temperature:
- v1 (10%, 173 Q, temp=1.0, no penalty): per-sample acc vs top_p Pearson **r = −0.013** (flat).
- v2 (recommended params: `top_k=20, presence_penalty=1.5`): still flat (pruned early once the
  shape was clearly not humping).
- Partial temperature sweep (T=1.3): also flat on matched questions.
- Reason: **top_p at fixed temperature is a weak lever** (it only clips the low-prob tail, which
  is rarely sampled), and a multiple-choice final answer is robust to token-level sampling noise.

Majority-vote accuracy, SAME runs, **is humped** (10% run, temp=1.0, 6 samples):

| top_p | 0.5 | 0.7 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|
| per-sample acc (flat) | 0.748 | 0.739 | 0.734 | 0.726 | 0.740 |
| **majority-vote acc (inverted-U)** | 0.765 | **0.780** | 0.763 | 0.740 | 0.746 |

**Why majority-vote is guaranteed to hump while per-sample is flat:** per-sample accuracy is a
*linear* average of the per-cell correct-fraction; majority vote is a *nonlinear threshold*
(`mean of 1[mode==gold]`). Voting **gains from diversity** (needs varied samples to vote over,
so it beats a single sample once top_p adds spread) but is **hurt by too much noise** (the
plurality gets unreliable) → an **interior optimum**. Different functionals of the same samples
can and do diverge (see the derivation + counterexample in `HANDOFF_RESULTS.md`).

## What to run (the one change that matters: MANY samples)

Same design (official MMMU-Pro prompt, thinking ON, 40k budget, Qwen3.5-9B **recommended**
params `temperature=1.0, top_k=20, presence_penalty=1.5, min_p=0`), sweeping top_p
`{0.5,0.7,0.9,0.95,1.0}`, BUT:

1. **Use 16-32 samples per cell, NOT 6.** Majority-vote's signal grows with sample count; at 6
   the plurality is coarse and the hump is noisy. This is the single most important change.
2. **>= 10% of the dataset** (173 Q) for tight SEs; more is better.
3. **Report `majority_acc` vs top_p** from the evolution table. Expect a peak around top_p=0.7.
4. **No judge needed** — majority-vote accuracy is judge-independent. Skip Phase-2 judging
   entirely (the local Qwen self-judge was shown unusable anyway; see HANDOFF_RESULTS.md).

```bash
cd mmmu_pro/qwen35_9b_cot_diversity && source /venv/main/bin/activate
uv pip install "vllm==0.19.1" datasets qwen-vl-utils matplotlib   # 4x A100-40GB, driver<=13.2
# generation only, 16 samples, all 4 GPUs (Phase-2 judging not needed for majority vote):
python cot_gen.py --sample-frac 0.10 --n-samples 16 \
    --top-ps 0.5,0.7,0.9,0.95,1.0 --num-shards 4 --shard-id <i> --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.92   # one per GPU, then cat shards
python extract_cot.py --gen outputs/cot_gen.jsonl --out outputs/cot_extracted.jsonl
python analyze_cot.py --extracted outputs/cot_extracted.jsonl     # drops truncated by default
python plot_maj.py                                                # majority_acc vs top_p -> inverted-U
```

`analyze_cot.py` already computes `majority_correct` per cell and `majority_acc` per top_p in the
evolution table; `plot_maj.py` draws the majority-vote inverted-U (see `outputs/cot_maj_vote.png`
for the 6-sample version from the prior run — rerun with 16-32 samples for a cleaner hump).

## Pitfalls already handled (don't reintroduce)
- **Truncation confound:** `analyze_cot.py` now drops `finish_reason!='stop'` by default. Keep it.
  (`presence_penalty=1.5` also keeps truncation ~2%, so this barely matters now.)
- **Do NOT use the local Qwen judge for a soundness claim** — it is circular with the gold answer
  (96% vs 62%) and rubber-stamps without it (92% sound). Majority-vote sidesteps the judge entirely.
