# Does `top_p` have an optimum? A re-run, and the answer

**Question.** As you raise `top_p`, a vision-language model's reasoning gets more varied.
Is there a middle setting that is best — an inverted-U in accuracy?

**Answer.** Not at temperature 1.0. Yes at temperature 1.6, with the peak near `top_p`≈0.3.
The previously reported inverted-U at T=1.0 was an artifact of discarding truncated
generations, and does not survive counting them.

`Qwen/Qwen3.5-9B` on MMMU-Pro `standard (10 options)`, 86 questions (5%, seed 20260706),
16 samples per (question, `top_p`) cell, official MMMU-Pro CoT prompt, 2x A100-40GB.

---

## 1. The headline

**Per-sample accuracy (k=1), 86 questions, spoiled-ballot counting:**

| T | p=0.1 | p=0.2 | p=0.3 | p=0.4 | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=0.95 | p=0.98 | p=0.99 | p=1.0 | shape |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1.0** | — | — | — | — | 0.7536 | 0.7674 | 0.7645 | 0.7725 | 0.7660 | 0.7725 | 0.7791 | **0.7958** | 0.7892 | rising to the edge |
| **1.6** | 0.8281 | 0.8594 | **0.8620** | 0.8490 | 0.8568 | — | 0.7344 | — | 0.2578 | 0.1823 | — | — | 0.1380 | **interior peak** |

*(T=1.6 row is an interim 24-question subset; see §5.)*

**At T=1.0 there is no interior optimum.** Over a 9-point grid the argmax sits at the 0.99/1.0
edge at every k ∈ {1,2,4,8,16}, the quadratic is *up*-opening at k=1..8, and the
pre-registered shape criterion reaches at most P=0.34 against a required 0.95. The omnibus
repeated-measures ANOVA rejects only at k=1 (F=2.28, p=0.021), and it rejects in favour of a
*monotone rise*, not a peak.

**At T=1.6 the curve inverts, hard.** Accuracy collapses from 0.86 at `top_p`=0.3 to 0.14 at
1.0 — near chance for 10 options. The quadratic is decisively down-opening (−1.78 to −2.12),
the omnibus gives F=35–81 at p≈0, and at k=4 the joint shape criterion reaches **0.964,
clearing the pre-registered 0.95**.

![result](outputs/corrected_topp_result.png)

## 2. Why the original result was wrong

The prior write-up reported an interior peak at `top_p`=0.5 in per-sample accuracy. It came
entirely from dropping generations with `finish_reason != "stop"` before analysis, because
truncation is *not* uniform across the swept axis:

| T=1.0, truncated | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=0.95 | p=0.98 | p=0.99 | p=1.0 |
|---|---|---|---|---|---|---|---|---|---|
| | 9.23% | 6.18% | 4.65% | 3.42% | 1.38% | 0.65% | 0.51% | 0.29% | **0.07%** |

A 130x gradient along the axis being measured. Deleting those traces inflates exactly the
`top_p` values that truncate most:

| k=1 | p=0.5 | p=0.7 | p=0.9 | p=0.95 | p=1.0 | argmax | P(shape) |
|---|---|---|---|---|---|---|---|
| exclude truncated (original rule) | **0.8013** | 0.7877 | 0.7761 | 0.7776 | 0.7896 | **0.5** | 0.015 |
| spoiled ballot (this work) | 0.7536 | 0.7645 | 0.7660 | 0.7725 | **0.7892** | **1.0** | 0.190 |

The filtering choice alone moves the peak from the 1.0 edge to 0.5 and is worth 4.8
percentage points at `top_p`=0.5. Even under the original rule the peak was never
significant (Holm-adjusted p=0.22).

**Counting rule used here (spoiled ballot).** Every cell has exactly 16 *ballots*, one per
generated sample. A ballot is valid if the trace terminated and an answer parsed; otherwise
it is *spoiled* — it consumes budget and does not vote. A generation that never produced an
answer is a failure that cost inference, not an event that did not happen. Nothing is ever
silently dropped, so every cell has the same denominator, and balancing no longer discards
39 of 86 questions. `sentinel` (spoiled ballots vote for a losing sentinel) and `exclude`
(the original rule) are reported alongside so the delta between them stays visible.

## 3. Why T=1.0 *cannot* show an inverted-U

This is mechanical, not empirical. At T≤1, lowering `top_p` truncates the tail and
renormalises — it only ever *sharpens* the distribution. The axis therefore runs from
near-greedy at `top_p`→0 to the model's true distribution at `top_p`=1.0, and stops. An
inverted-U needs a regime where sampling is *worse than the model*, which requires inflating
the tail. There is no such regime at T≤1, so theory predicts rise-then-plateau — which is
what we measure.

The data confirms no over-diversification is reached at T=1.0: the self-consistency gain
(maj@16 − per-sample) is ~6 pp at every `top_p` and does not collapse at the top.

At T=1.6 the tail *is* inflated, `top_p` truncation does real work, and the falling arm
appears. The failure mode also changes character, which is the mechanism made visible:

| T=1.6 spoiled ballots | p=0.1 | p=0.3 | p=0.5 | p=0.7 | p=0.9 | p=0.95 | p=1.0 |
|---|---|---|---|---|---|---|---|
| truncated (repetition loops) | 4.2% | 4.2% | 4.7% | 0.6% | 0% | 0% | **0%** |
| unparseable (incoherent output) | 0% | 0% | 0% | 1.6% | 6.1% | 8.6% | **13.4%** |

At T=1.0 the model fails at *low* `top_p` by looping. At T=1.6 it fails at *high* `top_p` by
producing text that never yields an answer. The optimum moves left as temperature rises —
the tradeoff lives on the `top_p`×T plane, not on any single `top_p` sweep.

## 4. Method

Pre-registered before looking at any curve, and applied identically to every arm:

- **Omnibus first.** Repeated-measures ANOVA over the `top_p` levels. If it does not reject,
  that is stated prominently — pairwise tests after a null omnibus are not evidence.
- **Shape test.** Two-lines: split at the argmax, fit OLS to each segment, require left
  slope > 0 and right slope < 0. Significance by bootstrap over *questions* (B=20,000), the
  same resample applied to every `top_p` column so pairing is preserved. α=0.05, so the claim
  stands only at P(shape) ≥ 0.95.
- **Joint criterion.** Two-lines alone is too easy to pass: when the argmax lands on the
  second-to-last grid point the "right slope < 0" arm is fitted to two points and reduces to
  the sign of one noisy difference. The reported criterion requires the *same* bootstrap
  replicate to also yield a down-opening quadratic.
- **Multiplicity.** Pairwise paired t-tests against the peak are Holm-corrected; raw and
  adjusted p are both reported, never only the winning comparison.
- **Power.** Reported on the full set and on the informative subset (questions not answered
  identically by all 16 samples at every `top_p`).
- **No silent exclusions.** Every analysis prints its per-`top_p` spoil rate, so any
  differential filtering is visible in the artifact itself.

**Sampling config is recorded in the data, not in prose.** Every generated row carries the
complete config (`top_k`, `min_p`, penalties, seed, `max_tokens`, `max_model_len`, KV dtype,
model, dtype) plus a `cfg_profile` tag. `cot_gen.py --resume` reads that stamp and aborts on
mismatch rather than interleaving incomparable traces, and `--sampling-profile` is required
with no default so the choice is always deliberate. This project's central failure was a run
invalidated by two flags nobody recorded; that is now impossible to repeat silently.

## 5. Caveats — read these

- **The T=1.6 low-`top_p` arm is interim.** `top_p` ∈ {0.1,0.2,0.3,0.4} was still generating
  when this was written; the 9-point T=1.6 numbers are on **24 of 86 questions**. The 5-point
  arm (0.5–1.0) is complete at 86.
- **The interior peak is carried by the right arm.** At k=1 the left arm is not significant:
  0.3 vs 0.1 = +3.4 pp (p=0.42), 0.3 vs 0.2 = +0.3 pp (p=0.93), while 0.3 vs 1.0 = +72.4 pp
  (p=4e-13). The honest description is *plateau then cliff* rather than a symmetric hill, and
  the peak's exact location (0.2 / 0.3 / 0.5) is unresolved.
- **The left arm is not a truncation artifact** — spoil rates across 0.1–0.5 are flat
  (4.2/5.0/4.2/2.9/4.7%), with no gradient. This is the key difference from the original
  result, whose left arm *was* the truncation gradient.
- **The T=1.0 raw traces are not in this repo.** The original 11,472-trace dataset was
  deleted on request. The T=1.0 numbers in `outputs/PHASE1_FINAL.md` were computed from it
  and **cannot be regenerated from what is committed** — `cots/t10_topup_*.jsonl.gz` hold
  only the four `top_p` levels generated here (0.6, 0.8, 0.98, 0.99), not the five original
  ones. The T=1.6 results are fully reproducible.
- **One model, one benchmark, 86 questions.** Paired SE ≈ 1.2 pp at T=1.0, so effects below
  ~3 pp are not resolvable there. Nothing here is claimed to generalise across models.
- **`top_p` < 0.5 at T=1.0 was not generated** — that region is dominated by repetition
  collapse (~10% of traces) and measures a decoding pathology rather than sampling diversity.

## 6. Layout

```
cots/                                  all traces generated by this work (gzipped)
  t10_topup_grid7.jsonl.gz             T=1.0, top_p 0.6 / 0.8            1,376
  t10_topup_dense.jsonl.gz             T=1.0, top_p 0.98 / 0.99          2,752
  t16_sweep.shard{0,1}.jsonl.gz        T=1.6, full sweep                12,384*
  qwen_recommended.jsonl.gz            top_k=20 / presence_penalty=1.5   2,688
outputs/
  PHASE1_FINAL.md/.json                T=1.0, 9-point grid, all rules
  PHASE1_T16.md/.json                  T=1.6, 5-point grid, 86 questions
  PHASE1_T16_INTERIM.md/.json          T=1.6, 9-point grid, 24 questions
  corrected_topp_result.png            the figure above
cot_gen.py                             generation (config-stamped, resume-guarded)
majk.py                                ballot-model maj@k + shape tests
phase1_analysis.py                     the full pre-registered analysis
result_chart.py                        the figure
run_*.sh, chain_*.sh, watch_*.sh       the launchers actually used
env_versions.txt                       vLLM / torch / driver versions
```
*\*shard totals grow until the low-`top_p` arm finishes; both shards are strided question
slices, so concatenate them to recover the arm.*

## 7. Reproduce

```bash
source /venv/main/bin/activate

# T=1.6 sweep (GPU). --sampling-profile is required; there is no default.
python cot_gen.py --sampling-profile neutral \
  --min-p 0 --repetition-penalty 1.0 --temperature 1.6 \
  --sample-frac 0.05 --n-samples 16 --top-ps "0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0" \
  --num-shards 2 --shard-id 0 --resume \
  --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
  --max-num-batched-tokens 16384 --chunk-questions 8 \
  --out outputs/t16_gen.jsonl --questions-out outputs/t16_q.json

# analysis (CPU, ~1 min). --temperature is mandatory when arms share a directory:
# cells are keyed on (id, top_p), so two temperatures in one glob would merge.
python phase1_analysis.py --glob "cots/t16_sweep.shard*.jsonl.gz" --temperature 1.6 \
  --grid 0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0 --out outputs/PHASE1_T16_FULL.md

python result_chart.py --t10 outputs/PHASE1_FINAL.json --t16 outputs/PHASE1_T16.json
```

## 8. What to do next

1. **Finish the T=1.6 low arm** (86 questions) — settles whether the left arm rises or is
   flat, and pins the peak location. Tightens the left-arm SE ~1.9x.
2. **Fill the `top_p`×T plane**: T ∈ {0.7, 1.0, 1.3, 1.6} × the 9-point grid. The prediction
   is sharp — curvature absent at 0.7, present and increasing through 1.6. A pattern across a
   grid is evidence in a way a single interior argmax is not.
3. **The soundness arm needs redoing before it is cited.** Its premises were extracted with
   truncated traces dropped by default, so the reported slope inherits the same bias this
   work removed from the accuracy arm. Independently, the two judging modes of identical
   claim sets disagreed systematically (atomic-pass/holistic-fail 219 vs 3, McNemar
   p=5.4e-61), so the instrument needs adjudicating before the metric means anything.
