# Does `top_p` have an optimum? A re-run, and the answer

**Question.** As you raise `top_p`, a vision-language model's reasoning gets more varied.
Is there a middle setting that is best — an inverted-U in accuracy?

**Answer.** **Yes at temperature 1.6**, where majority-vote accuracy has a genuine interior
optimum near `top_p`≈0.3–0.5. **No at temperature 1.0**, where the curve rises and then flattens.
The T=1.0 arm was re-run at four times the sample on a denser grid specifically to test a
possible peak at `top_p`=0.99; it does not survive.

`Qwen/Qwen3.5-9B` on MMMU-Pro `standard (10 options)`, official MMMU-Pro CoT prompt, 2x A100-40GB,
16 samples per (question, `top_p`) cell.

| arm | questions | grid | ballots |
|---|---|---|---|
| T=1.6 | 86 (5%, seed 20260706) | 9 points, 0.1–1.0 | 12,384 |
| T=1.0 | **345 (20%, nested superset of the 86)** | **10 points, 0.5–1.0** | **55,200** |

**Counting.** One rule, not selectable. Every (question, `top_p`) cell has exactly 16
**ballots**, one per generated sample. A ballot is valid if the trace terminated and an
answer parsed; otherwise it is **spoiled** — it consumes inference budget and does not vote.
Nothing is ever dropped, so every cell has the same denominator and no result can be
manufactured by discarding failures on an axis that fails unevenly.

---

## 1. The headline

**T=1.6, 86 questions, 9-point grid — an interior optimum:**

| k | p=0.1 | p=0.2 | p=0.3 | p=0.4 | p=0.5 | p=0.7 | p=0.9 | p=0.95 | p=1.0 | argmax | F | P(joint) | quad a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7355 | 0.7478 | 0.7464 | 0.7420 | **0.7616** | 0.6170 | 0.1926 | 0.1490 | 0.1221 | 0.5 | 163.8 | 0.916 | −1.51 |
| 16 | 0.7907 | 0.8023 | **0.8605** | 0.8140 | 0.8372 | 0.7326 | 0.2791 | 0.1977 | 0.1860 | 0.3 | 72.8 | **0.998** ✅ | −1.71 |

The quadratic is decisively down-opening. The pre-registered joint criterion (P ≥ 0.95) is
**met at k=16**.

**T=1.0, 345 questions, 10-point grid — no interior optimum:**

| k | p=0.5 | p=0.6 | p=0.7 | p=0.8 | p=0.9 | p=0.925 | p=0.95 | p=0.975 | p=0.99 | p=1.0 | argmax | F | P(joint) | quad a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.6991 | 0.7014 | 0.7168 | 0.7145 | 0.7266 | 0.7293 | 0.7293 | 0.7237 | **0.7366** | 0.7299 | 0.99 | 9.44 | 0.532 | −0.0071 |
| 16 | 0.7623 | 0.7565 | **0.7768** | 0.7420 | 0.7710 | 0.7652 | 0.7681 | 0.7507 | 0.7594 | 0.7710 | 0.7 | 1.65 | 0.375 | +0.0293 |

The argmax at k=1 *is* interior, at 0.99. It is not a peak: the joint criterion reaches 0.532
against a required 0.95, the quadratic is indistinguishable from zero, and k=16 puts its argmax
at 0.7 with a quadratic of the wrong sign.

![result](outputs/corrected_topp_result.png)

*A: the T=1.0 curve — a rise to 0.99 then flat. B: T=1.0 against T=1.6 at k=1; the falling arm
exists only at T=1.6. C: the T=1.6 interior optimum (note the y-range — the left arm is a few pp
and the right arm is a 64 pp cliff, so the "inverted-U" is asymmetric, not a symmetric hump).
D: the T=1.0 detail over 0.9–1.0 with ±1 SEM. On a zoomed y-range a sub-pp wobble looks like a
peak, which is exactly why the error bars are drawn: they overlap at every point.*

## 2. The 0.99 peak replicates in size and fails in significance

The previous T=1.0 run (n=86) put its k=1 argmax at 0.99, 0.65 pp above `top_p`=1.0. This re-run
was designed to test that specific contrast at four times the sample:

| | Δ (0.99 − 1.0), k=1 | SE | t | p |
|---|---|---|---|---|
| previous, n=86 | +0.65 pp | 0.95 pp | 0.69 | 0.495 |
| **this run, n=345** | **+0.67 pp** | **0.45 pp** | **1.49** | **0.137** |

The effect size replicated almost exactly and the standard error halved. It is still not
significant — and **that outcome was fixed by the design before any token was generated.** A
power calculation from the earlier run's paired SE predicted t≈1.38 and ~27% power at this n;
observed t=1.49. Detecting a 0.65 pp effect at 80% power needs ~1,434 questions, 83% of the
benchmark. Quadrupling the sample was never going to settle it, and it did not.

At k=16 the contrast runs the other way, Δ = **−1.16 pp** (p=0.286), favouring the edge. That is
the arm where a coverage-vs-precision optimum is theoretically expected and where the T=1.6
effect is strongest — here it is the arm with the least sign of one.

The honest statement is **not** "there is no peak" and **not** "the peak is real". It is:
**the argmax sits reproducibly at 0.99, and 0.99 ≈ 1.0 within noise.**

## 3. What is established instead: a monotone rise

The omnibus at k=1 is now decisive — F=9.44, p≈0, against F=2.28 in the earlier n=86 run. `top_p`
genuinely does affect accuracy at T=1.0:

| | `top_p` 0.5 | `top_p` 0.99 | Δ |
|---|---|---|---|
| k=1 | 0.6991 | 0.7366 | **+3.75 pp** |

This is real structure the smaller run could not establish. But its shape is **rise-then-plateau
with a sub-pp wobble at the top**, not an inverted-U — which is precisely why the omnibus passes
while every shape test fails. A significant omnibus is evidence that the axis matters, not
evidence of the shape someone hoped to find.

## 4. Why T=1.0 cannot show an inverted-U, and T=1.6 can

Mechanical, not empirical. At T≤1, lowering `top_p` truncates the tail and renormalises — it
only ever *sharpens*. The axis runs from near-greedy at `top_p`→0 to the model's true
distribution at 1.0, and stops. An inverted-U needs a regime where sampling is *worse than the
model*, which requires inflating the tail. No such regime exists at T≤1, so theory predicts
rise-then-plateau — which is what T=1.0 measures, now at 345 questions. At T=1.6 the tail is
inflated, `top_p` truncation does real work, and the falling arm appears.

The failure modes separate cleanly, which is the mechanism made visible:

| T=1.6 | p=0.1 | p=0.2 | p=0.3 | p=0.4 | p=0.5 | p=0.7 | p=0.9 | p=0.95 | p=1.0 |
|---|---|---|---|---|---|---|---|---|---|
| truncated (repetition loops) | 9.01% | 10.25% | 9.16% | 7.12% | 4.72% | 0.58% | 0% | 0% | **0%** |
| unparseable (incoherent) | 0% | 0% | 0% | 0% | 0% | 1.60% | 6.10% | 8.58% | **13.44%** |

Low `top_p` fails by looping; high `top_p` fails by emitting text that never yields an answer.
At T=1.0 the second failure mode never appears at all — **zero** unparseable traces at every one
of the ten levels, across all 55,200 ballots. Only the looping arm exists, and it decays
monotonically. That asymmetry *is* the explanation for the missing falling arm.

## 5. Why the counting rule matters

Truncation is not uniform along the swept axis. At T=1.0 it runs 11.09% at `top_p`=0.5 to 0.42%
at 1.0 — a 26x gradient (full spoil table in `outputs/RESULT_T10.md`). Every analysis prints its
per-`top_p` spoil rate. Any rule that drops failed generations therefore inflates exactly the
cells that fail most, and compares different subsets of samples across the very axis under test.
In the earlier run that was worth 4.8 pp at `top_p`=0.5 — on its own enough to move the argmax
off the edge and produce an interior peak that is not there. Those drop-truncated numbers were
removed with the rule and are in git history at `7cb13f0` as `exclude_full`.

A generation that never produced an answer is a failure that consumed inference budget, not
an event that did not happen. It stays in the denominator.

## 6. Method

Pre-registered before any curve was inspected, applied identically to every arm:

- **Omnibus first.** Repeated-measures ANOVA over the `top_p` levels. A null omnibus is
  reported prominently — pairwise tests after one are not evidence.
- **Shape test.** Two-lines: split at the argmax, OLS per segment, require left slope > 0 and
  right slope < 0. Bootstrapped over *questions* (B=20,000), the same resample applied to
  every `top_p` column so pairing is preserved. α=0.05 → the claim stands only at P ≥ 0.95.
- **Joint criterion.** Two-lines alone is too easy to pass: with the argmax on the
  second-to-last grid point the right arm is fitted to two points and reduces to the sign of
  one noisy difference. The reported figure requires the *same* bootstrap replicate to also
  yield a down-opening quadratic. (This caught a false "inverted-U" during development.)
- **Grid placement is part of the test.** An argmax at 0.99 can never have a thick right arm,
  because the axis ends at 1.0. The T=1.0 grid therefore places 0.925 and 0.975 between the
  published levels, so a peak anywhere at or below 0.975 is fitted on ≥3 points per segment
  rather than on the sign of one difference.
- **Multiplicity.** Pairwise tests against the peak are Holm-corrected; raw and adjusted p
  both reported, never only the winner.
- **Power stated in advance.** The T=1.0 re-run's decisive contrast was known to have ~27%
  power at n=345 before it was run. A null there is reported as *unresolved at this n*, not as
  evidence of absence.
- **Every question, no subsetting knob.** All the tests are paired, so a question answered
  identically at every `top_p` already contributes nothing to either the effect or the error
  term.
- **No silent exclusions.** Every analysis prints its per-`top_p` spoil rate.

**Config is recorded in the data, not in prose.** Every row carries the complete sampling
config plus a `cfg_profile` tag; `--resume` reads that stamp and aborts on mismatch;
`--sampling-profile` is required with no default. `analyze.py` additionally refuses to run if
any cell holds more ballots than the rest — a cell present in two source files would otherwise
blend two runs into a single curve.

## 7. Caveats

- **The T=1.0 argmax is interior and the evidence does not resolve it.** ~27% power on the
  0.99-vs-1.0 contrast. This run bounds the effect (≈0.67 pp, replicated) rather than refuting
  it. Settling it needs ~1,434 questions.
- **A pre-registered secondary prediction FAILED, in both arms.** The optimum was declared
  non-decreasing in k. At T=1.6 it falls, 0.5 → 0.3. At T=1.0 it moves 0.99 → 0.7. The k=16
  argmax is unstable in general and should not be read as a location estimate.
- **n=345, not 346.** `test_Geography_252` references 35 images against vLLM's per-prompt limit
  of 8 and can never be generated; `cot_gen.py` detects and skips such questions at startup.
  It is not one of the original 86, so the nested-superset property is unaffected.
- **The T=1.6 arm is still 86 questions.** Its interior optimum has not been re-tested at 20%,
  and the k-oscillation between 0.3 and 0.5 leaves the peak location open.
- **One mechanism is untested.** Truncation shrinks the vote pool at low `top_p`, and a smaller
  pool makes the majority noisier. The clean test — subsampling every cell to a common
  valid-ballot count — has not been run.
- **One model, one benchmark.** Nothing here is claimed to generalise.
- **`top_p` < 0.5 at T=1.0 was not generated** — dominated by repetition collapse.

## 8. Layout

Three scripts, and the data. Nothing here is scaffolding.

```
cots/                              all committed traces (gzipped)
  t16_sweep.shard{0,1}.jsonl.gz    T=1.6, 9-point sweep, 86 q      12,384
  t10_20pct.shard{0a,0b,0c,        T=1.0, 10-point sweep, 345 q    52,448
             1a,1b,1c,3}.jsonl.gz    (split at ~40MB per file)
  t10_topup_dense.jsonl.gz         T=1.0, top_p 0.98 / 0.99         2,752
  t10_topup_grid7.jsonl.gz         T=1.0, top_p 0.6 / 0.8           1,376
  qwen_recommended.jsonl.gz        top_k=20 / presence_penalty=1.5  2,688
outputs/
  RESULT_T16.md/.json              T=1.6, 9-point, 86 questions
  RESULT_T10.md/.json              T=1.0, 10-point, 345 questions  <- the re-run
  RESULT_T10_LOCAL.md/.json        T=1.0 restricted to 0.9-1.0     <- the peak test
  corrected_topp_result.png        the figure
cot_gen.py                         generation (config-stamped, resume-guarded)
analyze.py                         ballot-model maj@k + the pre-registered tests
result_chart.py                    the figure
env_versions.txt                   vLLM / torch / driver versions
```

The T=1.0 arm reuses 172 cells from `t10_topup_*`, so those files are part of the dataset, not
history. `qwen_recommended.jsonl.gz` carries a **different sampling config** and must never be
globbed together with the rest — `cot_gen.py`'s resume guard and `analyze.py`'s config stamp
both refuse it, but the analysis glob is `cots/t10_*` for this reason.

## 9. Reproduce

Both arms are reproducible end to end from what is committed.

```bash
source /venv/main/bin/activate

# 1. GENERATE. --sampling-profile is required, no default: the flags that decide this
#    result must never come from an argparse default again. Run once per GPU, --shard-id 0/1.
#    T=1.0 arm (~71h on 2x A100-40GB). --nest-from keeps the 5% run's exact question set,
#    so its traces stay reusable and its numbers are a literal subset.
CUDA_VISIBLE_DEVICES=0 python cot_gen.py --sampling-profile neutral \
  --temperature 1.0 --min-p 0 --repetition-penalty 1.0 \
  --sample-frac 0.20 --nest-from 0.05 --n-samples 16 \
  --top-ps "0.5,0.6,0.7,0.8,0.9,0.925,0.95,0.975,0.99,1.0" \
  --num-shards 2 --shard-id 0 --resume --resume-glob "*/t10_*.jsonl*" \
  --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
  --max-num-batched-tokens 16384 --chunk-questions 8 \
  --out outputs/t10_20pct.jsonl --questions-out outputs/t10_20pct_q.json

# 2. ANALYSE (CPU, ~1 min). --temperature is mandatory: cells are keyed on (id, top_p), so
#    two temperatures in one glob would merge into 32-ballot cells and blend two experiments.
python analyze.py --glob "cots/t10_*.jsonl.gz" --temperature 1.0 \
  --grid 0.5,0.6,0.7,0.8,0.9,0.925,0.95,0.975,0.99,1.0 --out outputs/RESULT_T10.md
python analyze.py --glob "cots/t10_*.jsonl.gz" --temperature 1.0 \
  --grid 0.9,0.925,0.95,0.975,0.99,1.0 --out outputs/RESULT_T10_LOCAL.md
python analyze.py --glob "cots/t16_sweep.shard*.jsonl.gz" --temperature 1.6 \
  --grid 0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0 --out outputs/RESULT_T16.md

# 3. FIGURE
python result_chart.py --zoom
```

Step 2 on the committed traces reproduces the `outputs/RESULT_*.md` files bit-identically.

## 10. Next

1. **Take the T=1.6 arm to 20%.** It is the arm with a real effect and it is still the arm with
   86 questions. Same nested-superset trick, same grid.
2. **Equalise the vote pool** across `top_p` (subsample every cell to a common valid-ballot
   count) to close the one untested mechanism behind the k=16 T=1.6 result. CPU-only.
3. **Fill the `top_p`×T plane**: T ∈ {0.7, 1.0, 1.3, 1.6}. Prediction is sharp — curvature
   absent at 0.7, increasing through 1.6. A pattern across a grid beats a single argmax.
4. **Densify 0.2–0.6 at T=1.6** to pin the peak location, which the k-oscillation leaves open.
5. **The soundness arm needs redoing before it is cited.** Its premises were extracted with
   truncated traces dropped by default, so the reported slope inherits the bias removed here.
   Independently, the two judging modes of identical claim sets disagreed systematically
   (atomic-pass/holistic-fail 219 vs 3, McNemar p=5.4e-61).
