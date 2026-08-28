# How to run the visual-premise arm

Measures two things about the T=1.0 traces already committed in `cots/`, and plots them
against the accuracy curve `analyze.py` produces:

- **soundness** — of the visual claims a trace makes, what fraction survive checking against
  the actual image
- **diversity** — how varied those claims are, across the samples of one question

Nothing here regenerates traces. Inputs are `cots/t10_*.jsonl.gz` (committed) plus two
downloaded models. One 24GB GPU is enough; the whole arm is ~15 h for 8 of the 16 samples.

## 0. Environment

```bash
source /venv/main/bin/activate
uv pip install vllm datasets sentence-transformers matplotlib scipy
uv pip uninstall torchaudio        # vLLM imports it; the preinstalled build is cu12.8 vs torch cu13.0
hf download Qwen/Qwen3.5-9B                 # extractor, ~18GB
hf download OpenGVLab/InternVL3-8B-AWQ      # judge, ~7GB
hf download BAAI/bge-large-en-v1.5          # diversity encoder, ~1.3GB
```

## 1. Prep: traces → balanced sample layers

```bash
python vp_prep.py            # ~75 s; writes outputs/vp_layers/layer{00..15}.jsonl.gz + vp_q.json
```

Splits the 55,200 T=1.0 traces into one file per `sample_idx`. Each layer is a complete
345 × 10 grid, so stopping after any layer leaves a curve computed on the same cells at every
`top_p`. Applies the **ballot rule** here: a truncated trace gets `pred=None` and is scored
wrong, never dropped — the parser's bare-letter fallback would otherwise read a stray letter
out of a repetition loop and lift accuracy by up to 2.6 pp, concentrated at low `top_p`.

## 2 & 3. Extract premises, then judge them against the image

```bash
./run_vp_watchdog.sh                     # 8 samples per cell (~15 h)
VP_LAST_LAYER=15 ./run_vp_watchdog.sh    # all 16 (~30 h)
```

The watchdog loops `run_vp.sh`, which per layer runs `vp_extract.py` → `vp_judge.py` →
`vp_result.py` and restarts on any death (both stages resume from their own output, so a
restart costs only an engine reload). Failures land in `outputs/VP_FAILURES.txt`; an empty
file means a clean run. Watch it with:

```bash
tail -f logs/vp_run.log | grep -E "^=== layer|^\[extract\] [0-9]+/|^\[judge\] [0-9]+/"
```

To run the stages by hand instead:

```bash
python vp_extract.py --layers "outputs/vp_layers/layer00.jsonl.gz"   # Qwen3.5-9B, temp 0
python vp_judge.py                                                    # InternVL3-8B-AWQ, temp 0
```

Measured throughput on one RTX 3090: extraction **2,056 traces/h**, judging **16,495
traces/h**. Judging is cheap because the prompt puts rubric + question + images *before* the
premise list, so prefix caching encodes each question's images once rather than once per trace.

## 4. Soundness + accuracy

```bash
python vp_result.py     # -> outputs/RESULT_VP.md/.json + premise_soundness_vs_topp.png
```

Aborts if its recomputed maj@1 disagrees with `outputs/RESULT_T10.json` when all 16 samples
are present, so this arm cannot silently plot a different accuracy definition than the repo's.

## 5. Diversity

```bash
python vp_diversity.py  # -> outputs/RESULT_VP_DIVERSITY.md/.json + vp_diversity_cells.json
```

**Count-matching is the whole point.** Premise count itself rises ~15% along the axis and
Vendi is an effective *count*, so unmatched it re-measures the confound: +15.4% with curvature
`a=+0.30` unmatched, versus +9.0% with `a=-0.44` matched. Every cell is subsampled to exactly
K items, 24 times, averaged. A model-free numeric measure runs alongside, because no encoder
tested scores a misread number at more than 0.21 of a different claim.

## 6. The three-line chart

```bash
python vp_one_chart.py  # -> outputs/topp_correctness_and_diversity.png
```

Soundness, accuracy and diversity share one axis by **indexing each series to its own value
at `top_p`=0.5**, so what is compared is the shape and size of change rather than levels that
were never commensurable. Absolute values are annotated on every point, and the legend carries
each series' 0.5 → 1.0 endpoints. Rescaling Vendi as `(VS-1)/(K-1)` was tried and rejected: at
K=20 it parks diversity at 0.13, where a +10% change looks like a flat line.

## Optional: the controls that make the numbers interpretable

```bash
python vp_embed_probe.py    # five encoders vs misread-number sensitivity -> why bge-large
python vp_judge_shuffle.py  # re-judge with premises reordered -> judge test-retest (kappa)
```

The second is worth running before trusting any single verdict: re-judging the same premises
in a different order flips **43% of UNSOUND** calls (κ=0.36). Aggregate means over ~10⁵
premises survive that; individual trace verdicts do not.

## What came out

| quantity | `top_p` 0.5 → 1.0 | RM-ANOVA |
|---|---|---|
| visual-premise soundness | 0.941 → 0.929 (−1.2%) | F=1.63, p=0.10 |
| premise diversity (count-matched Vendi) | 3.382 → 3.730 (+10.3%) | F=42.5, p=8e-71 |
| answer accuracy (maj@1) | 0.677 → 0.713 (+5.3%) | F=6.43, p=4e-09 |

Diversity rises, accuracy rises, perception does not. Across the grid soundness and accuracy
anticorrelate (r=−0.51); within a question r=−0.03 over 158 questions. Given κ=0.36, the flat
soundness result is properly stated as *no trend detectable with a judge this noisy* — label
noise attenuates real effects toward the null.

Two caveats worth carrying: the extractor sees only the first 6,000 tokens of each trace
(40.3% are clipped, and the clip rate drifts 36.8% → 44.1% along the axis), and a per-trace
*average* soundness cannot see the single decisive premise — some questions do flip on one
specific visual read while winners and losers both score near 100% sound.
