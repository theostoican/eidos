# Visual-premise analysis: how to run it, and what came out

Measures two things about the reasoning traces an arm already has committed in its `cots/`,
and plots them against that arm's accuracy:

- **soundness** — of the visual claims a trace makes, what fraction survive checking against
  the actual image
- **diversity** — how varied those claims are, across the samples of one question

Nothing here regenerates traces. The code is model-agnostic and lives in this directory; an
**arm** is a directory with a `cots/`, an `analyze.py` result to check against, and a
`vp_arm.sh` saying which traces to use:

| arm directory | `vp_arm.sh` selects | cells |
|---|---|---|
| `../qwen35_9b_cot_diversity` | Qwen3.5-9B, **T=1.6**, `top_p` 0.1–1.0 (9 pts) | 86 q × 9 × 16 samples |
| `../internvl35_8b_cot_diversity` | InternVL3.5-8B, **T=1.2**, `top_p` 0.1–1.0 (10 pts) | 172 q × 10 × 8 samples |

Same extractor, judge and encoder for every arm, so only the traces differ.

## 0. Environment

```bash
source /venv/main/bin/activate
uv pip install vllm datasets sentence-transformers matplotlib scipy
uv pip uninstall torchaudio        # vLLM imports it; a preinstalled cu12.8 build clashes with vLLM's torch
hf download Qwen/Qwen3.5-9B                 # extractor, ~19GB
hf download OpenGVLab/InternVL3-8B-AWQ      # judge, ~6GB
hf download BAAI/bge-large-en-v1.5          # diversity encoder
```

One 24GB GPU is enough. **Host driver older than CUDA 13?** vLLM's default wheel pulls torch
cu130, which will not run (and consumer GPUs get no forward-compat). Install the cu129 release
wheel instead — minor-version compatibility covers any 12.x driver:

```bash
uv pip install "https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu129
```

## 1. Run an arm

```bash
cd ../qwen35_9b_cot_diversity       # or ../internvl35_8b_cot_diversity
../common/run_vp_arm.sh             # prep -> extract+judge (watchdog) -> diversity -> chart
tail -f logs/vp_run.log | grep -E "^=== layer|^\[extract\] [0-9]+/|^\[judge\] [0-9]+/"
```

Measured on one RTX 4090: extraction **~4,600 traces/h**, judging **~40,000 traces/h** — Qwen
T=1.6 (12,384 traces) 3.2 h, InternVL T=1.2 (13,760 traces) 3 h. Everything resumes, so
re-running after a crash is safe; failures land in `outputs/VP_FAILURES.txt` (absent = clean).

Products, per arm, in `outputs/`:

| file | |
|---|---|
| `topp_correctness_and_diversity.png` | **the chart**: soundness, maj@k accuracy and diversity on one axis |
| `premise_soundness_vs_topp.png` | soundness and maj@1 in absolute units, ±1 SEM |
| `RESULT_VP.md/.json`, `RESULT_VP_DIVERSITY.md/.json` | the numbers behind both (not committed; the charts are) |
| `vp_layers/`, `vp_premises.jsonl`, `vp_verdicts.jsonl`, … | intermediates, gitignored |

## 2. What the stages do

`run_vp_arm.sh` sources `./vp_arm.sh`, then:

1. **`vp_prep.py`** splits the arm's traces into one file per `sample_idx`. Each layer is a
   complete question × `top_p` grid, so stopping after any layer leaves a balanced dataset.
   Applies the **ballot rule**: a truncated trace gets `pred=None` and is scored wrong, never
   dropped — the parser's bare-letter fallback would otherwise read a stray letter out of a
   repetition loop.
2. **`run_vp_watchdog.sh` → `run_vp.sh`**, per layer: **`vp_extract.py`** (Qwen3.5-9B, temp 0,
   text-only: pull out the claims about what the image shows, first 6,000 tokens of the trace)
   → **`vp_judge.py`** (InternVL3-8B-AWQ, temp 0, sees image + question + premises, never the
   gold answer or the trace's conclusion) → **`vp_result.py`** (running report).
   Judging is cheap because rubric + question + images come *before* the premise list, so
   prefix caching encodes each question's images once rather than once per trace.
3. **`vp_diversity.py`** — count-matched Vendi over `bge-large` embeddings. **Count-matching is
   the whole point:** premise count is not constant along the axis and Vendi is an effective
   *count*, so unmatched it re-measures the confound. Every cell is subsampled to exactly K
   premises, 24 times, averaged.
4. **`vp_one_chart.py`** — the three series share an axis by **indexing each to its own value
   at the leftmost `top_p`**; absolute values are annotated on every point.

`vp_result.py` recomputes maj@1 from the same traces and **aborts if it disagrees with the
arm's committed `analyze.py` result** once every sample is judged; maj@k is computed by
`analyze.majk_cell` itself. Truncated traces count as failures for accuracy but their premises
are *kept* for soundness and diversity (dropping them would subset the data along the swept
axis); the report carries a "truncated dropped" row so the size of that choice is visible.

## 3. What came out

| arm | soundness | maj@8 accuracy | premise diversity (count-matched Vendi) |
|---|---|---|---|
| **Qwen3.5-9B T=1.6**, `top_p` 0.1→1.0, 16 samples, 49 q paired | 0.964 → 0.914 (−5.1%), F=1.99, p=0.046 | 0.780 → 0.153, peak **0.844 at 0.5**, F=67.9 | 2.36 → 2.76 (+17%), K=8, 26 q, F=18.5, p=1e-20 |
| **InternVL3.5-8B T=1.2**, `top_p` 0.1→1.0, 8 samples, 97 q paired | 0.953 → 0.951 (−0.2%), F=1.49, p=0.15 | 0.670 → 0.670, flat, F=0.83, p=0.59 | 2.79 → 3.48 (+24%), K=20, 47 q, F=29.3, p=2e-39 |
| *Qwen3.5-9B T=1.0*, `top_p` 0.5→1.0, 8 of 16 samples, 303 q paired (not charted here) | 0.943 → 0.932 (−1.2%), F=1.13, p=0.34 | 0.739 → 0.746, flat, F=1.04, p=0.40 | 3.39 → 3.70 (+9%), K=20, 251 q, F=39.5 |

- **Diversity rises with `top_p` in every arm.** Soundness barely moves.
- **Where accuracy collapses (Qwen T=1.6, `top_p` ≥ 0.9), soundness does not collapse with
  it**: accuracy loses ~69 pp, soundness ~5. It is the one arm where soundness declines
  detectably, and the one where soundness and accuracy correlate positively across the grid
  (r=+0.84, p=0.004; the other arms are negative). Within a question r≈0 in all three.
- What collapses instead is *how many visual claims get made*: 104 premises per cell at 0.1,
  11 at 1.0, and the judge's UNVERIFIABLE share rises 7% → 18%. At high `top_p` the model
  mostly stops stating reads before it starts stating wrong ones.
- maj@1 tells a different story from maj@8 (Qwen T=1.0 rises +5.5%; InternVL has an interior
  optimum at 0.8) — both are in `RESULT_VP.md`, and the arm READMEs explain the gap.

## 4. Caveats

- **The judge is noisy.** Re-judging the same premises in a different order flipped 43% of
  UNSOUND calls in the original T=1.0 run (κ=0.36; the `vp_judge_shuffle.py` control that
  measured this is not in the repo). Aggregates over ~10⁴–10⁵ premises survive that;
  individual verdicts do not, and label noise attenuates real effects toward the null.
- **For the InternVL arm the judge is a near relative** (InternVL3-8B judging InternVL3.5-8B,
  same InternViT-300M lineage). Shared blind spots read as SOUND, so that arm's soundness
  *level* is an upper bound; only its shape across `top_p` is comparable with the Qwen arms.
- **Soundness is paired on questions with a judged premise at every `top_p`.** InternVL states
  no extractable visual claim in ~52% of traces (it often reasons from the question text), so
  97 of 172 questions pair; Qwen T=1.6 pairs 49 of 86 because of the collapse region.
- **K differs between arms** (8 vs 20, set in `vp_arm.sh`): at K=20 only 6 Qwen-T=1.6 questions
  survive at every `top_p`. Vendi *levels* are therefore not comparable across charts; shapes are.
- The extractor sees only the first 6,000 tokens of a trace (24–50% of traces are clipped, and
  the clip rate itself drifts along the axis), and a per-trace *average* soundness cannot see
  the single decisive premise.
- One extractor, one judge, one benchmark. Nothing here is claimed to generalise.
