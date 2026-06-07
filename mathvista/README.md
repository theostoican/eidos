# Qwen3.5-9B × MathVista — failure mining

Runs the multimodal **`Qwen/Qwen3.5-9B`** model on the **MathVista** (`AI4Math/MathVista`,
`testmini` split) benchmark, scores each prediction against the ground truth, and **saves the
inputs the model gets wrong** into [`failures/`](failures/) (image + metadata) for inspection.

The loop streams through `testmini` until it has collected `TARGET_FAILURES` (default **10**)
*completed* wrong answers, then stops — so a run finishes in a few minutes on a single 24 GB
GPU (developed on an RTX 4090 on vast.ai).

## Contents

| File | What it is |
|------|------------|
| [`qwen35_mathvista_failures.ipynb`](qwen35_mathvista_failures.ipynb) | The notebook (Colab/Jupyter), executed with outputs + a failure-grid figure embedded. |
| [`run_eval.py`](run_eval.py) | The same pipeline as a standalone script (streams per-sample progress to stdout). |
| [`failures/`](failures/) | One sub-folder per wrong case (`image.png` + `info.json`), plus `summary.json`. |

## How it works

1. **Model** — `Qwen/Qwen3.5-9B`, loaded with `AutoModelForImageTextToText` in bf16 (~18 GB).
   It is a reasoning VL model; we run it with `enable_thinking=False` so it answers concisely
   (long chain-of-thought is much slower and gets truncated) and `max_new_tokens=1536`.
2. **Prompt** — the dataset's own `query` field (question + answer-format hint) plus the image,
   wrapped in the chat template. No custom/system prompt.
3. **Scoring** — heuristic, MathVista-style: option-letter match for multiple-choice,
   numeric-tolerance for free-form numbers, string match for free-form text. A small extractor
   pulls the final answer out of the model's text (`\boxed{...}`, `Answer: X`, trailing line).
4. **Failure selection** — only **completed** (`finished=True`) wrong answers count. A truncated
   generation never emitted a final answer, so its "wrong" prediction would be an extraction
   artifact; those are skipped (and counted as `n_truncated_skipped`).

## Run it

```bash
# in an env that already has torch (e.g. the vast.ai image)
pip install -U "transformers>=4.57" accelerate datasets qwen-vl-utils pillow matplotlib
python run_eval.py           # or: TARGET_FAILURES=5 MAX_SAMPLES=80 python run_eval.py
```

## Caveat: a "failure" is a *disagreement with the gold label*

On a noisy benchmark, some disagreements are the benchmark's fault, not the model's. In the
sample run, two of the ten saved cases were **gold-label errors where the model was actually
correct** (`pid_20`, a ChartQA item; `pid_50`, an AI2D food-web item), and one was a phrasing
technicality (`pid_7`). Automated scoring cannot tell these apart from real model errors —
eyeball the saved cases before drawing conclusions.
