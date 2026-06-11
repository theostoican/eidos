# Best-of-N control vs two-stage premise inference (MathVista)

Does **two-stage premise inference with focus modes** recover failures that a plain
**best-of-N monolithic** model cannot? Run on **50 MathVista `testmini`** examples with
`Qwen/Qwen3.5-9B` (`enable_thinking=False`, `max_new_tokens=1024`).

## The two arms

| | Control (arm a) | Two-stage (modes) |
|---|---|---|
| prompt | the dataset's vanilla `query` — **no focus modes** | per mode: generate a visual premise (greedy), then answer conditioned on it (greedy) |
| diversity from | **20 temperature samples** (T=0.8, top_p=0.95) | **20 focus-mode prompts** (greedy) |
| calls / example | 1 greedy (defines pass/fail) + 20 samples | 20 × (premise + answer) |

A **failure** = the greedy baseline is wrong **and finished** (truncated generations excluded).
The two-stage arm is run on exactly the control's failures.

## Result (8 genuine failures)

| approach | failures recovered |
|---|---|
| control — majority vote (deployable) | **0/8** |
| two-stage — majority vote (deployable) | **1/8** (pid 27) |
| two-stage — oracle (best of 20 modes) | **2/8** (pid 19, 27) |

Two-stage only comes out ahead at the **oracle** ceiling (a perfect mode-picker, 2× the calls),
and by **2 cases out of 8** — direction/mechanism, not a measured effect size (n=8). The
unrecoverable failures (pids 7, 20, 50) are benchmark **label noise** (a phrasing technicality and
two gold-label errors where the model is arguably right).

## Files

- [`analysis.ipynb`](analysis.ipynb) — **Colab/Jupyter, no GPU.** Reads the saved outputs and
  shows the prompts, the per-failure answer distributions, the control-vote-vs-two-stage-oracle
  comparison, and **worked examples with the actual premise/answer prompts** (pids 19, 27).
  Executed with outputs embedded; in Colab the first cell clones the repo.
- `run_monolithic_best_of_n.py` — the control (best-of-N monolithic). → `monolithic_best_of_n/`
- `run_two_stage_on_failures.py` — two-stage modes on the control's failures. → `two_stage_on_failures/`
- `prep_analysis_data.py` — bakes `analysis_data.json` + `failure_examples/*.png` so the
  notebook is self-contained on Colab.
- `build_notebook.py` — regenerates `analysis.ipynb`.

## Reproduce

```bash
pip install -U "transformers>=4.57" accelerate datasets qwen-vl-utils pillow matplotlib
python run_monolithic_best_of_n.py        # N_EXAMPLES=50 N_SAMPLES=20 GEN_BATCH=1 MAX_NEW=1024
python run_two_stage_on_failures.py        # reads monolithic_best_of_n/summary.json for failure pids
python prep_analysis_data.py               # dataset-derived example data + images
python build_notebook.py                   # executes analysis.ipynb
```
