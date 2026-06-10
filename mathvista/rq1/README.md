# RQ1 — single prompt per mode

Same target as RQ2: the **8 MathVista failure cases** that `Qwen/Qwen3.5-9B` got wrong
(mined in [`../rq2/failures/`](../rq2/failures/)). The research question is whether
**attention-directing focus modes** can recover those failures — but here we test the
*cheap, single-call* version of that idea.

## RQ1 vs RQ2

| | RQ2 (`../rq2/`) | RQ1 (here) |
|---|---|---|
| Stage 1 | **generate** a visual premise per mode (`run_premises.py`) | — |
| Stage 2 | **re-answer** the question conditioned on each premise (`run_answer_per_premise.py`) | — |
| Single stage | — | **one prompt per mode** that directs attention *and* asks the question |
| Model calls per (image, mode) | 2 | **1** |
| Modes | 20 focus modes | the same 20 modes |

RQ1 folds the "look here, then answer" into a **single prompt**: `Focus: <mode> …`
immediately followed by the question and the dataset's answer-format hint. The model is
queried **once per mode for each image**, scored with the same MathVista-style heuristic
extractor/grader as RQ2, and performance is reported at the end.

## Run it

```bash
# from this rq1/ directory (reads the 8 images from ../rq2/failures/)
pip install -U "transformers>=4.57" accelerate datasets qwen-vl-utils pillow
python run_single_prompt_per_mode.py        # or: MAX_NEW=768 python run_single_prompt_per_mode.py
```

## Outputs (`answers_per_mode/`)

- `pid_<n>.json` — per case: every mode's prompt, raw response, extracted prediction, and correctness.
- `summary.json` — per-mode correct counts, cases fixed by ≥1 mode, fixed pids.
- `single_prompt_report.md` — human-readable table per case.

"Fixed" = at least one mode's single-call answer matches the gold label that the
unguided baseline missed.

## Visualize the results

[`single_prompt_analysis.ipynb`](single_prompt_analysis.ipynb) (Colab/Jupyter, no GPU —
the RQ1 analogue of [`../rq2/premise_analysis.ipynb`](../rq2/premise_analysis.ipynb))
reads the saved outputs and shows, per case, the image + question, the prompt template,
and every mode's answer, then the accuracy aggregations and a per-case **RQ1 vs RQ2**
comparison. Executed with outputs embedded; in Colab the first cell clones the repo.
