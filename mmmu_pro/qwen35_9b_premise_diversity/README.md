# Comprehensive visual-premise vs. correctness (Qwen3.5-9B on MMMU-Pro)

Does **sampling diversity** predict **perception correctness** when a vision-language
model writes ONE *comprehensive* visual premise — a single statement that incorporates
EVERY fact it can read off the image (not solve the question)? Each comprehensive
premise is judged true/false against the image under a **strict-binary** rule: correct
only if every fact in it is accurate.

## Experiment

- **Model:** Qwen/Qwen3.5-9B via vLLM, thinking ON, 16384-token budget, temperature 1.0,
  top_k off.
- **Examples (6):** `outputs/questions.json` — test_Biology_259, test_Chemistry_400,
  test_Clinical_Medicine_302, test_Electronics_137, test_Physics_61,
  validation_Clinical_Medicine_16.
- **Prompt (comprehensive):** "write ONE comprehensive visual premise that incorporates
  EVERY fact you can read directly from the image … read labels character-by-character …
  Do NOT answer or solve." Output is a single `Premise: <long statement>`. See
  `PREMISE_HEADER` in `premise_gen_comp.py`.
- **Sweep:** 5 `top_p` (0.5, 0.7, 0.9, 0.95, 1.0) × 16 samples = 480. Truncated
  generations (finish_reason≠'stop') removed → **408 clean premises** (mean 94 words),
  30 (example,top_p) cells, ~13.6/cell.
- **Judge:** vision LLM-as-judge (Claude), one verdict per premise, **strict binary** —
  a premise is correct only if EVERY fact in it is accurate; any single misread fails it.

## Key results

- **Overall premise accuracy 56.1%** (229/408) — low because each ~94-word premise packs
  ~13 facts and one misread fails the whole statement.
- **Diversity ↔ correctness is strongly negative** (strongest of the premise variants):
  vendi↔correct Pearson **−0.56**, Spearman **−0.59**; cosine **−0.66 / −0.70**.
- **top_p barely moves correctness** (−0.15; 0.591→0.510 across 0.5→1.0) while raising
  diversity (top_p↔vendi +0.26).
- Per-example accuracy: Biology 98%, Physics 64%, Chemistry 50%, Electronics 49%
  (~half misread the j20Ω inductor as j120Ω), Clinical_Medicine_302 32%,
  validation_Clinical_Medicine_16 29%.

|  top_p | Vendi | cos_dist | frac_correct |
|-------:|------:|---------:|-------------:|
|    0.5 |  1.85 |    0.144 |        0.591 |
|    0.7 |  1.85 |    0.131 |        0.586 |
|    0.9 |  2.09 |    0.157 |        0.470 |
|   0.95 |  2.12 |    0.162 |        0.506 |
|    1.0 |  2.12 |    0.159 |        0.510 |

## Pipeline

| script | what |
|--------|------|
| `premise_gen_comp.py` | Comprehensive single-premise sweep → `outputs/premises_comp.jsonl` |
| `extract_comp.py` | Extract the one `Premise:` statement per sample → `outputs/premises_comp_extracted.jsonl` |
| `judge_prep_comp.py` | One judge packet per question → `outputs/judge_comp/<id>.json` |
| (vision judge) | Claude scores each premise strict-binary → `outputs/judge_comp/<id>.verdicts.json` |
| `merge_comp.py` | Flatten verdicts → `outputs/verdicts_comp.jsonl` |
| `analyze_comp.py` | Diversity (Vendi, cosine over MiniLM) + correctness → `outputs/premise_report_comp.json` |
| `plot_comp.py` | Scatter + per-example accuracy + harmonic-mean figures |

## Reproduce

```bash
python premise_gen_comp.py                      # generate (GPU)
python extract_comp.py
python judge_prep_comp.py
# vision-judge each outputs/judge_comp/<id>.json -> <id>.verdicts.json  (Claude, per image)
python merge_comp.py
python analyze_comp.py
python plot_comp.py
```
