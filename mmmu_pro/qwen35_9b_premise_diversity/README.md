# eidos — Visual-premise diversity vs. correctness (Qwen3.5-9B on MMMU-Pro)

Does **sampling diversity** predict **correctness** when a vision-language model reads a
single *visual premise* (one fact read directly from an image) off an MMMU-Pro question?

## Experiment (v6)

- **Model:** Qwen/Qwen3.5-9B via vLLM, **thinking ON**, **4096-token cap**, prompt that
  encourages a short premise.
- **Examples:** 24 MMMU-Pro questions (one per subject), all drawn from baseline *failures*.
- **Sweep:** 5 `top_p` (0.5, 0.7, 0.9, 0.95, 1.0) × **16 samples** = **1920 generations**.
  Generations that hit the 4096-token cap before emitting a premise (`finish_reason='length'`,
  **232 = 12%**) are **truncated runs and are removed from every part of the analysis**, leaving
  **1688 samples** (mean **14.1** per (example, top_p) cell, range 4–16). All 24 questions are kept.
- The **extracted premise** (the single visual fact on the model's `Premise:` line, parsed by
  `extract_premises.py`) is the unit of analysis: both the diversity embedding and the
  correctness judge operate on it, not on the `<think>` trace, so the two measure the same object.
- Each extracted premise is judged true/false **against the image** (vision LLM-as-judge): true =
  an accurate reading of the image, false = a misread value/label/location/identity/relation.
  Overall premise accuracy is **85.6%** (1445/1688 judged correct).
- **Diversity** per (example, top_p) cell: **Vendi score** and mean pairwise cosine distance
  over MiniLM embeddings of the extracted premise. **Correctness** = fraction of premises judged correct.

## Key findings

- **Diversity ↔ correctness is negative**: more scattered premises → more of them wrong.
  With truncated runs removed and both the embedding and the judge operating on the extracted
  premise, vendi↔correct Pearson ≈ **−0.32** (Spearman **−0.49**), rising to **−0.56** when the 3
  image-hallucination questions are dropped (those are *low* diversity *and* low correctness, so
  they sit off the trend). *(Leaving truncated runs in had inflated the raw figure to −0.51 — about
  half of it a truncation artifact — which is why they are excluded everywhere here.)*
- **top_p barely moves correctness** (~0.81–0.86, a mild decline across the range); it mainly
  raises diversity. Correctness is modestly best at **low top_p (≈0.5–0.7)**.

## Layout

| Path | What |
|------|------|
| `run_mmmupro.py` | Baseline MMMU-Pro inference (full 1730-question run) |
| `premise_gen_v6.py` | v6 premise sweep (thinking on, 4096, short-premise prompt) → raw full text |
| `extract_premises.py` | Parse the `Premise:` line out of each raw generation → extracted premise |
| `judge_prep.py` / `judge_merge.py` | Build judge packets (premise-only) / merge verdicts |
| `premise_analyze.py` | Diversity (Vendi, cosine) + correctness correlations |
| `plot_harmonic_per_example.py` | Per-example harmonic-mean plot |
| `analysis_v6.ipynb` | Self-contained Colab notebook (clean set, embedded data) |
| `outputs/` | premises / verdicts / reports / figures |
| `logs/` | run logs |

The vLLM virtualenv (`vllm-env/`) is intentionally git-ignored.
