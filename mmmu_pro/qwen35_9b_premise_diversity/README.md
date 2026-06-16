# eidos — Visual-premise diversity vs. correctness (Qwen3.5-9B on MMMU-Pro)

Does **sampling diversity** predict **correctness** when a vision-language model reads a
single *visual premise* (one fact read directly from an image) off an MMMU-Pro question?

## Experiment (v6)

- **Model:** Qwen/Qwen3.5-9B via vLLM, **thinking ON**, **4096-token cap**, prompt that
  encourages a short premise.
- **Examples:** 24 MMMU-Pro questions (one per subject), all drawn from baseline *failures*.
- **Sweep:** 5 `top_p` (0.5, 0.7, 0.9, 0.95, 1.0) × **16 samples** = **1920 generations**.
- Each premise is judged true/false **against the image** (vision LLM-as-judge).
- **Diversity** per (example, top_p) cell: **Vendi score** and mean pairwise cosine distance
  over MiniLM embeddings. **Correctness** = fraction of premises judged correct.

## Key findings

- **Diversity ↔ correctness is negative**: more scattered premises → more of them wrong.
  On the clean set (truncated samples removed) vendi↔correct Pearson ≈ **−0.26**
  (**−0.37** dropping 3 image-hallucination questions). With truncated samples included it
  looks stronger (−0.51), but ~half of that is a truncation artifact.
- **top_p barely moves correctness** (~0.79–0.83 across the range); it mainly raises diversity.
  Correctness is modestly best at **low top_p (≈0.5–0.9)**.
- **Truncation:** even at a 4096 cap with thinking on, **12% (232/1920)** of generations hit
  the cap before emitting a premise. The "clean" analysis drops these everywhere.

## Layout

| Path | What |
|------|------|
| `run_mmmupro.py` | Baseline MMMU-Pro inference (full 1730-question run) |
| `premise_gen_v6.py` | v6 premise sweep (thinking on, 4096, short-premise prompt) |
| `judge_prep.py` / `judge_merge.py` | Build judge packets / merge verdicts |
| `premise_analyze.py` | Diversity (Vendi, cosine) + correctness correlations |
| `plot_harmonic_per_example.py` | Per-example harmonic-mean plot |
| `analysis_v6.ipynb` | Self-contained Colab notebook (clean set, embedded data) |
| `outputs/` | premises / verdicts / reports / figures |
| `logs/` | run logs |

The vLLM virtualenv (`vllm-env/`) is intentionally git-ignored.
