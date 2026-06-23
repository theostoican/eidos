# Handoff — comprehensive visual-premise experiment (for the next agent)

Context for whoever picks this up. This documents the current state of the
`mmmu-pro-premise-diversity` branch experiment dir
`mmmu_pro/qwen35_9b_premise_diversity/`.

## What this experiment is

Does **nucleus-sampling diversity predict perception correctness** when Qwen3.5-9B
reads the visual content of an MMMU-Pro question? The model is prompted to produce
**ONE comprehensive visual premise** — a single statement folding in EVERY fact it can
read off the image — and is explicitly told NOT to answer/solve. Each premise is judged
true/false against the image by a vision LLM judge (Claude) under a **strict-binary**
rule: correct only if EVERY fact in the statement is accurate (one misread fails it).

- 6 fixed examples (`outputs/questions.json`): test_Biology_259, test_Chemistry_400,
  test_Clinical_Medicine_302, test_Electronics_137, test_Physics_61,
  validation_Clinical_Medicine_16. Five are MMMU-Pro standard(10 options) test; the
  validation one is the all85-notebook 10-option recast of MMMU validation
  (gold J = Glioblastoma) with the genuine MMMU image.
- Sweep: top_p {0.5,0.7,0.9,0.95,1.0} x 16 samples = 480 gens, thinking ON, temp 1.0,
  top_k off, 16384-token budget. Truncated (finish_reason != 'stop') removed -> 408 clean.

## Headline results (see README.md for the full table)

- Overall premise accuracy **56.1%** (229/408).
- **vendi <-> correct: Pearson -0.56, Spearman -0.59** (cosine -0.66 / -0.70) over the 30
  (example, top_p) cells — the negative diversity<->correctness relationship is strong here.
- top_p <-> correctness ~ -0.15 (0.591 -> 0.510 across 0.5->1.0); top_p <-> vendi +0.26.
- Per-example accuracy: Biology 98%, Physics 64%, Chemistry 50%, Electronics 49%,
  Clinical_Medicine_302 32%, validation_Clinical_Medicine_16 29%.

## Backstory: three prompt variants were explored this session

Only the **comprehensive** variant remains in the repo (the user asked to leave no past
artifacts). The other two were run and then removed; their numbers, for reference:
1. **One-premise** (a single narrow fact/sample): overall acc 82.2%, vendi<->correct
   -0.41 / -0.49. Limitation: the model picks only the most salient fact, so decisive
   misreads in skipped components are invisible.
2. **Enumeration** (a numbered list of ~13 separate facts/sample; 5,721 pooled premises):
   overall acc 90.9%, vendi<->correct -0.40 / -0.31. Each fact graded independently.
3. **Comprehensive** (this repo): one ~94-word premise/sample, strict-binary. Strongest
   negative correlation.

## The Electronics_137 perception finding

test_Electronics_137 (nodal analysis, gold A = 90 + j120) was a baseline failure: the
model misreads the **far-left inductor as j120Ω when the image shows j20Ω**, which
propagates through correct algebra to the wrong option F (30 + j120). Verified: solving
with j20 -> 90 + j120 (A); with j120 -> 31.5 + j118 (F). In the comprehensive premises
this exact misread is what flips a premise from correct to wrong (everything else in the
statement is accurate). Per-cell: 8/16 correct at top_p 0.5, 11/16 at top_p 1.0.

## File map

| file | what |
|------|------|
| `premise_gen_comp.py` | generation (vLLM, self-contained; PREMISE_HEADER = the comprehensive prompt) |
| `extract_comp.py` | extract the one `Premise:` statement/sample -> premises_comp_extracted.jsonl |
| `judge_prep_comp.py` | one judge packet per question -> outputs/judge_comp/<id>.json |
| `merge_comp.py` | flatten <id>.verdicts.json -> verdicts_comp.jsonl |
| `analyze_comp.py` | Vendi + cosine diversity + correctness + correlations -> premise_report_comp.json |
| `plot_comp.py` | comp_diversity_vs_correctness.png + harmonic_per_example_comp.png |
| `outputs/premises_comp.jsonl` | 480 raw generations (full <think> + premise) |
| `outputs/verdicts_comp.jsonl` | 408 strict-binary verdicts |
| `outputs/judge_comp/` | per-question packets + Claude verdicts + _index.json |

## Reproduce / continue

```bash
python premise_gen_comp.py        # GPU; needs vllm, datasets, sentence-transformers
python extract_comp.py
python judge_prep_comp.py
# re-judge: run a vision judge over each outputs/judge_comp/<id>.json -> <id>.verdicts.json,
#   STRICT-BINARY (premise correct only if every fact is accurate). One agent per image.
python merge_comp.py
python analyze_comp.py
python plot_comp.py
```

## Open items / possible next steps

- The judge is **Claude**, ad hoc (no committed deterministic judge script). For
  comparability a canonical judge model/prompt could be committed and re-run.
- The strict-binary rubric makes accuracy heavily length-dependent (more facts -> more
  chances to fail). A "fraction-of-facts-correct" rubric would decouple that.
- Env: Qwen/Qwen3.5-9B runs on a 24GB GPU at max_model_len 20480; top_p>=0.95 can OOM at
  high concurrency — use max_num_seqs 4-6. Analysis embedding needs sentence-transformers + GPU.
