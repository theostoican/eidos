# Handoff — CoT-diversity run COMPLETE + results + next steps

The experiment described in `HANDOFF.md` has now been **run at 10% scale and analyzed**.
This file records what was actually run, the headline result (which **overturned** the naive
reading), and the next steps. Read `HANDOFF.md` first for the original design.

## What was run (2026-07-11)

- **Hardware:** a **4× A100-SXM4-40GB** node (NOT the single 4090 the original HANDOFF assumed),
  driver 595 / CUDA 13.2. Pinned stack **vLLM 0.19.1** (cu128, torch 2.10) in `/venv/main`.
- **Config:** MMMU-Pro standard(10 opt) test, **10% deterministic sample = 173 Q** (seed 20260706),
  **top_p {0.5,0.7,0.9,0.95,1.0} × 6 samples** (reduced from 16 for speed), thinking ON,
  temp 1.0, **40960-token budget**, bf16 KV (exact params). **5,190 CoTs.**
- **Prompt:** switched to the **verbatim official MMMU-Pro prompt** (fetched from
  MMMU-Benchmark/MMMU `mmmu-pro/prompts.yaml` → `cot.standard`, with the official
  `infer_transformers.py` suffix assembly: `question\n{options}\n{prompt}`, `<image N>`→`[image]`,
  images appended after text). `cot_gen.py::build_content` was rewritten to match exactly.
- **Judge:** **local Qwen3.5-9B self-judge** (`judge_qwen_cot.py`, strict soundness rubric) — NOT
  Sonnet (no API key on box; far cheaper). 0.1% unparsed at scale.
- **Runtime:** generation ~3.1 h (all 4 GPUs), judging ~22 min (all 4 GPUs), analysis ~3 min.
  Launcher: `run_cot_all4.sh 0.10 6` (generate on 4 GPUs, then judge on 4 GPUs, merge).

## Headline result — the naive correlation was a TRUNCATION ARTIFACT

Raw analysis (all traces) suggested a **−0.35 diversity↔soundness** correlation — i.e. "more
diverse CoTs are less sound." **This is an artifact.** Root cause: `extract_cot.py`/`analyze_cot.py`
**keep truncated traces in the diversity computation** (by design), but truncation is strongly
top_p-dependent (**11.2% at top_p=0.5 → 0.6% at top_p=1.0**). A truncated trace is a long
degenerate repetition-loop (all 234 are pinned at the 40960 cap; completed traces avg ~7.6k
tokens) that (a) is an embedding outlier inflating its cell's Vendi, and (b) is auto-failed by the
judge rubric. That single mechanism manufactures the negative correlation.

**Controlling for it (completed traces only, `cot_report_stopped.json`):**

| relationship | RAW (all) | CONTROLLED (completed) |
|---|---|---|
| top_p ↔ diversity | −0.02 | **+0.19** (rises, as expected) |
| diversity ↔ CoT-soundness | −0.35 | **+0.01 (gone)** |
| diversity ↔ answer-accuracy | −0.17 | **+0.07 (gone)** |
| soundness vs top_p | 0.82→0.89 (rises) | ~0.89 flat |

**Conclusions:**
1. **No diversity↔correctness relationship at the CoT level** once truncation is controlled. The
   answer-accuracy version (+0.07) needs no judge, so this null is not a judge artifact.
2. **Diversity genuinely rises with top_p (+0.19)**, consistent with the premise experiment (+0.26).
3. **Low top_p → repetition-loop degeneration** (Holtzman-style) is the real phenomenon; it is what
   the naive analysis mistook for a diversity effect.

## Judge caveat — the self-judge is gold-anchored (quantified)

Qwen judged its own CoTs sound **96.0%** when the final answer was correct vs **62.0%** when wrong,
despite a rubric saying "right letter via wrong reasoning is UNSOUND." So `cot_correct` is largely a
proxy for `answer_correct`, not an independent quality signal. Treat it as soft; prefer the
judge-free answer-accuracy for the main claim.

## Files (this run)

| file | what | committed? |
|---|---|---|
| `cot_gen.py` | generation — **now uses the verbatim official prompt** | ✅ |
| `judge_qwen_cot.py` | local Qwen soundness judge (strict rubric, `--watch` streaming, shard-friendly) | ✅ |
| `judge_prep_cot.py` | (alt) build image+CoT judge packets for an external/Sonnet judge | ✅ |
| `run_cot_all4.sh` | generate on 4 GPUs then judge on 4 GPUs (**the launcher used**) | ✅ |
| `run_cot_parallel.sh` | alt: 3 GPUs generate + 1 GPU streaming-judges in parallel | ✅ |
| `analyze_cot.py` | analysis (⚠️ keeps truncated in diversity — see next steps) | ✅ (unchanged) |
| `plot_cot.py`, `plot_value_vs_topp.py` | figures (evolution, per-cell scatter, value-vs-top_p) | ✅ |
| `outputs/cot_report.json` / `cot_report_stopped.json` | results (all / completed-only) | ✅ (small) |
| `outputs/*.png` | figures — incl. `cot_value_vs_topp_completed.png` (the corrected chart) | ✅ |
| `outputs/cot_gen.jsonl`, `cot_extracted*.jsonl`, `verdicts_cot.jsonl` | raw (147M+) | ❌ gitignored, regenerable |

## Next steps (prioritized)

1. **FIX THE ANALYSIS (do first).** `analyze_cot.py` should exclude `finish_reason != 'stop'` traces
   (or report both) by default — currently it silently keeps them and reproduces the misleading
   −0.35. Add a `--completed-only` flag defaulting to True, or drop truncated in `extract_cot.py`.
   `cot_report_stopped.json` already has the correct completed-only numbers.
2. **Independent Sonnet-judge validation** (optional, cheap insurance). Use `judge_prep_cot.py` to
   emit packets and have Sonnet re-judge a **stratified subsample** (balanced correct/wrong, across
   top_p) with the same strict rubric. Compare agreement with Qwen to quantify the gold-anchoring
   (expect the 96/62 gap to shrink). Only needed if `cot_correct` will be used downstream.
3. **Mitigate truncation for any future run:** add a repetition penalty or `min_p` floor at low
   top_p (the root cause of the loops); raising `--max-tokens` alone won't help true infinite loops.
4. **Scale to the full dataset** if the 10% picture is worth confirming: `run_cot_all4.sh 1.0 6`
   (≈10× tokens → ~1.3–1.5 days on this node) — but note the core hypothesis already looks like a
   **null** at 10%, so weigh whether the full run is worth it, or pivot to the truncation/degeneration
   angle (which IS a real, top_p-dependent effect here).

## Reproduce

```bash
cd mmmu_pro/qwen35_9b_cot_diversity && source /venv/main/bin/activate
uv pip install "vllm==0.19.1" sentence-transformers datasets qwen-vl-utils matplotlib
./run_cot_all4.sh 0.10 6                       # ~3.5 h on 4× A100-40GB -> cot_gen.jsonl + verdicts_cot.jsonl
python extract_cot.py
python analyze_cot.py --verdicts outputs/verdicts_cot.jsonl        # raw (keeps truncated — see step 1)
# completed-only (the correct control):
python -c "import json;[open('outputs/cot_extracted_stopped.jsonl','w').write(''.join(l for l in open('outputs/cot_extracted.jsonl') if json.loads(l)['finish_reason']=='stop'))]"
python analyze_cot.py --extracted outputs/cot_extracted_stopped.jsonl --verdicts outputs/verdicts_cot.jsonl --out outputs/cot_report_stopped.json
python plot_value_vs_topp.py outputs/cot_report_stopped.json outputs/cot_value_vs_topp_completed.png "  — completed traces only"
```
