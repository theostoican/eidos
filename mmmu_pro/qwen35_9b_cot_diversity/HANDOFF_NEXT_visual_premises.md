# Handoff — NEXT: extract visual premises from the thinking trace, judge each premise's soundness

Read `HANDOFF.md`, `HANDOFF_RESULTS.md`, and `HANDOFF_MAJVOTE.md` first. This file defines the
next experiment. Everything below builds on the runs already done.

## Where things stand (one paragraph)

On MMMU-Pro / Qwen3.5-9B with the official prompt + thinking + recommended params, sweeping
**top_p** moves **diversity up** but leaves every **per-trace** quality metric **flat**:
per-sample answer accuracy (r=-0.013), and **whole-CoT soundness** — judged independently by
**Claude Sonnet** (no gold, 200 CoTs) — is **flat at ~0.61** across top_p (does NOT decrease).
The Qwen self-judge was unusable (circular ~0.89). The only metric with the inverted-U (peak
top_p≈0.7) is **majority-vote (self-consistency) accuracy**, a nonlinear aggregate. See
`outputs/cot_summary_all.png` for all of it in one chart.

**Key limitation this next step attacks:** whole-CoT soundness is a single coarse binary per
trace. A long reasoning trace makes MANY visual reads; some right, some wrong. Collapsing that to
one true/false throws away signal and is probably why soundness looks flat. The next experiment
measures soundness at the level of **individual visual premises extracted from the trace**.

## The next experiment

**Goal:** from each `<think>` trace, EXTRACT the discrete **visual premises** — the atomic
statements the model asserts about what it sees in the image (e.g. "the far-left inductor is
labelled j20Ω", "the woman holds a wax tablet", "curve C peaks at t=b") — then judge the
**soundness of EACH premise** against the image, and analyze **fraction-of-premises-correct**
(and premise diversity) vs top_p. This is the fine-grained version of the flat whole-CoT
soundness, and connects to the sibling `../qwen35_9b_premise_diversity/` experiment (which
generated premises with a separate prompt; here they come FROM the reasoning trace itself).

**Pipeline (reuse the harness already here):**
1. **Extract premises.** For each CoT (`outputs/sonnet_gen.shard*.jsonl` — 40 Q x 5 top_p x 2
   samples are on disk but gitignored; regenerate with `cot_gen.py` if gone), run an extraction
   step that pulls the visual premises as a numbered list. Two options:
   - LLM extraction (a Sonnet or local-Qwen pass: "list every distinct factual claim this trace
     makes about the image, one per line, verbatim-ish"), or
   - the sibling repo's premise-extraction approach (`../qwen35_9b_premise_diversity/extract_comp.py`).
2. **Judge each premise (Sonnet, per premise, STRICT, no gold).** Reuse the exact harness from
   this experiment: `sonnet_prep.py` (saves images + builds self-contained task files) and
   `sonnet_soundness_workflow.js` (the Workflow that fans out Sonnet subagents, each Reads the
   image + judges, returns `{sound}`). Adapt the task file to hold ONE premise (not the whole
   trace) + the image; the workflow schema/prompt barely changes. NOTE: pass `args` as a real
   JSON object AND parse defensively in the script (`typeof args==='string' ? JSON.parse : args`)
   — the first run got 0 agents because args arrived stringified (already fixed in the committed
   workflow script; keep that guard).
3. **Analyze.** Per (id, top_p): fraction of premises Sonnet ruled correct (= a graded, non-binary
   soundness), plus premise-count and premise diversity. Plot fraction-correct vs top_p — this is
   where a real decrease (if any) should show up, unlike the coarse whole-CoT binary.

**Judging notes learned here (don't relearn the hard way):**
- **No gold to the judge** — showing it makes the judge anchor on correctness (Qwen went 96% vs
  62%). Sonnet with no gold gave the honest ~0.61 whole-CoT number and caught real fabrications
  (misreading a wax tablet as a "mirror", inventing Latin inscriptions / an "initial L"). Keep it.
- **Sonnet, not the local Qwen self-judge**, for any soundness claim.
- **Scale:** whole-CoT was 200 Sonnet calls (~25 min, ~7.6M tokens). Per-premise is ~5-15x more
  calls (each trace has multiple premises) — budget accordingly, or subsample premises/cells.
- **More generation samples** if you also want the majority-vote curve (16-32, not 2).

## Files to reuse
| file | role |
|---|---|
| `cot_gen.py` | generation (official prompt, thinking, recommended params; 2D top_p x temps) |
| `sonnet_prep.py` | saves images + builds self-contained no-gold task files -> `outputs/sonnet_judge/` |
| `sonnet_soundness_workflow.js` | Workflow: fan out Sonnet subagents, judge, aggregate (args-parse guard included) |
| `plot_sonnet_soundness.py`, `plot_all.py` | figures |
| `outputs/sonnet_verdicts.json` | the 200 whole-CoT Sonnet verdicts (with per-premise reasons — useful seed) |
| `../qwen35_9b_premise_diversity/` | prior premise-level experiment (extraction + judge patterns) |

## Reproduce the whole-CoT Sonnet judge (starting point)
```bash
# generate (4 GPUs) if outputs/sonnet_gen.* is gone:
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python cot_gen.py --sample-frac 0.05 --limit 40 \
  --n-samples 2 --top-ps 0.5,0.7,0.9,0.95,1.0 --num-shards 4 --shard-id $i --resume \
  --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.92 \
  --out outputs/sonnet_gen.jsonl --questions-out outputs/sonnet_q.json & done; wait
python sonnet_prep.py            # -> outputs/sonnet_judge/task_*.txt (+ index.json)
# then run sonnet_soundness_workflow.js via the Workflow tool with
#   args = {"n": <#tasks>, "dir": "/abs/path/to/outputs/sonnet_judge"}
```
For the NEXT step, insert a premise-extraction stage before sonnet_prep and make each task file
hold one premise instead of the whole trace.
