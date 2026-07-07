# Handoff — CoT-diversity experiment (for the next agent, on a larger GPU)

You are taking over an experiment that is **fully built and calibrated but not yet run
at scale**, because the current box (single RTX 4090, 24 GB) is too small to finish in a
reasonable time. Everything you need to launch is here; the main thing you bring is a
**bigger GPU**. Read this whole file before running anything.

Experiment dir: `mmmu_pro/qwen35_9b_cot_diversity/` (branch `mmmu-pro-premise-diversity`).
It extends the premise-diversity experiment one dir over (`qwen35_9b_premise_diversity/`,
see its `HANDOFF.md`).

---

## 1. What this experiment is (design — CONFIRMED with the user)

Take the premise-diversity idea but apply it to the model's **actual chain-of-thought**
while it answers the real question:

- **Data:** a deterministic **5%** sample of **MMMU-Pro standard (10 options)** `test`
  (1730 Q → **87 questions**; not just failures). Seed `20260706`, written to
  `outputs/questions_5pct.json`. *(For the FULL dataset run the user later asked about,
  use `--sample-frac 1.0`.)*
- **Prompt:** the **official MMMU-Pro prompt** (identical to
  `qwen35_9b_thinking_full/run_mmmupro.py` `PROMPT_HEADER` — "Answer the following
  multiple-choice question … Answer: $LETTER … Think step by step"). NOT a premise prompt.
- **Sweep:** top_p **{0.5, 0.7, 0.9, 0.95, 1.0} × 16 samples**, thinking **ON**
  (`enable_thinking=True`), **temp 1.0, top_k off**, **40960-token budget**
  (`max_model_len 49152`), seed 1234. "Model params the same" as the repo's diversity sweep.
- **Unit of analysis:** the `<think>…</think>` **CoT** of each sample (kept full; the final
  `Answer: X` is kept too).
- **Three quantities, all tracked as an EVOLUTION over top_p:**
  1. **Diversity** of the 16 CoTs per (id, top_p) cell — Vendi score + mean pairwise cosine
     distance over MiniLM embeddings. CoTs are long, so each CoT is **chunked (~180 words)
     and its chunk-embeddings mean-pooled** into one vector (MiniLM caps at ~512 tokens;
     embedding only the opening would misrepresent a 20k-token trace).
  2. **CoT correctness** = **LLM-judged reasoning soundness** (user's explicit choice): a
     vision judge reads the image + full CoT and rules the reasoning sound/unsound,
     independent of the final letter. **Judge harness NOT yet built** — see §5.
  3. **Final-answer accuracy** = parsed `Answer: X` vs gold, per-sample mean; plus
     **majority-vote (self-consistency)** accuracy per question.

---

## 2. CRITICAL environment gotcha (read before you `pip install` anything)

This instance's NVIDIA driver is **570.86 → max CUDA 12.8** (`vast-capabilities | jq
'.hardware.gpu.cuda'`). The current **vLLM (0.20.2–0.24.0) are all built for CUDA 13**
(`libcudart.so.13`, pin torch 2.11) and **will not run** on a ≤12.8 driver (and this is a
consumer Ada card, so no forward-compat).

**The working combo is pinned:** **vLLM 0.19.1** — the newest cu12 build
(`libcudart.so.12`, torch 2.10.0+cu128) that **still natively registers
`Qwen3_5ForConditionalGeneration`** (it ships `qwen3_5.py`). Installed versions
(`outputs/env_versions.txt`):

```
vllm==0.19.1  torch==2.10.0(+cu128)  transformers==5.13.0
sentence-transformers==5.6.0  datasets==5.0.0  qwen-vl-utils==0.0.14  numpy==2.2.6
```

**On your larger GPU:** check ITS `driver_max_cuda` first.
- If the new box's driver supports **CUDA 13** (H100/H200/most modern datacenter hosts
  usually do), you MAY use a current cu13 vLLM (≥0.24) — likely faster/better Qwen3.5
  support. Verify it still registers `Qwen3_5ForConditionalGeneration`.
- If it's **≤12.8**, reuse the pinned **vLLM 0.19.1** recipe above.
- **Blackwell (B200, cc 10.0/12.0)** needs CUDA ≥12.8 wheels — 0.19.1 (cu128) should work,
  but a cu13 vLLM is the safer match there. See base image guide §12.

Install pattern that worked here:
```bash
source /venv/main/bin/activate
uv pip install sentence-transformers datasets qwen-vl-utils
# then the CUDA-matched vLLM (here: the cu12 one)
uv pip install "vllm==0.19.1"
python -c "import vllm,torch;print(vllm.__version__,torch.__version__,torch.cuda.is_available())"
```

Model: **`Qwen/Qwen3.5-9B`** (public, image-text-to-text, ~17.7 GB bf16). It is a **hybrid
GDN (gated-delta-net linear attention) + full-attention** model — that's why vLLM logs
mamba/attention page-size lines; harmless. 32 layers, 4 KV heads, head_dim 256 →
**128 KB/token bf16 KV**.

**Two gotchas already handled in the code (don't re-break them):**
- **Thinking format:** Qwen3.5's chat template emits the opening `<think>` as part of the
  PROMPT, so generated text is `[reasoning] </think> [answer]` — the opening tag is
  usually ABSENT and the **closing `</think>` is the delimiter**. `extract_cot.py` splits
  on `</think>` (verified: 31/32 calib gens closed it; the 1 that didn't was truncated).
- **Embedding deps:** `sentence-transformers 5.6.0` hard-imports `torchcodec`, which needs
  FFmpeg 4/5 (`libavutil.so.56`) that this box (FFmpeg 6) lacks. To avoid that entirely,
  `analyze_cot.py` loads MiniLM via **plain `transformers` (mean-pooling)** — so the
  analysis needs NO sentence-transformers/torchcodec/FFmpeg. (FFmpeg 6 was apt-installed
  anyway, but the analyzer doesn't rely on it.)

---

## 3. Calibration findings (why a bigger GPU is needed)

Ran 2 questions × 16 samples @ top_p 0.9, full 40k budget (`outputs/calib_*`):

- **CoTs are long:** out_tokens **mean ≈15.9k, median ≈20k, p90 ≈30k**, 1/32 hit the 40k cap.
- **Throughput on the 4090: 268 tok/s**, because fp8-KV still left only **2.2 GB KV →
  35,904-token pool → 2.72× concurrency** at 49k ctx. Weights (17.7 GB) starve the KV cache.
- **Projection (5% · 5 top_p · 16 · bf16 · 40k ≈ 110M tokens):** **~4.8 days** on this 4090.

The bottleneck is **KV capacity → concurrency**, not raw FLOPs. A big-VRAM GPU fixes it.

### GPU sizing (estimates from the 2-Q calibration — re-calibrate to confirm!)

| Target | Tokens | Needs | Est. hardware |
|---|---|---|---|
| **5% run** (87 Q × 5 top_p × 16) | ~110 M | keep bf16, big KV | **1× A100/H100 80 GB → ~9–14 h** |
| **FULL run** (1730 Q × 5 top_p × 16) | ~2.2 B | ~25k tok/s for 24 h | **8×H100 (~12 h)** or **8×A100 (~24–26 h)**; 4×H100 or 3×H200 ≈ 24 h |

Per-GPU aggregate decode assumed: A100 ~3k, H100 ~6k, H200 ~9k tok/s (9B, ~20k ctx, high
concurrency). fp8 **weights** would ~halve the GPU count but change outputs (not "params
the same").

---

## 4. Exactly how to run on the larger GPU

**Step 0 — re-calibrate (10 min)** to lock throughput and pick `--max-num-seqs`:
```bash
cd mmmu_pro/qwen35_9b_cot_diversity && source /venv/main/bin/activate
python cot_gen.py --limit 3 --top-ps 0.9 --n-samples 16 \
  --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.92 \
  --questions-out outputs/calib_q.json --out outputs/calib.jsonl
# watch the "Maximum concurrency" + "GPU KV cache size" lines and decode tok/s
```

**Step 1 — generation on the multi-GPU node (CONFIRMED plan).** Just use the launcher —
it detects all GPUs, runs one vLLM engine per GPU over a strided question slice (data-
parallel, `cot_gen.py --num-shards/--shard-id`), pre-warms the dataset+model cache once,
`--resume`s automatically, and merges the shard outputs:
```bash
cd mmmu_pro/qwen35_9b_cot_diversity
nohup ./run_node.sh 1.0 > logs/node.log 2>&1 &   # 1.0 = FULL MMMU-Pro (1730 Q); use 0.05 for the 5% run
tail -f logs/shard0.log                          # watch one shard
```
Output: `outputs/cot_gen.jsonl` (merged) + `outputs/questions.json`. Re-running the launcher
resumes (skips (id,top_p) cells already complete). **Flags baked into the launcher:**
`--kv-cache-dtype auto` (bf16 KV — fp8 was ONLY the 24GB workaround; on ≥48GB use bf16 to
keep params EXACT), `--max-num-seqs 128`, `--gpu-mem-util 0.92`. For <48GB cards pass
`--kv-cache-dtype fp8` as an extra arg: `./run_node.sh 1.0 --kv-cache-dtype fp8`.

Manual / single-process alternatives (the launcher wraps these):
```bash
# data-parallel by hand (one GPU): CUDA_VISIBLE_DEVICES=i python cot_gen.py --num-shards N --shard-id i --resume ...
# tensor-parallel (one engine, all GPUs): python cot_gen.py --tensor-parallel-size N ...   # simpler, lower throughput for 9B
```
Notes: DP scales more linearly than TP for a 9B model. `--num-shards>1` auto-suffixes the
out/questions filenames with `.shard{i}`; `run_node.sh` cats them back together.

**Step 2 — extract, then analyze:**
```bash
python extract_cot.py   --gen outputs/cot_gen.jsonl --out outputs/cot_extracted.jsonl
python analyze_cot.py   --extracted outputs/cot_extracted.jsonl \
                        --verdicts outputs/verdicts_cot.jsonl   # verdicts optional
```
`analyze_cot.py` prints the **evolution table** (vendi, cos_dist, answer_acc, majority_acc,
cot_correct per top_p) + correlations (top_p↔each; diversity↔answer_acc; diversity↔cot_correct)
and writes `outputs/cot_report.json`. If `verdicts_cot.jsonl` is absent it still does
everything except `cot_correct` (prints n/a).

---

## 5. The judging piece — NOT built yet, and it's the expensive open decision

"CoT correctness" = a **vision judge (Claude) ruling each CoT's reasoning sound**. At full
scope that's up to **7,000 judge calls** (138,400 for the full dataset), each reading a
~16k-token trace + image → ~110 M (≈2 B) judge-input tokens. **The GPU does NOT help this**
(it's API-bound) unless you use a local judge. The user has NOT yet chosen a scope. Options
to raise with them / implement:

1. **Subsample judging** — judge 4 of 16 samples per (id, top_p) cell → hundreds of calls;
   `cot_correct` = fraction sound over the subset. (Recommended for the 5% run.)
2. **Full judging** — every CoT; only viable for the 5% run, and still large.
3. **Local-Qwen self-judge** — run Qwen3.5-9B on the same node as the judge; ~free, scales,
   but weaker/circular. (Basically required for the FULL dataset.)
4. **Answer-match only** — drop the reasoning judge; `cot_correct := (answer==gold)`.
   Removes all judge cost; overrides the user's earlier "LLM-judged" choice.

**To build it** (mirror the premise experiment's `judge_prep_comp.py` → verdicts flow):
write `judge_prep_cot.py` that, per sample, emits a packet {image (save from the dataset via
`ds_index` in `questions_5pct.json`), question, options, gold, the CoT text} and have the
judge return **`{id, top_p, sample_idx, sound: bool}`** lines into `outputs/verdicts_cot.jsonl`
(the schema `analyze_cot.py` already reads). Use a STRICT rubric: reasoning is "sound" only
if the visual reads and inferential steps are correct (define precisely, like the premise
strict-binary rule).

---

## 6. File map

| file | what | status |
|------|------|--------|
| `run_node.sh` | **multi-GPU-node launcher**: 1 engine/GPU, data-parallel shards, cache pre-warm, resume, merge | ✅ built, syntax-checked |
| `cot_gen.py` | generation: sample, official prompt, top_p sweep, thinking ON, fp8/bf16 KV, TP + DP-shard + resume | ✅ built, calibrated + logic-tested |
| `extract_cot.py` | split on `</think>` + parse `Answer: X` → `cot_extracted.jsonl` | ✅ built + **tested on real gens** |
| `analyze_cot.py` | Vendi+cosine on chunked CoT (MiniLM via transformers), answer acc, majority vote, evolution-vs-top_p, correlations | ✅ built + **tested on real gens** |
| `judge_prep_cot.py` | build judge packets → `verdicts_cot.jsonl` | ❌ NOT built (see §5) |
| `plot_cot.py` | evolution plots (diversity/correctness/accuracy vs top_p) | ❌ NOT built (copy `../qwen35_9b_premise_diversity/plot_comp.py` pattern) |
| `outputs/questions_5pct.json` | the deterministic 5% sample (id → subject, gold, ds_index…) | written on first real run |
| `outputs/calib_*` | 2-Q calibration artifacts (log, gens, summary) | ✅ present |
| `outputs/env_versions.txt` | pinned working versions | ✅ present |

---

## 7. Decisions / status

1. **Compute plan — DECIDED: multi-GPU node, FULL MMMU-Pro (1730 Q).** Target hardware
   **~8×H100 (~12 h)** or **~8×A100 (~24–26 h)** (4×H100 or 3×H200 also ≈24 h). Run it with
   `./run_node.sh 1.0`. (The 5% run — `./run_node.sh 0.05` — fits on a single 80 GB card in
   ~9–14 h if they ever want the cheaper version.)
2. **Judging scope — STILL OPEN, but the scale forces it.** At full dataset that's up to
   138,400 CoTs. **Claude/full and Claude/subsample are effectively out**; realistic choices
   are **local-Qwen self-judge** (runs on the same node) or **answer-match only**. Pin this
   with the user before/while generating. Everything except `cot_correct` (diversity,
   answer accuracy, majority vote, evolution) computes regardless.
3. **Verify assumptions FIRST:** the ~15.9k-token CoT average is from only 2 (hard)
   questions. Do Step 0's 10-min calibration on the real node, confirm decode tok/s and the
   time estimate, and tune `--max-num-seqs` before kicking off the full ~24 h run.

---

## 8. Reproduce / sanity check what's here

```bash
cd mmmu_pro/qwen35_9b_cot_diversity && source /venv/main/bin/activate
# extract + analyze the existing 2-Q calibration (no judge) to see the pipeline end-to-end:
python extract_cot.py --gen outputs/calib_cot_gen.jsonl --out outputs/calib_extracted.jsonl
python analyze_cot.py --extracted outputs/calib_extracted.jsonl --out outputs/calib_report.json
```
This exercises extract+analyze on real generations. **Expected (already verified here):**
extract → `has_think=31/32 (97%)`, `answer_correct 26/32 (81.2%)`; analyze → one row
`top_p=0.9  vendi=1.71  cos_d=0.097  ans_acc=0.812  maj_acc=1.000  cot_ok=n/a`. Correlations
are NaN on the calibration (only one top_p / two questions → no variance); they populate once
the real 5-top_p sweep exists. `cot_correct` stays n/a until judging (§5) is built and run.

Calibration sample question ids (for reference): `test_Electronics_247`, `test_Art_Theory_420`.
