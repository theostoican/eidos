# Qwen3.5-9B — MMMU-Pro (standard, 10 options), full benchmark, thinking mode

Full-benchmark inference run of **`Qwen/Qwen3.5-9B`** on the entire
**MMMU-Pro `standard (10 options)`** split (1,730 questions), with thinking mode
enabled and a 40,960-token output budget. Served with vLLM (offline batched
inference) on a single NVIDIA RTX 4090.

## Results

| Metric | Value |
|---|---|
| **Accuracy** | **69.77%** (1,207 / 1,730) |
| Reference (Qwen reported MMMU-Pro) | ~69.2% |
| Unparseable answers | 2 (0.1%) |
| Hit the 40,960-token cap | 104 (6.0%) |
| Mean / median output tokens | 8,952 / 4,093 |
| Total output tokens | 15,487,169 |
| Decode throughput (aggregate) | 340 tok/s |
| Wall-clock | 12h 42m (single RTX 4090) |

The 69.77% lands essentially on the model's reported MMMU-Pro number — a sanity
check that prompting, image interleaving, thinking mode, and answer parsing are
all correct.

## Model notes

`Qwen3.5-9B` is a natively multimodal (early-fusion) model — there is no separate
`-VL` checkpoint. Architecture `Qwen3_5ForConditionalGeneration`: 32 layers with
**hybrid attention** (24 linear-attention / gated-delta-net layers + 8
full-attention layers), thinking mode on by default, 40,960-token max output.
Native dtype is bf16 (used here; running a bf16-trained model in fp16 risks
overflow, and bf16 is identical in memory/speed on Ada).

## Setup

```bash
python3.12 -m venv vllm-env && source vllm-env/bin/activate
uv pip install -U vllm --torch-backend=auto      # vLLM 0.22.1 (needs torch==2.11.0)
uv pip install -U transformers datasets pillow accelerate
```

vLLM 0.22.1 was built against `torch==2.11.0`; if a later torch gets pulled in,
pin it back (`uv pip install --torch-backend=auto torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0`)
or `vllm._C` fails to load with an ABI/undefined-symbol error.

## Reproduce

```bash
export HF_HOME=/path/to/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_mmmupro.py \
  --gpu-mem-util 0.95 --max-num-seqs 32 --max-num-batched-tokens 4096 \
  --max-model-len 45056 --max-pixels 1048576 --max-tokens 40960 \
  --chunk-size 500 \
  --out outputs/results.jsonl
```

Calibration: add `--limit 24`.

### Why these flags (24 GB fit)

bf16 weights are ~17.7 GiB on a 24 GiB card, leaving ~5.8 GiB for everything else.

- `--max-num-seqs 32` — caps CUDA-graph capture batch size (default captures up
  to 512, which OOMs during graph capture here).
- `--gpu-mem-util 0.95` — reclaims room for the KV cache (≈117k tokens; max
  concurrency ~2.6× at full 45k context, much higher at typical lengths).
- `--max-model-len 45056` — fits the full 40,960 output budget plus input.
- `--max-pixels 1048576` — caps image resolution so vision-token input stays
  small (prefill is a negligible fraction of cost here).
- `--chunk-size 500` — checkpoints results every 500 questions (crash safety +
  live ETA). Costs ~1 h overall vs. a single batch (per-chunk drain tails);
  set `--chunk-size 0` for a single un-chunked, slightly faster, uncheckpointed run.

## Performance

The cost is dominated by **thinking-token volume**: ~15.5M output tokens at a
compute-bound ~340 tok/s on one RTX 4090 (GPU pinned at 100%). The output-length
distribution is heavily right-skewed (median 4,093 but ~6% of questions run to
the full 40,960 cap), and those few long generations dominate wall-clock at the
tail of each batch.

To speed up without changing the spec (bf16 + thinking + 40,960): **data-parallel
sharding across N GPUs** is ~linear (4–8 GPUs → ~2–3 h); the model's MTP head
also enables speculative decoding (~1.5–2.5×, outputs ~unchanged). FP8 weights or
a lower token cap give ~1.5–2× but trade a little precision / change the spec.

## Files

- `run_mmmupro.py` — the pipeline (dataset load, multimodal prompt building with
  `<image N>` interleaving, vLLM batched generation, `Answer: X` parsing, scoring).
- `outputs/results_summary.json` — aggregate metrics.
- `outputs/results.jsonl.gz` — per-question rows: `id`, `pred`, `gold`, `correct`,
  `finish_reason`, `out_tokens`, `n_images`, `subject`, and the full generated
  `text` (including the `<think>…</think>` trace). Decompress with `gunzip -k`.
