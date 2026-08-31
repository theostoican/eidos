#!/bin/bash
# Kimi counterpart to the T=1.6 10% Qwen arm.
#
# MODEL. moonshotai/Kimi-VL-A3B-Thinking-2506 -- 16.4B total / ~3B active MoE, vision +
# thinking. The K2/K3 family is image-capable but trillion-scale and cannot run here; this
# is the only Kimi vision-thinking model in this hardware class, so it is the capability
# match for Qwen3.5-9B rather than a parameter-count match (16.4B total, but A3B active).
#
# SAME QUESTIONS. --sample-frac 0.10 --nest-from 0.05 --sample-seed default reproduces the
# IDENTICAL 172-question set the Qwen arm uses (test_Geography_252 skipped the same way),
# so the two models are compared question-for-question, not on two different samples.
#
# TENSOR PARALLEL, not data-parallel. Weights are 32.8GB bf16; on one 40GB card that leaves
# ~5GB for KV = ~3 concurrent 49k-token sequences, which would crawl. TP=2 puts 16.4GB per
# card and leaves ~21GB for KV. Kimi-VL uses MLA (kv_lora_rank 512), so KV is ~31KB/token
# against Qwen's ~128KB -- the win is concurrency, not cache size.
#
# TEMPERATURE IS MATCHED IN RELATIVE, NOT ABSOLUTE, TERMS. The Qwen arm ran T=1.6 against
# Qwen3.5-9B's recommended thinking temperature of 1.0 -- a 1.6x inflation. Kimi's own defaults
# are LOWER (generation_config.json temperature 0.6; the model card's vLLM example uses 0.8), so
# reusing T=1.6 would push Kimi 2.0-2.7x past its calibrated point while Qwen sat at 1.6x, and the
# two arms would not be measuring the same regime. T=1.3 is 1.6x the card's 0.8. Measured at
# T=1.6 Kimi also degenerated: ~30k-token traces saturated the KV cache at 18 concurrent
# sequences and the arm projected to ~86 h.
#
# kv-cache-dtype auto (bf16) matches the Qwen arm: the point is to change the MODEL only.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home

python cot_gen.py --model moonshotai/Kimi-VL-A3B-Thinking-2506 \
  --sampling-profile neutral \
  --temperature 1.25 --min-p 0 --repetition-penalty 1.0 \
  --sample-frac 0.10 --nest-from 0.05 --n-samples 16 \
  --top-ps "0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0" \
  --tensor-parallel-size 2 --num-shards 1 --shard-id 0 --resume --disable-async-scheduling \
  --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.85 \
  --max-num-batched-tokens 8192 --chunk-questions 8 \
  --out outputs/kimi_t125_10pct.jsonl --questions-out outputs/kimi_t125_10pct_q.json
