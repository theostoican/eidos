#!/bin/bash
# T=1.6 arm, 5% -> 10%: nested superset (the 86 committed questions are retained exactly,
# 87 new ones are topped up from the complement). Cell-level --resume across the committed
# cots/ means the 774 existing cells are skipped, so only the new questions are generated.
# Config must match the committed t16 sampling_cfg exactly or cot_gen.py's guard aborts:
# neutral profile (top_k=-1, presence_penalty=0), seed 1234, kv-cache-dtype auto.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
S=$1
CUDA_VISIBLE_DEVICES=$S python cot_gen.py --sampling-profile neutral \
  --temperature 1.6 --min-p 0 --repetition-penalty 1.0 \
  --sample-frac 0.10 --nest-from 0.05 --n-samples 16 \
  --top-ps "0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0" \
  --num-shards 2 --shard-id "$S" --resume --resume-glob "*/t16_*.jsonl*" \
  --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
  --max-num-batched-tokens 16384 --chunk-questions 8 \
  --out outputs/t16_10pct.jsonl --questions-out outputs/t16_10pct_q.json
