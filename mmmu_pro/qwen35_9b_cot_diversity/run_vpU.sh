#!/bin/bash
# Inverted-U reproduction run. ORIGINAL sampling that produced the effect (commit 2219712):
#   top_k=-1 (OFF), presence_penalty=0, min_p=0, repetition_penalty=1.0, temp=1.0, top_p swept.
# The HEAD defaults (top_k=20, presence_penalty=1.5) were added later to suppress truncation
# and DESTROY the inverted-U by keeping high-top_p sampling coherent. Do NOT use them here.
# GPU 1 excluded (dead ECC). Grid front-loads the prior {1.0,0.9,0.95,0.7,0.5} for fast verify,
# then fills the rest of the 0.1-step sweep (+0.95 to exactly match prior grid).
cd "$(dirname "$0")"
source /venv/main/bin/activate
TOPPS="1.0,0.9,0.95,0.7,0.5,0.3,0.1,0.8,0.6,0.4,0.2"
GPUS=(0 2 3)
NSHARDS=${#GPUS[@]}
for s in "${!GPUS[@]}"; do
  g=${GPUS[$s]}
  CUDA_VISIBLE_DEVICES=$g python cot_gen.py \
    --sample-frac 0.05 --n-samples 16 --top-ps "$TOPPS" \
    --top-k -1 --presence-penalty 0 --min-p 0 --repetition-penalty 1.0 --temperature 1.0 \
    --num-shards "$NSHARDS" --shard-id "$s" --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
    --out outputs/u_gen.jsonl --questions-out outputs/u_q.json \
    > logs/u_gen.shard$s.log 2>&1 &
done
wait; echo "ALL_SHARDS_DONE"
