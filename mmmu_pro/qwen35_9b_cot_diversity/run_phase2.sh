#!/bin/bash
# HANDOFF Phase 2 -- GPU top-up of the NEUTRAL (v1) sweep. Usage: ./run_phase2.sh "0.6,0.8"
#
# Sampling is byte-identical to run_vpU.sh / HANDOFF 2.1: neutral profile (top_k=-1,
# presence_penalty=0) + min_p 0, repetition_penalty 1.0, temperature 1.0. Only TOPPS varies.
#
# Three jobs from HANDOFF 5, cheapest first (verified against the committed cots/):
#   top_p=0.8   29 of 86 questions missing ->   464 traces
#   top_p=0.6   57 of 86 questions missing ->   912 traces
#   top_p=0.98  86 of 86 questions missing -> 1,376 traces
#   top_p=0.99  86 of 86 questions missing -> 1,376 traces
# 0.1/0.2/0.3/0.4 are NOT generated -- excluded by HANDOFF 2.2.
#
# Writes to a NEW file, never into cots/u_gen.shard*.jsonl (HANDOFF 2.1.1(b)): the existing
# shards carry no config stamp, so contaminated rows appended there would be undetectable.
# --resume-glob reads the committed shards so already-complete cells are skipped, and the
# config guard in cot_gen.py aborts if any resumed file disagrees with this run's config.
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
mkdir -p logs outputs

TOPPS="${1:?usage: run_phase2.sh <comma-separated top_ps>}"
TAG="${2:-$(echo "$TOPPS" | tr ',.' '__')}"
GPUS=(0 1)
NSHARDS=${#GPUS[@]}
for s in "${!GPUS[@]}"; do
  g=${GPUS[$s]}
  CUDA_VISIBLE_DEVICES=$g setsid nohup python cot_gen.py \
    --sampling-profile neutral \
    --min-p 0 --repetition-penalty 1.0 --temperature 1.0 \
    --sample-frac 0.05 --n-samples 16 --top-ps "$TOPPS" \
    --num-shards "$NSHARDS" --shard-id "$s" \
    --resume --resume-glob "cots/u_gen.shard*.jsonl.gz" \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
    --max-num-batched-tokens 16384 --chunk-questions 8 \
    --out "outputs/vp2_${TAG}.jsonl" --questions-out outputs/vp2_q.json \
    > "logs/vp2_${TAG}.shard$s.log" 2>&1 < /dev/null &
done
echo "launched $NSHARDS shards on GPUs ${GPUS[*]} for top_p=$TOPPS (tag=$TAG)"
