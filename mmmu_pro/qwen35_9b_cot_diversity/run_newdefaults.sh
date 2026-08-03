#!/bin/bash
# Re-run of the top_p sweep under cot_gen.py's CURRENT ARGPARSE DEFAULTS
# ("qwen-recommended": top_k=20, presence_penalty=1.5, min_p=0, repetition_penalty=1.0, T=1.0).
#
# Why this run exists. The committed cots/ were generated with the v1 "neutral" sampling
# (top_k=-1, presence_penalty=0). That config produces repetition-loop truncation at a rate
# that varies ~90x across the sweep (9.8% at top_p=0.1 vs 0.1% at 1.0), and HANDOFF.md's
# audit showed the reported inverted-U is an artifact of dropping exactly those traces:
# re-analysed under the ballot rule the peak moves to the 1.0 edge and P(shape) falls from
# 0.76 to 0.11. The new defaults were added specifically to suppress that truncation, so
# this run tests the same hypothesis in the regime where the confound does not exist.
#
# NOTHING is passed on the sampling flags on purpose -- the defaults ARE the condition
# under test. The complete config is stamped into every output row (sampling_cfg /
# cfg_profile), so it is verifiable from the data rather than from this comment.
#
# Grid: the 7 fully-generated top_p of the v1 run, so the two runs are directly comparable.
# HANDOFF 2.2 excludes top_p<0.4 for the NEUTRAL config because that is where repetition
# collapse dominates; under these defaults that failure mode is what is being suppressed,
# and the low end is where the inverted-U's left arm has to live, so it is kept and the
# realised truncation rate is reported per top_p by majk.py.
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
mkdir -p logs outputs

TOPPS="0.1,0.3,0.5,0.7,0.9,0.95,1.0"
GPUS=(0 1)
NSHARDS=${#GPUS[@]}
for s in "${!GPUS[@]}"; do
  g=${GPUS[$s]}
  CUDA_VISIBLE_DEVICES=$g setsid nohup python cot_gen.py \
    --sample-frac 0.05 --n-samples 16 --top-ps "$TOPPS" \
    --num-shards "$NSHARDS" --shard-id "$s" --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
    --max-num-batched-tokens 16384 --chunk-questions 8 \
    --out outputs/nd_gen.jsonl --questions-out outputs/nd_q.json \
    > logs/nd_gen.shard$s.log 2>&1 < /dev/null &
done
echo "launched $NSHARDS shards on GPUs ${GPUS[*]}"
