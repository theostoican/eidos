#!/bin/bash
# Kimi temperature pilot: pick T empirically instead of guessing, before committing ~70 GPU-hours.
#
# WHY. T=1.3 (1.6x Kimi's card-recommended 0.8, matching Qwen's 1.6x over its own 1.0) turned out
# to be too hot for this model: 50% of generations at top_p=1.0 ran to the 40960-token cap against
# Qwen's 0% at T=1.6, and accuracy was depressed even at top_p=0.1 where only 1.6% spoiled --
# so the deficit is degraded reasoning, not lost ballots.
#
# SELECTION RULE, stated before the data exists: take the LOWEST T whose spoil rate at top_p=1.0
# is comparable to Qwen's ~12.6% -- stressed but not degenerate. T must stay ABOVE 1.0: the
# mechanism under test needs an inflated tail, and at T<=1 truncating top_p can only sharpen, so
# no interior optimum can exist there by construction (README section 4). Sub-1.0 candidates are
# swept anyway to show where the boundary actually falls for this model.
#
# DESIGN. The same 8 questions at every temperature, so T is compared within-question. 1.3 is
# included so the arm already generated is on the same footing as the candidates.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home

python cot_gen.py --model moonshotai/Kimi-VL-A3B-Thinking-2506 \
  --sampling-profile neutral \
  --temps "0.9,1.0,1.1,1.2,1.3" --min-p 0 --repetition-penalty 1.0 \
  --sample-frac 0.10 --nest-from 0.05 --limit 8 --n-samples 8 \
  --top-ps "0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0" \
  --tensor-parallel-size 2 --num-shards 1 --shard-id 0 --resume --disable-async-scheduling \
  --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.85 \
  --max-num-batched-tokens 8192 --chunk-questions 2 \
  --out outputs/kimi_pilot.jsonl --questions-out outputs/kimi_pilot_q.json
