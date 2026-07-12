#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")"; source /venv/main/bin/activate
jpids=()
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i python judge_qwen_cot.py --no-gold \
    --gen "outputs/cot_gen.shard${i}.jsonl" --questions outputs/questions.json \
    --verdicts "outputs/verdicts_cot_nogold.shard${i}.jsonl" \
    --gpu-mem-util 0.92 --batch 256 > "logs/judge_nogold${i}.log" 2>&1 &
  jpids+=($!)
done
for p in "${jpids[@]}"; do wait "$p" || echo "judge pid $p FAILED"; done
cat outputs/verdicts_cot_nogold.shard*.jsonl > outputs/verdicts_cot_nogold.jsonl
echo "[rejudge] DONE: $(wc -l < outputs/verdicts_cot_nogold.jsonl) no-gold verdicts"
