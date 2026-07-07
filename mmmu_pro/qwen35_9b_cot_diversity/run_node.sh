#!/bin/bash
# Data-parallel CoT generation across ALL GPUs on a multi-GPU node.
# One independent vLLM engine per GPU (better throughput than TP for a 9B model), each
# taking a strided slice of the questions (cot_gen.py --num-shards/--shard-id), with
# --resume so re-launching continues where a preempted run left off.
#
# Usage:
#   ./run_node.sh 1.0                 # FULL MMMU-Pro (1730 Q)  <-- the confirmed target
#   ./run_node.sh 0.05                # 5% sample
#   ./run_node.sh 1.0 --top-ps 0.9    # + any extra cot_gen.py args
#
# Assumes bf16 weights fit per GPU with room for KV (>=48GB cards -> --kv-cache-dtype auto).
# For <48GB GPUs, pass --kv-cache-dtype fp8 as an extra arg.
set -uo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate

FRAC="${1:-1.0}"; shift || true
NGPU=$(nvidia-smi -L | wc -l)
mkdir -p outputs logs
echo "[node] $NGPU GPUs | sample_frac=$FRAC | extra args: $* "

# Pre-warm caches ONCE so the N engines don't race on first download.
echo "[node] pre-warming dataset + model into HF cache ..."
python - <<'PY'
from datasets import load_dataset
load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B")
print("[node] caches warm")
PY

pids=()
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i python cot_gen.py \
    --sample-frac "$FRAC" --num-shards "$NGPU" --shard-id "$i" --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.92 \
    --out outputs/cot_gen.jsonl --questions-out outputs/questions.json \
    "$@" > "logs/shard${i}.log" 2>&1 &
  pids+=($!)
  echo "[node] launched shard $i on GPU $i (pid ${pids[$((${#pids[@]}-1))]}) -> logs/shard${i}.log"
done

echo "[node] waiting on ${#pids[@]} shards (tail -f logs/shard0.log to watch) ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || { echo "[node] shard pid $p FAILED"; fail=1; }; done

# Merge shard outputs + question maps.
cat outputs/cot_gen.shard*.jsonl > outputs/cot_gen.jsonl 2>/dev/null || true
python - <<'PY'
import glob, json
q={}
for f in glob.glob("outputs/questions.shard*.json"):
    q.update(json.load(open(f)))
json.dump(q, open("outputs/questions.json","w"), indent=2)
print(f"[node] merged {len(q)} questions -> outputs/questions.json")
PY
echo "[node] merged $(wc -l < outputs/cot_gen.jsonl) generations -> outputs/cot_gen.jsonl | fail=$fail"
echo "[node] next: python extract_cot.py && python analyze_cot.py  (see HANDOFF.md §4-5)"
exit $fail
