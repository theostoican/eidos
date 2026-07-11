#!/bin/bash
# Generate on ALL 4 GPUs, THEN judge on all 4 GPUs (fastest to final results).
#   Phase 1: 4 data-parallel cot_gen.py shards (official MMMU-Pro prompt, --resume).
#   Phase 2: 4 judge_qwen_cot.py processes, one per shard gen file, on all 4 GPUs.
# Verdicts + gens are merged at the end. Both phases resume/skip completed work.
#
# Usage:  ./run_cot_all4.sh <sample_frac> <n_samples>   e.g.  ./run_cot_all4.sh 0.10 6
set -uo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate

FRAC="${1:-0.10}"; NSAMP="${2:-6}"; NGPU=4
mkdir -p outputs logs
echo "[node] sample_frac=$FRAC | n_samples=$NSAMP | $NGPU GPUs generate then judge"

# Pre-warm dataset + model into the HF cache ONCE.
python - <<'PY'
from datasets import load_dataset
load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B")
print("[node] caches warm")
PY

# ---------- Phase 1: generation on all 4 GPUs ----------
echo "[node] === PHASE 1: GENERATION ($NGPU shards) ==="
gpids=()
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i python cot_gen.py \
    --sample-frac "$FRAC" --n-samples "$NSAMP" --num-shards "$NGPU" --shard-id "$i" --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.92 \
    --out outputs/cot_gen.jsonl --questions-out outputs/questions.json \
    > "logs/shard${i}.log" 2>&1 &
  gpids+=($!)
  echo "[node] gen shard $i -> GPU $i (pid ${gpids[-1]}) -> logs/shard${i}.log"
done
gfail=0
for p in "${gpids[@]}"; do wait "$p" || { echo "[node] gen pid $p FAILED"; gfail=1; }; done

cat outputs/cot_gen.shard*.jsonl > outputs/cot_gen.jsonl 2>/dev/null || true
python - <<'PY'
import glob, json
q={}
for f in sorted(glob.glob("outputs/questions.shard*.json")):
    q.update(json.load(open(f)))
json.dump(q, open("outputs/questions.json","w"), indent=2)
print(f"[node] merged {len(q)} questions -> outputs/questions.json")
PY
echo "[node] generation done: $(wc -l < outputs/cot_gen.jsonl) gens | gfail=$gfail"

# ---------- Phase 2: judging on all 4 GPUs (one process per shard) ----------
echo "[node] === PHASE 2: JUDGING ($NGPU GPUs) ==="
jpids=()
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i python judge_qwen_cot.py \
    --gen "outputs/cot_gen.shard${i}.jsonl" --questions outputs/questions.json \
    --verdicts "outputs/verdicts_cot.shard${i}.jsonl" \
    --gpu-mem-util 0.92 --batch 256 \
    > "logs/judge${i}.log" 2>&1 &
  jpids+=($!)
  echo "[node] judge shard $i -> GPU $i (pid ${jpids[-1]}) -> logs/judge${i}.log"
done
jfail=0
for p in "${jpids[@]}"; do wait "$p" || { echo "[node] judge pid $p FAILED"; jfail=1; }; done
cat outputs/verdicts_cot.shard*.jsonl > outputs/verdicts_cot.jsonl 2>/dev/null || true

echo "[node] ALL DONE | gens=$(wc -l < outputs/cot_gen.jsonl) | verdicts=$(wc -l < outputs/verdicts_cot.jsonl) | gfail=$gfail jfail=$jfail"
echo "[node] next: python extract_cot.py && python analyze_cot.py --verdicts outputs/verdicts_cot.jsonl"
exit $((gfail+jfail))
