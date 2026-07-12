#!/bin/bash
# Temperature x top_p sweep, GENERATION ONLY (no judging: the per-sample-accuracy prune
# decision is judge-independent). 4 data-parallel shards. top_p-outer / temp-inner, with
# top_p=0.95 (recommended) listed FIRST so its full temperature sweep completes before the
# second top_p arm -> lets us prune early on the primary hump test.
#   ./run_temp_sweep.sh <sample_frac> <n_samples>
set -uo pipefail
cd "$(dirname "$0")"; source /venv/main/bin/activate
FRAC="${1:-0.05}"; NSAMP="${2:-8}"; NGPU=4
# FULL top_p sweep at each temperature -> the inverted-U is checked WITH RESPECT TO p.
# temp-outer generation order (in cot_gen.py): temps listed first complete their full p-sweep
# first. 1.3 (high, where top_p has the most leverage) first, then 0.7, then 1.0.
TOPPS="0.5,0.7,0.9,0.95,1.0"; TEMPS="1.3,0.7,1.0"
mkdir -p outputs logs
echo "[node] TEMP SWEEP | frac=$FRAC n=$NSAMP | top_ps=$TOPPS temps=$TEMPS | $NGPU shards"
python - <<'PY'
from datasets import load_dataset
load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B"); print("[node] caches warm")
PY
gpids=()
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i python cot_gen.py \
    --sample-frac "$FRAC" --n-samples "$NSAMP" --top-ps "$TOPPS" --temps "$TEMPS" \
    --num-shards "$NGPU" --shard-id "$i" --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.92 \
    --out outputs/cot_gen.jsonl --questions-out outputs/questions.json \
    > "logs/shard${i}.log" 2>&1 &
  gpids+=($!)
  echo "[node] shard $i -> GPU $i (pid ${gpids[-1]}) -> logs/shard${i}.log"
done
gfail=0
for p in "${gpids[@]}"; do wait "$p" || { echo "[node] shard $p FAILED"; gfail=1; }; done
cat outputs/cot_gen.shard*.jsonl > outputs/cot_gen.jsonl 2>/dev/null || true
python - <<'PY'
import glob, json
q={}
for f in sorted(glob.glob("outputs/questions.shard*.json")): q.update(json.load(open(f)))
json.dump(q, open("outputs/questions.json","w"), indent=2); print(f"[node] merged {len(q)} questions")
PY
echo "[node] TEMP SWEEP GEN DONE | gens=$(wc -l < outputs/cot_gen.jsonl) | gfail=$gfail"
exit $gfail
