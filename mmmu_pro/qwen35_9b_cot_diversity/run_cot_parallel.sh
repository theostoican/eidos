#!/bin/bash
# Parallel CoT generation + local Qwen judging on a 4-GPU node.
#   GPUs 0,1,2 -> cot_gen.py  (3 data-parallel shards, official MMMU-Pro prompt, --resume)
#   GPU 3      -> judge_qwen_cot.py --watch  (streams over generations as they land)
# The judge does NOT wait for generation to finish; it judges each new CoT as it appears
# and writes outputs/verdicts_cot.jsonl. When generation completes we touch the stop-file
# so the judge drains the tail and exits, then we merge shard outputs.
#
# Usage:  ./run_cot_parallel.sh 0.10     # 10% of MMMU-Pro standard(10 opt) test (=173 Q)
set -uo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate

FRAC="${1:-0.10}"; shift || true
NGEN=3                        # GPUs 0..2 generate; GPU 3 judges
STOP=outputs/.judge_stop
mkdir -p outputs logs
rm -f "$STOP"
echo "[node] generation on GPUs 0..$((NGEN-1)) | judge on GPU $NGEN | sample_frac=$FRAC | extra: $*"

# Pre-warm dataset + model into HF cache ONCE so the 4 engines don't race to download.
echo "[node] pre-warming caches ..."
python - <<'PY'
from datasets import load_dataset
load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-9B")
print("[node] caches warm")
PY

# --- launch the 3 generation shards ---
gpids=()
for i in $(seq 0 $((NGEN-1))); do
  CUDA_VISIBLE_DEVICES=$i python cot_gen.py \
    --sample-frac "$FRAC" --num-shards "$NGEN" --shard-id "$i" --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.92 \
    --out outputs/cot_gen.jsonl --questions-out outputs/questions.json \
    "$@" > "logs/shard${i}.log" 2>&1 &
  gpids+=($!)
  echo "[node] gen shard $i -> GPU $i (pid ${gpids[-1]}) -> logs/shard${i}.log"
done

# The judge needs questions.json (id -> ds_index). Each shard writes questions.shard{i}.json
# right after sampling, BEFORE the slow model load. Wait for all NGEN, merge, then start judge.
echo "[node] waiting for shard question maps to merge questions.json ..."
for _ in $(seq 1 120); do
  n=$(ls outputs/questions.shard*.json 2>/dev/null | wc -l)
  [ "$n" -ge "$NGEN" ] && break
  sleep 5
done
python - <<'PY'
import glob, json
q={}
for f in sorted(glob.glob("outputs/questions.shard*.json")):
    q.update(json.load(open(f)))
json.dump(q, open("outputs/questions.json","w"), indent=2)
print(f"[node] merged {len(q)} questions -> outputs/questions.json")
PY

# --- launch the streaming judge on GPU NGEN ---
CUDA_VISIBLE_DEVICES="$NGEN" python judge_qwen_cot.py --watch \
  --gen outputs/cot_gen.jsonl --questions outputs/questions.json \
  --verdicts outputs/verdicts_cot.jsonl \
  --gpu-mem-util 0.92 --batch 256 --poll 60 --stop-file "$STOP" \
  > logs/judge.log 2>&1 &
jpid=$!
echo "[node] judge (watch) -> GPU $NGEN (pid $jpid) -> logs/judge.log"

# --- wait for generation, then tell the judge to drain + stop ---
gfail=0
for p in "${gpids[@]}"; do wait "$p" || { echo "[node] gen pid $p FAILED"; gfail=1; }; done
cat outputs/cot_gen.shard*.jsonl > outputs/cot_gen.jsonl 2>/dev/null || true
echo "[node] generation done ($(wc -l < outputs/cot_gen.jsonl) gens) | gfail=$gfail | signalling judge to finish"
touch "$STOP"
wait "$jpid"

echo "[node] ALL DONE | gens=$(wc -l < outputs/cot_gen.jsonl) | verdicts=$(wc -l < outputs/verdicts_cot.jsonl)"
echo "[node] next: python extract_cot.py && python analyze_cot.py --verdicts outputs/verdicts_cot.jsonl"
exit $gfail
