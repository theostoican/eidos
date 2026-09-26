#!/bin/bash
# Sequential arms toward ">= +1 pp over the card". Each arm: 8 samples on ALL 1,729 questions
# (the 322 outside the frozen 1,407 form a holdout for confirmation); resume-safe; the
# comparison is rebuilt after every arm. Order = expected value per GPU-hour.
cd "$(dirname "$0")"
source /venv/vllm/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.hf_home} HF_HUB_OFFLINE=1
cfg() { ( set -a; . "./serve_gpu$1.env"
    python -c 'import json,os; s=os.environ.get("SPEC") or None
print(json.dumps({"vllm":"0.26.0+cu129","gpu":"H100-80GB","tp":1,"prefix_caching":False,
 "kv_cache_dtype":os.environ["KV_DTYPE"],"spec_decode":json.loads(s) if s else None}))' ); }
arm() {  # arm <profile> <T> <top_p> <out-stem> [extra client args...]
  local prof=$1 T=$2 p=$3 stem=$4; shift 4
  echo "[chain] arm profile=$prof T=$T top_p=$p extra='$*' -> outputs/$stem.jsonl  start $(date -u)"
  python -u cot_gen_stream.py \
    --servers http://127.0.0.1:8100,http://127.0.0.1:8101 \
    --engine-cfg "[$(cfg 0), $(cfg 1)]" --engine-env serve_gpu0.env,serve_gpu1.env \
    --sampling-profile "$prof" --temperature "$T" --top-ps "$p" --n-samples 8 --sample-frac 1.0 "$@" \
    --kv-cache-dtype fp8 --no-request-seed --inflight 256 --step 8 --kv-high 0.95 --preempt-tol 2 \
    --out "outputs/$stem.jsonl" --questions-out "outputs/${stem}_q.json" --metrics-out "logs/${stem}_metrics.jsonl"
  echo "[chain] arm $stem done $(date -u)"
  python compare_all.py | tee outputs/RESULT_ALL_VS_CARD.md
  python compare_all.py --holdout | tee outputs/RESULT_HOLDOUT_VS_CARD.md
  python mix_ballots.py | tee outputs/RESULT_MIXED_BALLOTS.md
}
# arms are read one line at a time from chain_arms.txt (profile T top_p out-stem), re-read
# before each arm, so the queue can be edited while the chain runs. Lines starting with # skip.
i=0
while :; do
  line=$(grep -v '^#' chain_arms.txt | sed -n "$((i+1))p"); [ -z "$line" ] && break
  arm $line; i=$((i+1))
done
echo "[chain] all arms done $(date -u)"
