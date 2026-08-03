#!/bin/bash
# Runs to the end of the plan with no human step in between, because a manual handoff
# between two GPU jobs already cost ~10.75h of idle A100 time once.
#
#   1. wait for the Phase-2b (top_p 0.98/0.99) shards to finish
#   2. run the final Phase-1 analysis on the full 9-point grid at T=1.0
#   3. if the inverted-U is DEAD by the pre-registered criterion, launch the T=1.6 sweep
#
# NOTHING generated so far is touched. cots/ is read-only throughout; vp2_grid7.* and
# vp2_dense.* stay as written; the T=1.6 arm goes to its own outputs/t16_*.jsonl. The only
# new files are outputs/PHASE1_FINAL.{md,json} and the t16 arm.
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=${HF_HOME:-/workspace/.hf_home}

# ---- 1. wait for generation ------------------------------------------------
while :; do
  d=$(grep -l "^\[done\]" logs/vp2_dense.shard0.log logs/vp2_dense.shard1.log 2>/dev/null | wc -l)
  [ "$d" -ge 2 ] && break
  if [ "$(ps -eo args | grep -c '[c]ot_gen.py')" -eq 0 ]; then
    echo "GENERATION_DIED_EARLY $(date -u +%H:%M:%S)"; break
  fi
  sleep 60
done
echo "GENERATION_DONE $(date -u +%H:%M:%S)"

# ---- 2. final analysis on the 9-point grid ---------------------------------
python phase1_analysis.py \
  --extra-glob "outputs/vp2_*.shard*.jsonl" \
  --grid 0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.0 \
  --shape-boot 20000 --out outputs/PHASE1_FINAL.md > logs/phase1_final.log 2>&1
echo "ANALYSIS_DONE $(date -u +%H:%M:%S)"

# ---- 3. is the inverted-U dead? --------------------------------------------
# Pre-registered (HANDOFF 4): an interior peak is claimed only if P(shape) >= 0.95. We use
# the joint criterion (two-lines AND down-opening quadratic in the same bootstrap replicate).
# ALIVE at any k -> do NOT start the T-sweep; the T=1.0 arm answered the question already.
VERDICT=$(python - <<'PY'
import json
try:
    d = json.load(open("outputs/PHASE1_FINAL.json"))
    rows = d["spoiled_full"]
    best = max(r["p_joint"] for r in rows)
    print("ALIVE" if best >= 0.95 else "DEAD", f"{best:.3f}")
except Exception as e:
    print("ERROR", e)
PY
)
echo "VERDICT $VERDICT"
case "$VERDICT" in
  DEAD*) ;;
  *) echo "NOT_LAUNCHING_T16 ($VERDICT)"; exit 0 ;;
esac

# ---- 4. T=1.6 sweep --------------------------------------------------------
# Same 86 questions, same 16 samples, same neutral sampling profile, same seed. TEMPERATURE
# IS THE ONLY DIFFERENCE from the T=1.0 arm, so the two are directly comparable.
# Why this is the test that can actually produce curvature (HANDOFF 8): at T<=1, top_p can
# only REMOVE probability mass, so the axis runs from near-greedy to the model's true
# distribution and stops -- there is no over-diversified regime for the right arm to fall
# from. At T=1.6 the tail is inflated, so top_p truncation is doing real work and the
# falling arm becomes reachable. Prediction: curvature absent at T=1.0, present at T=1.6.
# Grid is HANDOFF 2.2's primary 5 points; chunk-by-question means any interrupted prefix is
# still a complete balanced dataset.
GPUS=(0 1)
for s in "${!GPUS[@]}"; do
  g=${GPUS[$s]}
  CUDA_VISIBLE_DEVICES=$g setsid nohup python cot_gen.py \
    --sampling-profile neutral \
    --min-p 0 --repetition-penalty 1.0 --temperature 1.6 \
    --sample-frac 0.05 --n-samples 16 --top-ps "0.5,0.7,0.9,0.95,1.0" \
    --num-shards 2 --shard-id "$s" --resume \
    --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
    --max-num-batched-tokens 16384 --chunk-questions 6 \
    --out outputs/t16_gen.jsonl --questions-out outputs/t16_q.json \
    > "logs/t16_gen.shard$s.log" 2>&1 < /dev/null &
done
echo "T16_LAUNCHED $(date -u +%H:%M:%S)"
