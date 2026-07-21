#!/bin/bash
# Autonomous end-to-end driver for the inverted-U + premise-soundness experiment.
# IDEMPOTENT and RESUMABLE: safe to run repeatedly; each stage skips if already done.
# A cron watchdog relaunches this if the session/process dies. Original sampling
# (top_k=-1, pp=0) is used for generation -- that is what reproduces the inverted-U.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs
LOCK=outputs/.orchestrator.lock
DONE=outputs/FINAL_SUMMARY.md
log(){ echo "[orch $(date -u +%H:%M:%S)] $*" | tee -a logs/orchestrator.log; }

# ---- single-instance lock (stale-safe) ----
if [ -f "$LOCK" ]; then
  pid=$(cat "$LOCK" 2>/dev/null || echo)
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then log "another orchestrator ($pid) alive; exit"; exit 0; fi
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

[ -f "$DONE" ] && { log "FINAL_SUMMARY exists; nothing to do"; exit 0; }
source /venv/main/bin/activate

TOPPS="1.0,0.9,0.95,0.7,0.5,0.3,0.1,0.8,0.6,0.4,0.2"
NTOP=11; GPUS=(0 2 3); NSHARDS=3
JUDGE_SAMPLES=4

# ============ STAGE 1: GENERATION (original sampling) ============
gen_complete(){ python - <<'PY'
import collections,glob,json,sys
cnt=collections.Counter()
for f in glob.glob("outputs/u_gen.shard*.jsonl"):
    for l in open(f):
        l=l.strip()
        if l:
            try: cnt[json.loads(l)["top_p"]]+=1
            except: pass
full=sum(1 for p,c in cnt.items() if c>=86*16)  # 86 q x 16 samples per top_p
# accept a top_p as done when it has >=86 cells worth (>= 86*16*0.98 to tolerate a few parse gaps)
done=sum(1 for p,c in cnt.items() if c>=int(86*16*0.98))
sys.exit(0 if done>=11 else 1)
PY
}
if ! gen_complete; then
  for attempt in $(seq 1 40); do
    if pgrep -f "cot_gen.py .*u_gen" >/dev/null; then sleep 120; else
      log "generation not complete and no gen process -> (re)launch shards (attempt $attempt)"
      for s in "${!GPUS[@]}"; do g=${GPUS[$s]}
        CUDA_VISIBLE_DEVICES=$g nohup python cot_gen.py --sample-frac 0.05 --n-samples 16 \
          --top-ps "$TOPPS" --top-k -1 --presence-penalty 0 --min-p 0 --repetition-penalty 1.0 \
          --temperature 1.0 --num-shards $NSHARDS --shard-id $s --resume \
          --kv-cache-dtype auto --max-num-seqs 128 --gpu-mem-util 0.95 \
          --out outputs/u_gen.jsonl --questions-out outputs/u_q.json \
          >> logs/u_gen.shard$s.log 2>&1 &
      done
      sleep 180
    fi
    gen_complete && { log "GENERATION COMPLETE"; break; }
    # incremental maj-vote peek for the log
    python verify_u.py >> logs/verify_progress.log 2>&1 || true
  done
fi
gen_complete || { log "generation still incomplete after retries; will retry next watchdog tick"; exit 1; }

# free any lingering gen engines before GPU-heavy stages
pkill -f "cot_gen.py .*u_gen" 2>/dev/null; sleep 5
for p in $(pgrep -f "VLLM::EngineCore"); do kill -9 $p 2>/dev/null; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1>1000'|wc -l)" -eq 0 ]; do sleep 3; done

# ============ STAGE 2: extract CoT + answers ============
cat outputs/u_gen.shard*.jsonl > outputs/u_gen.jsonl
if [ ! -s outputs/u_extracted.jsonl ]; then
  log "extract_cot"; python extract_cot.py --gen outputs/u_gen.jsonl --out outputs/u_extracted.jsonl 2>&1 | tee -a logs/orchestrator.log
fi

# ============ STAGE 3: extract visual premises (Qwen, fixed prompt) ============
if [ ! -s outputs/u_premises.jsonl ]; then
  log "extract_premises (Qwen3.5-9B, strict visual-only prompt)"
  CUDA_VISIBLE_DEVICES=0 python extract_premises.py --extracted outputs/u_extracted.jsonl \
    --out outputs/u_premises.jsonl --judge-samples $JUDGE_SAMPLES \
    --gpu-mem-util 0.90 --max-num-seqs 128 --batch 512 2>&1 | tee -a logs/orchestrator.log
fi

# ============ STAGE 4: judge premises vs image (InternVL3-38B-AWQ, TP=2, no gold) ============
if [ ! -s outputs/u_verdicts.jsonl ]; then
  log "judge_premises (InternVL3-38B-AWQ TP=2, NO GOLD, math->UNVERIFIABLE)"
  CUDA_VISIBLE_DEVICES=0,2 python judge_premises_internvl.py --premises outputs/u_premises.jsonl \
    --questions outputs/u_q.json --out outputs/u_verdicts.jsonl \
    --tensor-parallel-size 2 --gpu-mem-util 0.90 --max-num-seqs 16 \
    --max-model-len 16384 --batch 96 2>&1 | tee -a logs/orchestrator.log
fi

# ============ STAGE 5: analyze + plot ============
log "analyze_premises + majority/diversity"
python analyze_premises.py --extracted outputs/u_extracted.jsonl --premises outputs/u_premises.jsonl \
  --verdicts outputs/u_verdicts.jsonl --out outputs/u_report.json 2>&1 | tee -a logs/orchestrator.log
python analyze_cot.py --extracted outputs/u_extracted.jsonl --out outputs/u_cot_report.json 2>&1 | tee -a logs/orchestrator.log
python plot_premises.py outputs/u_report.json --out outputs/u_premise_summary.png \
  --note "5% MMMU-Pro, 16 samples, original sampling (top_k=-1)" 2>&1 | tee -a logs/orchestrator.log

# ============ STAGE 6: verify + summary ============
log "writing FINAL_SUMMARY.md"
python make_summary.py > "$DONE" 2>> logs/orchestrator.log
python final_chart.py 2>&1 | tee -a logs/orchestrator.log
log "ALL DONE -> $DONE"
