#!/bin/bash
# InternVL3.5-8B top_p sweep -- 4 data-parallel shards, one per GPU.
# The arm is selected by TEMP/TAG/GRID (see arm.sh); this directory commits the T=1.2 arm.
# Every flag that decides the result is stated here explicitly (no argparse defaults
# are relied on) and is stamped into every row by cot_gen.py.
#
#   ./run_sweep.sh            launch all 4 shards (detached, restart-on-crash, --resume)
#   ./run_sweep.sh status     progress per shard
#   ./run_sweep.sh stop       kill all shards + watchdogs
set -u
cd "$(dirname "$0")"
export HF_HOME=/workspace/.hf_home
# $HF_HOME/hub holds ghost blobs from a previous instance (stale handles, rm -rf
# cannot clear them). Model/dataset blobs go to a clean cache instead.
export HF_HUB_CACHE=/root/hf_hub
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# max_model_len is 40960, the LLM's declared context. Running 49152 (the Qwen arm's value) under
# VLLM_ALLOW_LONG_MAX_MODEL_LEN was tried: the rotary table is sized to 40960, and the engine dies
# with a device-side assert the moment any trace crosses it -- i.e. on every capped trace.
# FlashInfer JIT-compiles with the nvcc on PATH; the system CUDA 12.8 predates sm120 (RTX 5090)
# and fails with a misleading 'requires sm75' error. Point it at the CUDA 13.0 toolkit.
[ -d /usr/local/cuda-13.0 ] && export CUDA_HOME=/usr/local/cuda-13.0 && export PATH=/usr/local/cuda-13.0/bin:$PATH
source /venv/main/bin/activate >/dev/null 2>&1   # silent: the args action must print ONLY the array
. ./arm.sh                       # TEMP, TAG, OUTP, GENLOG, PIDF -- never share files between arms
TP="${TP:-1}"                    # tensor-parallel size per engine. TP shards the weights AND the
                                 # KV across GPUs, so KV/GPU grows ~linearly with TP -- the only
                                 # lever that raises concurrency without touching the numerics
                                 # (TP is exact up to reduction order). These cards have no P2P,
                                 # so NCCL stages through host memory: measure before trusting it.
NGPU="${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NSHARDS="${NSHARDS:-$((NGPU / TP))}"          # NSHARDS=8 SHARD_IDS="2 6" -> sub-shards: slice i of 4 == slices i,i+4 of 8
SHARD_IDS="${SHARD_IDS:-$(seq 0 $((NSHARDS-1)))}"   # SHARD_IDS="0 1 2" ./run_sweep.sh -> launch a subset
# 0.92 (the 32GB value) overshoots on 24GB: attempts 1-6 and 7-20 on 2026-09-09 all died with
# "tried to allocate 76 MiB, 69 MiB free" during warmup/graph capture. 0.88 leaves headroom.
MEMUTIL="${MEMUTIL:-0.88}"
CHUNK="${CHUNK:-16}"   # questions per llm.chat() call = the checkpoint interval:
                       # NOTHING is written until a chunk finishes, so a crash loses up to one.
KV="${KV:-auto}"        # KV=fp8 ./run_sweep.sh  -> fp8 KV cache (2x tokens in flight); default bf16
OUT=$OUTP.jsonl
MAX_RETRIES=20

gen_args=(
  --sampling-profile neutral
  --model OpenGVLab/InternVL3_5-8B
  --temperature "$TEMP" --top-k -1 --presence-penalty 0 --min-p 0 --repetition-penalty 1.0
  --top-ps "$GRID"
  --n-samples $NSAMPLES
  --sample-frac $SAMPLE_FRAC --nest-from 0.05 --sample-seed 20260706
  --max-tokens 40960 --max-model-len 40960
  --kv-cache-dtype "$KV" --gpu-mem-util $MEMUTIL --max-num-seqs 128 --max-num-batched-tokens 8192
  --chunk-questions $CHUNK --seed 1234        # was 8: fewer chunk-drain tails on relaunch (each tail = a lone capped trace at ~50 tok/s)
  --tensor-parallel-size $TP
  --num-shards $NSHARDS
  --resume --resume-glob "$OUTP.shard*.jsonl"
  --out "$OUT" --questions-out ${OUTP}_q.json
  ${EXTRA_ARGS:-}                      # e.g. EXTRA_ARGS="--spec-ngram 5" (engine-only knobs; see cot_gen.py)
)

case "${1:-run}" in
  args)   # print the generation argument list (shell-quoted) and exit. NEVER `source` this script:
          # its default action is `run`, and sourcing it launches every shard (2026-09-07 incident).
    echo "${gen_args[*]@Q}"; exit 0 ;;
  status)
    for i in $(seq 0 $((NSHARDS-1))); do
      f=$OUTP.shard$i.jsonl
      n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
      last=$(awk '/=== shard [0-9] attempt/{buf=""} /^\[gen\] chunk|^\[done\]|^\[abort\]|Error|=== shard [0-9] (DONE|GAVE UP)/{buf=$0} END{print buf}' $GENLOG$i.log 2>/dev/null | cut -c1-150)
      pid=$(cat $PIDF$i.pid 2>/dev/null); alive=$( [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && echo alive || echo dead )
      printf "shard %d [%s] rows=%-6s %s\n" "$i" "$alive" "$n" "$last"
    done
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null
    ;;
  stop)
    for i in $(seq 0 $((NSHARDS-1))); do
      pid=$(cat $PIDF$i.pid 2>/dev/null) && [ -n "$pid" ] && pkill -TERM -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null
    done
    sleep 3
    # EngineCore workers are separate processes, reparent to init, and rename themselves to
    # VLLM::EngineCore / VLLM::Worker_TP<n>, so neither the pid file nor a cot_gen.py pattern
    # finds them. nvidia-smi is NO USE here: inside a container it reports HOST pids, which do
    # not exist in this pid namespace (observed 2026-09-09 -- `kill` said "No such process"
    # while the GPUs stayed pinned at 100%). Match on the process name instead.
    # match on the process NAME (comm), not the command line: `pkill -f "VLLM::"` also matches
    # any shell whose argv happens to contain the string, including the one running this script.
    for p in $(ps -eo pid,comm | awk '$2 ~ /^VLLM::/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
    sleep 5
    left=$(ps -eo pid,comm | awk '$2 ~ /^VLLM::/' | wc -l)
    [ "$left" = 0 ] || echo "WARNING: $left VLLM process(es) still holding GPUs"
    echo stopped
    ;;
  run)
    mkdir -p logs outputs
    # Preflight. One 40960-token trace needs 5.6 GiB of bf16 KV (36 layers x 8 KV heads x 128 dim
    # x 2 for K+V x 2 B); weights are ~16.4 GiB. Capped traces monopolise the cache regardless of
    # its size, so throughput ~ (KV capacity / 40k) x per-sequence speed. Measured on a 5090 32GB:
    # 85.5k KV tokens = 2.1 traces = 110-725 tok/s. Below ~1.5 the engine runs one sequence at a
    # time and long-context decode collapses, so this refuses rather than burning days on it.
    tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$tot" ]; then
      # weights/GPU = (15.9-0.6 vision)/TP + 0.6 ; KV/GPU per trace = 5.62/TP (fp8: /2 again)
      kvdiv=$( [ "$KV" = fp8 ] && echo 2 || echo 1 )
      tr=$(awk -v t="$tot" -v tp="$TP" -v kd="$kvdiv" \
           'BEGIN{w=(15.89-0.6)/tp+0.6; printf "%.2f", (t/1024*0.92-w)/(5.62/tp/kd)}')
      agg=$(awk -v x="$tr" -v n="$NSHARDS" 'BEGIN{printf "%.1f", x*n}')
      echo "[preflight] ${tot} MiB/GPU, TP=$TP, KV=$KV -> ${tr} full-length traces/engine, ${agg} aggregate"
      if awk -v x="$tr" 'BEGIN{exit !(x<1.5)}'; then
        echo "[preflight] ABORT: only ${tr} full-length traces fit per GPU (need >=1.5)."
        echo "            This arm's cap is fixed at 40960 (the model's rotary limit) and the KV"
        echo "            dtype is bf16 to match the other arms, so neither can be traded away."
        echo "            Raise TP (TP=2 or TP=4 shards weights AND KV, exact numerics), or"
        echo "            KV=fp8 (2x, but perturbs the tail this sweep measures), or use >=32 GB"
        echo "            cards. PREFLIGHT=0 overrides and accepts ~1 sequence per engine."
        [ "${PREFLIGHT:-1}" = 0 ] || exit 3
      fi
    fi
    for f in $OUTP.shard*.jsonl; do
      [ -s "$f" ] || continue
      stamped=$(head -1 "$f" | grep -o '"kv_cache_dtype": "[a-z0-9]*"' | cut -d'"' -f4)
      if [ -n "$stamped" ] && [ "$stamped" != "$KV" ]; then
        echo "ABORT: $f holds rows stamped kv_cache_dtype=$stamped but KV=$KV was requested."
        echo "       cot_gen.py would refuse to resume into it; move/rename the file or match KV."; exit 2
      fi
    done
    if [ "$TP" -gt 1 ]; then
      # No P2P on consumer cards: keep NCCL off the peer path and off vLLM's custom all-reduce,
      # both of which hang rather than fall back on this topology.
      export NCCL_P2P_DISABLE=1 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
    fi
    g=0
    for i in $SHARD_IDS; do
      dev=$(seq -s, $((g * TP)) $((g * TP + TP - 1)))   # by position, not by shard id
      # watchdog: relaunch on any non-zero exit; --resume makes a restart cost one engine load
      setsid bash -c "
        for try in \$(seq 1 $MAX_RETRIES); do
          echo \"=== shard $i attempt \$try \$(date -u +%FT%TZ) ===\"
          CUDA_VISIBLE_DEVICES=$dev python cot_gen.py ${gen_args[*]@Q} --shard-id $i && { echo '=== shard $i DONE ==='; exit 0; }
          echo \"=== shard $i exited rc=\$? ; restarting in 30s ===\"; sleep 30
        done
        echo '=== shard $i GAVE UP ==='; exit 1
      " >> $GENLOG$i.log 2>&1 < /dev/null &
      echo $! > $PIDF$i.pid
      g=$((g + 1))
      sleep 2
    done
    echo "launched $NSHARDS shards (T=$TEMP tag=$TAG); $GENLOG*.log ; ./run_sweep.sh status"
    ;;
esac
