#!/bin/bash
# One vLLM engine pinned to one GPU:  serve.sh <gpu> <port>
# Sampling-relevant flags mirror cot_gen.py's LLM(...) exactly (bf16 weights, bf16 KV, 49152
# context, 8 images/prompt, seed 1234). Throughput knobs come from the environment:
#   MAX_NUM_SEQS, MAX_BATCHED, GPU_UTIL, CG_MAX (max CUDA-graph batch),
#   SPEC (a --speculative-config JSON, empty = off), PROFILE_DIR (enables POST /start_profile),
#   ATTN_BACKEND (e.g. FLASHINFER; default = vLLM's pick, FLASH_ATTN on H100),
#   COMPILATION_CFG (a --compilation-config JSON, e.g. cudagraph_mode).
# Prefix caching is OFF. For this hybrid (Gated-DeltaNet) model it forces
# mamba_cache_mode=align, which keeps a recurrent-state snapshot per 528-token block in every
# DeltaNet layer group; measured, that capped an 80GB engine at ~60-100 live sequences. The
# only thing it saves here is re-prefilling a ~1-3k-token prompt per sample.
GPU=$1; PORT=$2
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
source /venv/vllm/bin/activate
ARGS=(Qwen/Qwen3.5-9B --host 127.0.0.1 --port "$PORT"
      --dtype bfloat16 --kv-cache-dtype "${KV_DTYPE:-auto}" --max-model-len 49152
      --gpu-memory-utilization "${GPU_UTIL:-0.95}"
      --max-num-seqs "${MAX_NUM_SEQS:-512}" --max-num-batched-tokens "${MAX_BATCHED:-16384}"
      --limit-mm-per-prompt '{"image": 8, "video": 0}' --trust-remote-code
      --no-enable-prefix-caching --seed 1234 --disable-uvicorn-access-log)
[ -n "$CG_MAX" ] && ARGS+=(--max-cudagraph-capture-size "$CG_MAX")
[ -n "$SPEC" ] && ARGS+=(--speculative-config "$SPEC")
[ -n "$ATTN_BACKEND" ] && ARGS+=(--attention-backend "$ATTN_BACKEND")
[ -n "$COMPILATION_CFG" ] && ARGS+=(--compilation-config "$COMPILATION_CFG")
[ -n "$PROFILE_DIR" ] && ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\"}")
exec vllm serve "${ARGS[@]}"
