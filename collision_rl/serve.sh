#!/bin/bash
# Start or stop the vLLM server on 127.0.0.1:${PORT:-8100}.   ./serve.sh train | eval | stop
#   train: 40% of GPU memory, leaving the rest to train.py on the same GPU;  eval: 90%.
#   Model: $VLM_MODEL (default Qwen/Qwen3-VL-2B-Instruct). Runtime LoRA loading is on, for the adapters train.py writes.
cd "$(dirname "$0")"; mkdir -p logs
PORT=${PORT:-8100}; MODEL=${VLM_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
stop() { P=$(cat logs/serve.pid 2>/dev/null); [ -n "$P" ] && kill "$P" 2>/dev/null
         for i in $(seq 1 60); do [ -n "$P" ] && kill -0 "$P" 2>/dev/null || break; sleep 2; done; rm -f logs/serve.pid; }
stop; [ "$1" = stop ] && exit 0
case "$1" in train) UTIL=0.40 ;; eval) UTIL=0.90 ;; *) echo "usage: $0 train|eval|stop"; exit 2 ;; esac
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
setsid vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --kv-cache-dtype fp8 --max-model-len 8192 \
  --gpu-memory-utilization $UTIL --max-num-seqs 192 --max-num-batched-tokens 16384 --limit-mm-per-prompt '{"image": 8, "video": 0}' \
  --trust-remote-code --enable-prefix-caching --seed 1234 --disable-uvicorn-access-log --enable-lora --max-lora-rank 16 --max-loras 2 \
  > "logs/serve_$1.log" 2>&1 < /dev/null &
echo $! > logs/serve.pid
until curl -sf "localhost:$PORT/v1/models" >/dev/null || ! kill -0 "$(cat logs/serve.pid)" 2>/dev/null; do sleep 5; done
curl -sf "localhost:$PORT/v1/models" >/dev/null && echo "server up ($1, $MODEL)" || { echo "server failed to start:"; tail -20 "logs/serve_$1.log"; exit 1; }
