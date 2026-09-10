#!/bin/bash
# Smoke tests for the InternVL arm. Logs go to FILES, never through `| tail`: a pipeline's
# exit status is tail's, which turned an engine-init crash into "exit 0" once already.
#   ./smoke.sh gen     cot_gen.py end-to-end on GPU 0 (2 q x 2 samples x top_p {0.5,1.0})
#   ./smoke.sh prompt  render + inspect vLLM's actual prompts on GPU 1
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
# $HF_HOME/hub holds ghost blobs from a previous instance (stale handles, rm -rf
# cannot clear them). Model/dataset blobs go to a clean cache instead.
export HF_HUB_CACHE=/root/hf_hub PYTHONUNBUFFERED=1
[ -d /usr/local/cuda-13.0 ] && export CUDA_HOME=/usr/local/cuda-13.0 && export PATH=/usr/local/cuda-13.0/bin:$PATH
mkdir -p logs
case "${1:-gen}" in
  gen)
    CUDA_VISIBLE_DEVICES=0 python cot_gen.py --sampling-profile neutral --limit 2 --n-samples 2 \
      --top-ps 0.5,1.0 --max-tokens 4096 --chunk-questions 2 \
      --out outputs/smoke.jsonl --questions-out outputs/smoke_q.json > logs/smoke_gen.log 2>&1
    rc=$?; echo "smoke gen rc=$rc"; grep -E "^\[gen\]|^\[done\]|^\[skip\]|Error|RuntimeError" logs/smoke_gen.log | tail -6; exit $rc ;;
  prompt)
    CUDA_VISIBLE_DEVICES=1 python outputs/smoke_prompt.py > logs/smoke_prompt.log 2>&1
    rc=$?; echo "smoke prompt rc=$rc"; grep -E "^########|^--- OUTPUT|^PARSED|Error|RuntimeError" logs/smoke_prompt.log | tail -8; exit $rc ;;
esac
