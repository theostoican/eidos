#!/bin/bash
# analyze.py on the T=1.2 / top_k=-1 / presence 1.5 traces (t12_sweep.jsonl + the 0.95 point in
# t12_penalty.jsonl), then the figure (card reference computed inside t12_chart.py). CPU only; rerunnable.
cd "$(dirname "$0")"; source /venv/vllm/bin/activate
GRID=${GRID:-0.5,0.6,0.7,0.75,0.8,0.85,0.95}
GLOB=${GLOB:-cots/t12_*.jsonl.gz}      # the committed shards; GLOB="outputs/t12_[sp]*.jsonl" for a live run
CARD=${CARD:-cots/card_t10.shard*.jsonl.gz}
python ../common/analyze.py --glob "$GLOB" --temperature 1.2 --grid "$GRID" --ks 1,8 \
  --out outputs/RESULT_T12_SWEEP_FULL.md > logs/analyze_t12.log 2>&1 || { tail -n 5 logs/analyze_t12.log; exit 1; }
python t12_chart.py --card "$CARD" "$@"
