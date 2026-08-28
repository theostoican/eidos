#!/bin/bash
# Visual-premise arm, one SAMPLE LAYER at a time: extract -> judge -> chart.
#
# Layer-at-a-time because both passes need the whole 24GB card, and because a finished layer
# is a COMPLETE, BALANCED dataset over the full 345 x 10 grid: stop after any layer and the
# curve is still computed on the same cells at every top_p.
#
# Both passes resume from their own output, so re-running is safe: finished work is skipped.
set -u
cd "$(dirname "$0")"
source /venv/main/bin/activate

# printf, not `seq -w`: seq pads to the width of its LARGEST argument, so `seq -w 0 15` gives
# 00..15 while `seq -w 0 7` gives 0..7 -- every filename then misses and the loop silently
# completes having done nothing.
LAST=${VP_LAST_LAYER:-7}
for i in $(seq 0 "$LAST"); do
  S=$(printf "%02d" "$i")
  L="outputs/vp_layers/layer${S}.jsonl.gz"
  if [ ! -f "$L" ]; then echo "MISSING $L" | tee -a outputs/VP_FAILURES.txt; continue; fi
  # A stage that dies must not idle the GPU until someone notices: retry once (both stages
  # resume), then move on and record it. An empty VP_FAILURES.txt means a clean run.
  echo "=== layer ${S}: extract  $(date -u +%FT%TZ)"
  python vp_extract.py --layers "$L" --batch 256 \
    || { echo "RETRY extract ${S}"; python vp_extract.py --layers "$L" --batch 64; } \
    || { echo "FAILED extract ${S}" | tee -a outputs/VP_FAILURES.txt; continue; }
  echo "=== layer ${S}: judge    $(date -u +%FT%TZ)"
  python vp_judge.py --batch 128 \
    || { echo "RETRY judge ${S}"; python vp_judge.py --batch 32; } \
    || { echo "FAILED judge ${S}" | tee -a outputs/VP_FAILURES.txt; continue; }
  echo "=== layer ${S}: chart    $(date -u +%FT%TZ)"
  python vp_result.py > "outputs/RESULT_VP_layer${S}.log" 2>&1 || echo "chart skipped ${S}"
  echo "=== layer ${S}: DONE     $(date -u +%FT%TZ)"
done
echo "=== all layers complete $(date -u +%FT%TZ)"
touch outputs/VP_COMPLETE
