#!/bin/bash
# Visual-premise pipeline, one SAMPLE LAYER at a time: extract -> judge -> report.
#
# Run FROM AN ARM DIRECTORY (normally via run_vp_arm.sh): every data path is relative to the
# current directory, every script is resolved from this file's own directory (common/).
#
# Layer-at-a-time because both passes need the whole 24GB card, and because a finished layer
# is a COMPLETE, BALANCED dataset over the full question x top_p grid: stop after any layer
# and the curve is still computed on the same cells at every top_p.
#
# Both passes resume from their own output, so re-running is safe: finished work is skipped.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /venv/main/bin/activate

OUT=${VP_OUT:-outputs}
LAYERS_DIR=${VP_LAYERS_DIR:-$OUT/vp_layers}
PREMISES=${VP_PREMISES:-$OUT/vp_premises.jsonl}
QUESTIONS=${VP_QUESTIONS:-$OUT/vp_q.json}
VERDICTS=${VP_VERDICTS:-$OUT/vp_verdicts.jsonl}
FAILURES=${VP_FAILURES:-$OUT/VP_FAILURES.txt}
COMPLETE=${VP_COMPLETE:-$OUT/VP_COMPLETE}
RESULT_ARGS=${VP_RESULT_ARGS:?set VP_RESULT_ARGS (--grid, --n-samples, --reference, --vote-k)}
LABEL=${VP_LABEL:?set VP_LABEL (it contains spaces, so it travels separately)}

# printf, not `seq -w`: seq pads to the width of its LARGEST argument, so `seq -w 0 15` gives
# 00..15 while `seq -w 0 7` gives 0..7 -- every filename then misses and the loop silently
# completes having done nothing.
LAST=${VP_LAST_LAYER:-7}
for i in $(seq 0 "$LAST"); do
  S=$(printf "%02d" "$i")
  L="$LAYERS_DIR/layer${S}.jsonl.gz"
  if [ ! -f "$L" ]; then echo "MISSING $L" | tee -a "$FAILURES"; continue; fi
  # A stage that dies must not idle the GPU until someone notices: retry once (both stages
  # resume), then move on and record it. An absent VP_FAILURES.txt means a clean run.
  echo "=== layer ${S}: extract  $(date -u +%FT%TZ)"
  python "$HERE/vp_extract.py" --layers "$L" --out "$PREMISES" --batch 256 \
    || { echo "RETRY extract ${S}"; python "$HERE/vp_extract.py" --layers "$L" --out "$PREMISES" --batch 64; } \
    || { echo "FAILED extract ${S}" | tee -a "$FAILURES"; continue; }
  echo "=== layer ${S}: judge    $(date -u +%FT%TZ)"
  python "$HERE/vp_judge.py" --premises "$PREMISES" --questions "$QUESTIONS" --out "$VERDICTS" --batch 128 \
    || { echo "RETRY judge ${S}"; python "$HERE/vp_judge.py" --premises "$PREMISES" --questions "$QUESTIONS" --out "$VERDICTS" --batch 32; } \
    || { echo "FAILED judge ${S}" | tee -a "$FAILURES"; continue; }
  echo "=== layer ${S}: report   $(date -u +%FT%TZ)"
  python "$HERE/vp_result.py" --verdicts "$VERDICTS" --premises "$PREMISES" \
    --layers "$LAYERS_DIR/layer*.jsonl.gz" --out "$OUT/RESULT_VP.md" \
    --png "$OUT/premise_soundness_vs_topp.png" --label "$LABEL" $RESULT_ARGS \
    > "$OUT/RESULT_VP_layer${S}.log" 2>&1 || echo "report skipped ${S}"
  echo "=== layer ${S}: DONE     $(date -u +%FT%TZ)"
done
echo "=== all layers complete $(date -u +%FT%TZ)"
touch "$COMPLETE"
