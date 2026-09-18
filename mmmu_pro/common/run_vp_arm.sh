#!/bin/bash
# The whole visual-premise analysis for ONE arm: prep -> extract+judge -> diversity -> chart.
#
#   cd ../qwen35_9b_cot_diversity     && ../common/run_vp_arm.sh      # Qwen3.5-9B,     T=1.6
#   cd ../internvl35_8b_cot_diversity && ../common/run_vp_arm.sh      # InternVL3.5-8B, T=1.2
#
# The arm is whatever ./vp_arm.sh in the current directory says it is (trace glob, temperature,
# grid, samples per cell, reference result). Extractor, judge and encoder are the same for
# every arm, so only the traces differ. Everything resumes: re-running after a crash is safe.
# Products: outputs/premise_soundness_vs_topp.png and outputs/topp_correctness_and_diversity.png
# (plus RESULT_VP*.md/.json and the gitignored intermediates they were computed from).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f ./vp_arm.sh ] || { echo "run from an arm directory: ./vp_arm.sh not found in $PWD"; exit 1; }
. ./vp_arm.sh
source /venv/main/bin/activate
OUT=outputs
LAST=$((VP_LAYERS - 1))
mkdir -p logs "$OUT"

if [ ! -f "$OUT/vp_layers/layer00.jsonl.gz" ] || [ ! -f "$OUT/vp_q.json" ]; then
  echo "##### prep  $(date -u +%FT%TZ)"
  python "$HERE/vp_prep.py" --glob "$VP_GLOB" --temperature "$VP_TEMP" --grid "$VP_GRID" \
    --outdir "$OUT/vp_layers" --questions-out "$OUT/vp_q.json" || exit 1
fi

echo "##### extract+judge ($VP_LABEL)  $(date -u +%FT%TZ)"
VP_OUT=$OUT VP_LAST_LAYER=$LAST VP_LABEL="$VP_LABEL" \
  VP_RESULT_ARGS="--grid $VP_GRID --reference $VP_REFERENCE --n-samples $VP_NSAMPLES --vote-k $VP_VOTE_K" \
  "$HERE/run_vp_watchdog.sh" || echo "##### watchdog gave up"

echo "##### diversity + chart  $(date -u +%FT%TZ)"
python "$HERE/vp_diversity.py" --premises "$OUT/vp_premises.jsonl" --grid "$VP_GRID" \
  --layers "$VP_LAYERS" --k-cell "$VP_K_CELL" --label "$VP_LABEL" \
  --out "$OUT/RESULT_VP_DIVERSITY.md" --cells-out "$OUT/vp_diversity_cells.json" \
  > logs/vp_diversity.log 2>&1 && \
python "$HERE/vp_one_chart.py" --vp "$OUT/RESULT_VP.json" --div "$OUT/RESULT_VP_DIVERSITY.json" \
  --out "$OUT/topp_correctness_and_diversity.png" >> logs/vp_diversity.log 2>&1 \
  || { echo "##### diversity/chart FAILED (see logs/vp_diversity.log)"; exit 1; }
echo "##### ALL DONE  $(date -u +%FT%TZ)"
