#!/bin/bash
# Waits for the T=1.6 arm to exist and finish, then analyses it -- so the last leg of the
# plan also completes with no manual step. Deliberately a SEPARATE script from
# chain_after_dense.sh, which is currently running: bash reads a script incrementally as it
# executes, so editing a live one can corrupt its control flow mid-run.
cd "$(dirname "$0")"
source /venv/main/bin/activate

# 1. wait for the chain to launch the T=1.6 arm (it only does so on a DEAD verdict).
#    Bounded: if the verdict is ALIVE the arm never starts and this must not hang forever.
for _ in $(seq 720); do
  [ -f logs/t16_gen.shard0.log ] && break
  sleep 30
done
if [ ! -f logs/t16_gen.shard0.log ]; then
  echo "T16_NEVER_LAUNCHED -- verdict was probably ALIVE; nothing to analyse"; exit 0
fi

# 2. wait for both shards to finish (or for generation to die)
while :; do
  d=$(grep -l "^\[done\]" logs/t16_gen.shard0.log logs/t16_gen.shard1.log 2>/dev/null | wc -l)
  [ "$d" -ge 2 ] && break
  if [ "$(ps -eo args | grep -c '[c]ot_gen.py')" -eq 0 ]; then
    echo "T16_GENERATION_STOPPED_EARLY $(date -u +%H:%M:%S) -- analysing what completed"
    break
  fi
  sleep 60
done

# 3. analyse the T=1.6 arm ALONE. --temperature 1.6 is mandatory, not decorative: cells are
#    keyed on (id, top_p), so without it T=1.0 and T=1.6 rows merge into 32-ballot cells and
#    silently average two experiments into one curve.
python phase1_analysis.py \
  --glob "outputs/t16_gen.shard*.jsonl" --temperature 1.6 \
  --grid 0.5,0.7,0.9,0.95,1.0 \
  --shape-boot 20000 --out outputs/PHASE1_T16.md > logs/phase1_t16.log 2>&1
echo "T16_ANALYSIS_DONE $(date -u +%H:%M:%S)"
python - <<'PY'
import json
try:
    d=json.load(open("outputs/PHASE1_T16.json"))
    best=max(r["p_joint"] for r in d["spoiled_full"])
    row=max(d["spoiled_full"], key=lambda r: r["p_joint"])
    print(f"T16_VERDICT {'INVERTED-U' if best>=0.95 else 'STILL FLAT'} "
          f"best_p_joint={best:.3f} at k={row['k']} argmax={row['argmax']} quad_a={row['quad_a']:+.4f}")
except Exception as e:
    print("T16_VERDICT ERROR", e)
PY
