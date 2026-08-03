#!/bin/bash
# When the low-top_p T=1.6 extension finishes, analyse the FULL 9-point T=1.6 curve.
cd "$(dirname "$0")"
source /venv/main/bin/activate
while :; do
  d=$(grep -l "^\[done\]" logs/t16low.shard0.log logs/t16low.shard1.log 2>/dev/null | wc -l)
  [ "$d" -ge 2 ] && break
  [ "$(ps -eo args | grep -c '[c]ot_gen.py')" -eq 0 ] && { echo "T16LOW_STOPPED_EARLY"; break; }
  sleep 60
done
echo "T16LOW_GEN_DONE $(date -u +%H:%M:%S)"
python phase1_analysis.py --glob "outputs/t16_gen.shard*.jsonl" --temperature 1.6 \
  --grid 0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0 \
  --shape-boot 20000 --out outputs/PHASE1_T16_FULL.md > logs/phase1_t16full.log 2>&1
echo "T16LOW_ANALYSIS_DONE $(date -u +%H:%M:%S)"
python - <<'PY'
import json
d=json.load(open("outputs/PHASE1_T16_FULL.json"))
for r in d["spoiled_full"]:
    print(f"  k={r['k']:>2} argmax={r['argmax']} P(joint)={r['p_joint']:.3f} "
          f"quad_a={r['quad_a']:+.4f} p_omni={r['p_omnibus']:.4f}")
best=max(r["p_joint"] for r in d["spoiled_full"])
print(f"T16FULL_VERDICT {'INVERTED-U CONFIRMED' if best>=0.95 else 'no interior peak'} best_p_joint={best:.3f}")
PY
