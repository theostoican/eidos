#!/bin/bash
# Re-run the pre-registered shape tests after every completed chunk of the Kimi T=1.2 arm.
#
# EXPLORATORY BY CONSTRUCTION. This tests the same hypothesis repeatedly as n grows, which is
# optional stopping: the P(joint) printed here is NOT a valid alpha=0.05 quantity under this
# procedure. It is early warning -- whether to keep spending GPU hours -- not a result.
# Decision point is n>=88 (chunk 11), the sample the repo's original Qwen arm was published on.
set -u
cd /root/eidos/mmmu_pro/qwen35_9b_cot_diversity
source /venv/main/bin/activate
F=outputs/kimi_t125_10pct.jsonl
S=/tmp/claude-0/-root/5140552e-fa27-4ed7-bf9f-9f9c2cb834dd/scratchpad/SHAPE
last=0
while true; do
  if [ -f "$F" ]; then
    n=$(wc -l < "$F")
    if [ "$n" -ge $((last + 1152)) ]; then
      last=$n
      if python analyze.py --glob "$F" --temperature 1.25 --ks 1,16 \
           --grid 0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0 --out "$S.md" > /dev/null 2>&1; then
        python - "$S" <<'PY'
import json,re,sys
s=sys.argv[1]
nq=re.search(r"n = (\d+) questions", open(s+".md").read()).group(1)
d=json.load(open(s+".json"))
r=next(x for x in d["results"] if x["k"]==1); r16=next(x for x in d["results"] if x["k"]==16)
# A down-opening quadratic is NOT evidence of a hump: a flat curve with a cliff at the
# right edge fits one too. A genuine interior optimum needs a LEFT ARM -- a real rise from
# the left edge up to the peak. Report it explicitly and require it before crying HUMP.
m=r["means"]; imax=m.index(max(m))
rise=m[imax]-m[0]; drop=m[imax]-m[-1]
flat=max(m[:-2])-min(m[:-2])          # spread excluding the two right-most (cliff) points
v=("FLAT/NULL" if r["p_omnibus"]>0.05
   else "HUMP?" if (r["p_joint"]>=0.60 and r["quad_a"]<0 and rise>=0.03) else
        "CLIFF-ONLY" if (r["quad_a"]<0 and rise<0.03) else "NO-HUMP")
print(f"SHAPE n={nq} | k1 argmax={r['argmax']} F={r['F']:.1f} p={r['p_omnibus']:.3f} "
      f"Pjoint={r['p_joint']:.2f} a={r['quad_a']:+.2f} rise={rise:+.3f} drop={drop:+.3f} "
      f"flat={flat:.3f} | k16 argmax={r16['argmax']} a={r16['quad_a']:+.2f} | {v}", flush=True)
PY
      else
        echo "SHAPE analyze FAILED at $n rows" 
      fi
    fi
  fi
  sleep 60
done
