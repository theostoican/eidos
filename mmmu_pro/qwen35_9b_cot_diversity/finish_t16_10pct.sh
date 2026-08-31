#!/bin/bash
# Runs when both generation shards have exited: verify the grid is COMPLETE, then compress
# into cots/ (repo convention), re-analyse the T=1.6 arm at n=173, and redraw the figure.
# Refuses to analyse a partial grid -- analyze.py only WARNS on under-modal cells, which
# would silently report a curve computed on uneven denominators across the swept axis.
set -u
cd /root/eidos/mmmu_pro/qwen35_9b_cot_diversity
source /venv/main/bin/activate

while pgrep -f "cot_gen.py.*shard-id" > /dev/null; do sleep 60; done
echo "=== generation processes exited $(date -u +%FT%TZ)"

for s in 0 1; do
  f="outputs/t16_10pct.shard${s}.jsonl"
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
  gzip -c "$f" > "cots/t16_10pct.shard${s}.jsonl.gz"
done

python - <<'PY' || exit 1
import glob, gzip, json, collections, sys
cells = collections.Counter(); ids = set()
for f in sorted(glob.glob("cots/t16_*.jsonl.gz")):
    for line in gzip.open(f, "rt"):
        d = json.loads(line)
        if d.get("temperature") != 1.6: continue
        cells[(d["id"], d["top_p"])] += 1; ids.add(d["id"])
sizes = collections.Counter(cells.values())
print(f"[verify] questions={len(ids)} cells={len(cells)} rows={sum(cells.values())} ballots/cell={dict(sizes)}")
bad = {k: v for k, v in cells.items() if v != 16}
expect_q = len(ids)
if bad or len(cells) != expect_q * 9:
    print(f"[verify] INCOMPLETE: {len(bad)} cell(s) != 16 ballots, {len(cells)} cells for {expect_q} questions x 9 top_p")
    sys.exit(1)
print("[verify] grid complete")
PY

python analyze.py --glob "cots/t16_*.jsonl.gz" --temperature 1.6 \
  --grid 0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0 --out outputs/RESULT_T16.md > logs/analyze_t16.log 2>&1 \
  && python result_chart.py --zoom > logs/chart.log 2>&1 \
  && echo "=== ANALYSIS+CHART COMPLETE $(date -u +%FT%TZ)" || echo "=== POSTPROCESS FAILED"
