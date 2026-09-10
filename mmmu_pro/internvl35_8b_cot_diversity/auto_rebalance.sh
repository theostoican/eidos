#!/bin/bash
# When a shard prints DONE, hand its GPU to the shard with the MOST remaining work via rebalance.sh
# (which waits for that shard's chunk boundary, then splits its slice across the two GPUs).
# Each GPU is handed over once; each slow shard is split at most once.
set -u
cd "$(dirname "$0")"
. ./arm.sh
declare -A used_gpu split_shard
cells() { python3 -c "import json;print(len(json.load(open('${OUTP}_q.shard$1.json')))*10)" 2>/dev/null || echo 0; }
while :; do
  for d in 0 1 2 3; do
    [ -n "${used_gpu[$d]:-}" ] && continue
    grep -q "=== shard $d DONE" $GENLOG$d.log 2>/dev/null || continue
    used_gpu[$d]=1
    best=""; bestrem=0
    for i in 0 1 2 3; do
      [ "$i" = "$d" ] && continue; [ -n "${split_shard[$i]:-}" ] && continue
      grep -q "=== shard $i DONE" $GENLOG$i.log 2>/dev/null && continue
      rem=$(( $(cells $i) - $(wc -l < $OUTP.shard$i.jsonl 2>/dev/null || echo 0) / 8 ))
      [ "$rem" -gt "$bestrem" ] && { best=$i; bestrem=$rem; }
    done
    [ -z "$best" ] && { echo "[auto] shard $d done; nothing left to split $(date -u +%TZ)"; continue; }
    split_shard[$best]=1
    echo "[auto] shard $d DONE -> splitting shard $best ($bestrem cells left) onto GPUs $best,$d $(date -u +%TZ)"
    ./rebalance.sh $best $d
  done
  alldone=1; for i in 0 1 2 3; do grep -q "=== shard $i DONE" $GENLOG$i.log 2>/dev/null || alldone=0; done
  [ $alldone = 1 ] && { echo "[auto] all 4 original shards DONE $(date -u +%TZ)"; exit 0; }
  sleep 60
done
