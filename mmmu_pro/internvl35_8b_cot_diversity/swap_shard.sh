#!/bin/bash
# swap_shard.sh <i> [EXTRA_ARGS...]  -- relaunch ONE shard with new engine flags at its next CHUNK
# BOUNDARY (i.e. right after its next chunk is saved), so no in-flight work is lost. Stops only
# that shard's watchdog + python + EngineCore (identified as the process owning GPU <i>).
set -u
cd "$(dirname "$0")"
. ./arm.sh
i=$1; shift; EXTRA="$*"
f=$OUTP.shard$i.jsonl
base=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
echo "[swap $i] waiting for next chunk (rows now $base) ... $(date -u +%TZ)"
while :; do n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 ); [ "$n" -gt "$base" ] && break; sleep 20; done
echo "[swap $i] chunk landed (rows $n) $(date -u +%TZ); stopping shard $i"
wp=$(cat $PIDF$i.pid 2>/dev/null); [ -n "$wp" ] && { pkill -TERM -P "$wp" 2>/dev/null; kill "$wp" 2>/dev/null; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $i)
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$uuid" | cut -d, -f1); do kill -9 "$p" 2>/dev/null; done
timeout 60 bash -c "until [ \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i) -lt 2000 ]; do sleep 1; done"
echo "[swap $i] GPU $i free ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $i)); relaunching with: $EXTRA"
SHARD_IDS="$i" EXTRA_ARGS="$EXTRA" ./run_sweep.sh | tail -1
