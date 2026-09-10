#!/bin/bash
# rebalance.sh <slow_shard i> <free_gpu g>
# At shard i's NEXT CHUNK BOUNDARY (so nothing in flight is lost), stop it and relaunch its remaining
# work as two 8-way sub-shards: slice i of 4 == slices i and i+4 of 8. Sub-shard i runs on GPU i
# (appending to $OUTP.shard<i>.jsonl via --resume); sub-shard i+4 runs on GPU g and
# writes $OUTP.shard<i+4>.jsonl. --resume-glob covers ALL shard files, so completed
# cells are skipped; the two slices are disjoint, so no cell is ever generated twice.
set -u
cd "$(dirname "$0")"
. ./arm.sh
i=$1; g=$2; j=$((i+4))
f=$OUTP.shard$i.jsonl
base=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
echo "[rebalance $i->$i+$j on GPU $i,$g] waiting for shard $i chunk boundary (rows $base) $(date -u +%TZ)"
while :; do n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 ); [ "$n" -gt "$base" ] && break; grep -q "=== shard $i DONE" $GENLOG$i.log 2>/dev/null && { echo "shard $i finished by itself"; exit 0; }; sleep 20; done
echo "[rebalance] boundary at rows $n $(date -u +%TZ); stopping shard $i"
wp=$(cat $PIDF$i.pid 2>/dev/null); [ -n "$wp" ] && { pkill -TERM -P "$wp" 2>/dev/null; kill "$wp" 2>/dev/null; }
uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $i)
for p in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader | grep "$uuid" | cut -d, -f1); do kill -9 "$p" 2>/dev/null; done
timeout 60 bash -c "until [ \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i) -lt 2000 ]; do sleep 1; done"
# sub-shard i on GPU i, sub-shard j on GPU g: launch via run_sweep with NSHARDS=8. run_sweep pins
# CUDA_VISIBLE_DEVICES=<shard id>, so for j we launch by hand with the same args.
NSHARDS=8 SHARD_IDS="$i" ./run_sweep.sh | tail -1
source /venv/main/bin/activate; export HF_HOME=/workspace/.hf_home PYTHONUNBUFFERED=1 CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH
args=$(NSHARDS=8 ./run_sweep.sh args)
setsid bash -c "for try in \$(seq 1 20); do echo \"=== shard $j attempt \$try \$(date -u +%FT%TZ) ===\"; CUDA_VISIBLE_DEVICES=$g python cot_gen.py $args --shard-id $j && { echo '=== shard $j DONE ==='; exit 0; }; echo \"=== shard $j exited rc=\$? ; restarting in 30s ===\"; sleep 30; done; echo '=== shard $j GAVE UP ==='" >> $GENLOG$j.log 2>&1 < /dev/null &
echo $! > $PIDF$j.pid
echo "[rebalance] launched sub-shard $i on GPU $i and sub-shard $j on GPU $g $(date -u +%TZ)"
