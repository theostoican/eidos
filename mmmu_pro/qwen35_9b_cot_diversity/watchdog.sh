#!/bin/bash
# Supervisor-managed watchdog: keeps the idempotent orchestrator advancing until the
# experiment is complete, surviving Claude-agent restarts and container reboots.
DIR=/workspace/eidos/mmmu_pro/qwen35_9b_cot_diversity
cd "$DIR"
while true; do
  if [ -f outputs/FINAL_SUMMARY.md ]; then
    echo "[watchdog $(date -u +%H:%M:%S)] experiment complete; idle"
    sleep 600; continue
  fi
  echo "[watchdog $(date -u +%H:%M:%S)] advancing orchestrator"
  bash orchestrator.sh >> logs/orchestrator.log 2>&1
  sleep 90
done
