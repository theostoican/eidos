#!/bin/bash
# Keep the GPU busy until the run finishes.
#
# The first attempt at this run died four minutes in on one over-length prompt and left the
# card idle for ten hours before anyone looked. Both stages resume from their own output, so
# a death should never cost more than an engine reload. This restarts run_vp.sh until it
# plants VP_COMPLETE, and gives up after MAX_TRIES so a genuinely broken config is
# not retried forever.
#
# Run from an arm directory; honours the same VP_* environment variables as run_vp.sh.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_TRIES=${MAX_TRIES:-40}
OUT=${VP_OUT:-outputs}
COMPLETE=${VP_COMPLETE:-$OUT/VP_COMPLETE}
LOG=${VP_LOG:-logs/vp_run.log}
mkdir -p "$(dirname "$LOG")"
for i in $(seq 1 "$MAX_TRIES"); do
  if [ -f "$COMPLETE" ]; then
    echo "[watchdog] complete after $((i-1)) attempt(s) $(date -u +%FT%TZ)"; exit 0
  fi
  echo "[watchdog] attempt $i/$MAX_TRIES $(date -u +%FT%TZ)"
  "$HERE/run_vp.sh" >> "$LOG" 2>&1
  echo "[watchdog] run_vp.sh exited $? $(date -u +%FT%TZ)"
  sleep 10
done
echo "[watchdog] giving up after $MAX_TRIES attempts $(date -u +%FT%TZ)"
exit 1
