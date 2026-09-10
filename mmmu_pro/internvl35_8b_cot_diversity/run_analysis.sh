#!/bin/bash
# Analysis for one arm. The arm is selected by TEMP/TAG (see arm.sh); --temperature is mandatory
# because cells are keyed on (id, top_p) only, and --ks 1,8 because these arms have 8 ballots per
# cell, not 16. analyze.py refuses mixed models, mixed temperatures and duplicated cells.
#
#   ./run_analysis.sh                            commit + analyse T=1.2  (default arm)
#   TEMP=<T> TAG=<tag> ./run_analysis.sh         commit + analyse another temperature arm
#   ./run_analysis.sh partial                    analyse whatever exists SO FAR, straight from
#                                                outputs/, committing nothing. Safe to repeat as
#                                                often as you like while generation is running.
set -euo pipefail
cd "$(dirname "$0")"
source /venv/main/bin/activate
. ./arm.sh
mkdir -p cots outputs
KS="${KS:-1,8}"
# GRID comes from arm.sh: the grid generated is the grid analysed

count_q() { cat "$@" 2>/dev/null | python -c \
  "import sys,json;print(len({json.loads(l)['id'] for l in sys.stdin if l.strip()}))"; }

if [ "${1:-commit}" = partial ]; then
  # No merge and no commit: the committed dataset stays whatever it was. analyze.py warns about
  # cells below the modal ballot count (a question caught mid-generation) and build_matrix keeps
  # only questions present at EVERY top_p -- so a partial run is a smaller balanced n, never a
  # ragged one, and the ballot rule (truncated = spoiled = counts as wrong) is identical.
  set -- $OUTP.shard*.jsonl
  [ -e "$1" ] || { echo "[partial] no traces yet: $OUTP.shard*.jsonl"; exit 1; }
  n=$(cat "$@" | wc -l); q=$(count_q "$@")
  out=outputs/PRELIM_${TAG}_${q}q
  echo "[partial] T=$TEMP  $n traces, $q questions touched -> $out.md"
  python analyze.py --glob "$OUTP.shard*.jsonl" --temperature "$TEMP" --ks "$KS" --grid "$GRID" --out $out.md
  python result_chart_ivl.py --json $out.json --out $out.png || echo "[partial] chart skipped"
  exit 0
fi

GZ=cots/ivl35_$TAG.jsonl.gz
# The raw per-shard files are gitignored working copies; only the gzipped dataset is committed.
# In a fresh checkout there is nothing to merge, so analyse the committed dataset as-is rather
# than failing -- that is what makes the committed result reproducible from a clone.
if compgen -G "$OUTP.shard*.jsonl" > /dev/null; then
  # Write via a temp file and mv. `cat ... > $GZ` truncates $GZ the instant the redirect opens,
  # so any failure in the pipeline destroys the committed dataset before it can be rebuilt --
  # which is exactly what happened on 2026-09-10 when the glob matched nothing.
  cat $OUTP.shard*.jsonl | gzip -9 > "$GZ.tmp"
  mv "$GZ.tmp" "$GZ"
  echo "merged $(zcat $GZ | wc -l) rows, $(zcat $GZ | count_q /dev/stdin) questions -> $GZ"
elif [ -f "$GZ" ]; then
  echo "no raw shards; analysing the committed $GZ as-is ($(zcat $GZ | wc -l) rows)"
else
  echo "neither $OUTP.shard*.jsonl nor $GZ exists -- nothing to analyse"; exit 1
fi
python analyze.py --glob "$GZ" --temperature "$TEMP" --ks "$KS" --grid "$GRID" \
  --out outputs/RESULT_IVL35_${TAG^^}.md
python result_chart_ivl.py --json outputs/RESULT_IVL35_${TAG^^}.json --out outputs/ivl35_${TAG}_result.png
