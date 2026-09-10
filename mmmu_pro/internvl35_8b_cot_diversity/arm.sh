# This arm's identity. Sourced by every script here. TEMP/TAG/GRID select an arm, and every
# trace, log and pid path derives from TAG, so a second temperature arm can be run alongside
# this one without sharing a single file. That separation is by PATH, not by check:
# analyze.py's temperature guard would refuse a mixed glob, but only after the GPU hours had
# already been spent. Committed here: T=1.2 only.
TEMP="${TEMP:-1.2}"
TAG="${TAG:-t12}"
GRID="${GRID:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
# The pre-registered shape test needs the argmax to be INTERIOR: _two_lines() returns False when
# it lands on the first or last grid point, so P(shape)/P(joint) are ~0 there BY CONSTRUCTION.
# Measured on this arm's own data, restricted to 0.3-0.8: the k=1 argmax is 0.8, the right
# endpoint, and P(shape) reads 0.030 where the full grid reads 1.000. Keep the peak bracketed.
# Question set: 0.10 = the 172-q 10% sample, 0.05 = the 86-q set nested inside it (and the
# exact set the Qwen T=1.6 arm used). --nest-from/--sample-seed are fixed so the sets stay nested.
SAMPLE_FRAC="${SAMPLE_FRAC:-0.10}"
NSAMPLES="${NSAMPLES:-8}"
OUTP="outputs/ivl35_$TAG"                 # $OUTP.shard<i>.jsonl , ${OUTP}_q.json , ${OUTP}_q.shard<i>.json
GENLOG="logs/gen.$TAG.shard"              # $GENLOG<i>.log
PIDF="logs/$TAG.shard"                    # $PIDF<i>.pid
