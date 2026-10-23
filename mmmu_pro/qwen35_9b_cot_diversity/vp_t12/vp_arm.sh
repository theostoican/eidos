# Visual-premise arm for the T=1.2 / top_k=-1 / presence 1.5 full-benchmark sweep (README §12),
# sourced by ../../common/run_vp_arm.sh from THIS directory. Kept apart from ../vp_arm.sh
# (the T=1.6 arm) so the two arms' outputs/ never overwrite each other.
VP_GLOB="../cots/t12_*.jsonl.gz"
VP_TEMP=1.2
VP_GRID=0.5,0.6,0.7,0.75,0.8,0.85,0.95
VP_LABEL="Qwen3.5-9B, MMMU-Pro standard (10 options), T=1.2, top_k off, presence 1.5"
VP_REFERENCE=../outputs/RESULT_T12_SWEEP_FULL.json   # vp_result.py aborts if its maj@1 disagrees with this
VP_NSAMPLES=8                             # ballots per cell in the traces
VP_LAYERS=8                               # extract+judge all of them (1,729 questions x 7 top_p per layer)
VP_VOTE_K=8                               # maj@k rows; the FIRST is the one charted
VP_K_CELL=20                              # count-matching size for cell Vendi (as the T=1.0 arm)
