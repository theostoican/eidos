# Visual-premise arm for this directory, sourced by ../common/run_vp_arm.sh.
# The T=1.6 sweep: the arm with the interior optimum, and the one where sampling collapses
# above top_p~0.7 -- i.e. where a top_p effect on visual reads should show if it exists.
VP_GLOB="cots/t16_sweep.shard*.jsonl.gz"
VP_TEMP=1.6
VP_GRID=0.1,0.2,0.3,0.4,0.5,0.7,0.9,0.95,1.0
VP_LABEL="Qwen3.5-9B, MMMU-Pro standard (10 options), T=1.6"
VP_REFERENCE=outputs/RESULT_T16.json      # vp_result.py aborts if its maj@1 disagrees with this
VP_NSAMPLES=16                            # ballots per cell in the traces
VP_LAYERS=16                              # how many of them to extract+judge (86 questions: use all)
VP_VOTE_K=8,16                            # maj@k rows; the FIRST is the one charted
# Count-matching size for cell Vendi. 20 (the other arms) leaves 6 questions paired across this
# grid, because traces at top_p>=0.9 state few visual claims (11 premises/cell at 1.0); 8 keeps 26.
VP_K_CELL=8
