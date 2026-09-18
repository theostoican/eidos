# Visual-premise arm for this directory, sourced by ../common/run_vp_arm.sh.
VP_GLOB="cots/ivl35_t12.jsonl.gz"
VP_TEMP=1.2
VP_GRID=0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
VP_LABEL="InternVL3.5-8B, MMMU-Pro standard (10 options), T=1.2"
VP_REFERENCE=outputs/RESULT_IVL35_T12.json   # vp_result.py aborts if its maj@1 disagrees with this
VP_NSAMPLES=8                                # ballots per cell in the traces
VP_LAYERS=8                                  # how many of them to extract+judge
VP_VOTE_K=8                                  # maj@k rows; the FIRST is the one charted
VP_K_CELL=20                                 # count-matching size for cell Vendi
