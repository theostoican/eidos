# Local vLLM patch: incremental penalty-path cache (2026-09-25)

Applied to /venv/vllm (vLLM 0.26.0) on this instance; originals in vllm_patch_backup/.
Files: vllm/v1/sample/ops/penalty_cache.py (new), ops/penalties.py (`_convert_to_tensors`
routes through the cache), v1/sample/rejection_sampler.py (`_combine_outputs_with_spec_tokens`
returns unmaterialised rows; bad-words path materialises them).

Why: with presence_penalty=1.5 (card baseline, T=1.2 arm) upstream rebuilt every request's
full output history into a padded tensor from Python lists each decode step, plus one list
copy per draft token under MTP: ~220 ms CPU/step, GPUs 20-30% busy, ~3k tok/s total vs
~10k without penalties. The patch appends only new tokens to per-request numpy buffers and
assembles the padded tensor on the GPU from a flat pinned copy.

Numerics: the tensor handed to apply_penalties is identical to upstream's (same shape, pad,
dtype). Verified offline over 600 adversarial steps (id reuse, -1 placeholders, spec rollbacks,
clears, list replacement) and at runtime: the first 50 calls and every 200th are compared
against the upstream construction; any mismatch logs an error and disables the cache for the
rest of the process (upstream path). VLLM_PENALTY_CACHE=0 disables it. Sampling results are
therefore unchanged; rows are not stamped differently. Baseline rows generated before the
restart (all 11,256, finished 03:15 UTC) ran on the unpatched engines.

Test: /tmp/claude-0/-root/*/scratchpad/test_penalty_cache.py (copy kept as tests/test_penalty_cache.py).
