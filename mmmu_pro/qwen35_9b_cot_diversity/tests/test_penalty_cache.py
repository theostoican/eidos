import random, gc, time, torch, numpy as np
from vllm.v1.sample.ops.penalty_cache import PenaltyCache, LazyRows
from vllm.utils.torch_utils import make_tensor_with_pad
try:
    from vllm.v1.sample.rejection_sampler import RejectionSampler
    combine = RejectionSampler._combine_outputs_with_spec_tokens
    print("using patched RejectionSampler._combine_outputs_with_spec_tokens")
except Exception as e:
    print("rejection_sampler import failed:", repr(e)[:200]); raise

def upstream_combine(output_token_ids, spec_token_ids):
    if spec_token_ids is None: return output_token_ids
    result = []
    for out, spec in zip(output_token_ids, spec_token_ids):
        if len(spec) == 0: continue
        result.append(out)
        for i in range(len(spec) - 1):
            result.append([*result[-1], spec[i]])
    return result

VOCAB = 151936; dev = torch.device("cpu")
rng = random.Random(0)
cache = PenaltyCache(); cache.dead = False
reqs = []  # live output lists
def new_req(): return [rng.randrange(VOCAB) for _ in range(rng.choice([0, 1, 1, 2, 300, 2000]))]
for _ in range(60): reqs.append(new_req())
n_checks = 0; t_cache = t_up = 0.0
for step in range(600):
    # mutations mirroring gpu_model_runner: extend, optimistic -1 then rollback, del tail, clear, replace, remove/add
    for lst in reqs:
        r = rng.random()
        if r < 0.70: lst.extend(rng.randrange(VOCAB) for _ in range(rng.randint(1, 3)))
        elif r < 0.80: lst.extend([-1] * rng.randint(1, 3))            # async placeholders
        elif r < 0.88:
            if len(lst) > 3: del lst[-rng.randint(1, 3):]               # spec rollback
        elif r < 0.90: lst.clear()
        else: pass
        # fix placeholders (as update_async_output_token_ids would)
        for i in range(max(0, len(lst) - 8), len(lst)):
            if lst[i] == -1: lst[i] = rng.randrange(VOCAB)
    if rng.random() < 0.3:   # finish some requests, add new ones (id reuse pressure)
        for _ in range(rng.randint(1, 4)):
            if reqs: reqs.pop(rng.randrange(len(reqs)))
        gc.collect()
        for _ in range(rng.randint(1, 4)): reqs.append(new_req())
    if rng.random() < 0.05 and reqs:    # resume: list object replaced
        i = rng.randrange(len(reqs)); reqs[i] = list(reqs[i])
    rng.shuffle(reqs)
    batch = list(reqs) + ([None] if rng.random() < 0.1 else [])
    spec = [[rng.randrange(VOCAB) for _ in range(rng.choice([0, 2, 2, 2, 1]))] for _ in batch] if rng.random() < 0.85 else None
    rows = combine(batch, spec)
    if spec is None:
        assert rows is batch
        rows_up = [b if b is not None else [] for b in batch]
    else:
        assert isinstance(rows, LazyRows)
        rows_up = upstream_combine([b if b is not None else [] for b in batch], spec)
    rows_in = rows if spec is not None else [b if b is not None else [] for b in batch]
    t0 = time.perf_counter(); got = cache.build(rows_in, VOCAB, dev); t_cache += time.perf_counter() - t0
    t0 = time.perf_counter(); ref = make_tensor_with_pad(rows_up, pad=VOCAB, dtype=torch.int64, device="cpu"); t_up += time.perf_counter() - t0
    assert not cache.dead, f"cache disabled itself at step {step}"
    assert got.shape == ref.shape and torch.equal(got, ref), f"MISMATCH step {step} {got.shape} {ref.shape}"
    n_checks += 1
print(f"OK: {n_checks} steps identical to upstream; internal verifications {cache.stats}; "
      f"cache {t_cache*1e3/n_checks:.2f} ms/step vs upstream {t_up*1e3/n_checks:.2f} ms/step (CPU device, incl. pad fill)")
# big-batch timing closer to production: 170 reqs x ~8k tokens, spec 2
reqs = [[rng.randrange(VOCAB) for _ in range(rng.randint(2000, 20000))] for _ in range(170)]
cache = PenaltyCache(); cache.calls = 10**6  # skip verification for timing
for i in range(5):
    for l in reqs: l.extend([rng.randrange(VOCAB)] * 2)
    spec = [[1, 2] for _ in reqs]
    rows = combine(reqs, spec)
    t0 = time.perf_counter(); got = cache.build(rows, VOCAB, dev); tc = time.perf_counter() - t0
    t0 = time.perf_counter(); ref = make_tensor_with_pad(upstream_combine(reqs, spec), pad=VOCAB, dtype=torch.int64, device="cpu"); tu = time.perf_counter() - t0
    assert torch.equal(got, ref)
    print(f"prod-size step {i}: cache {tc*1e3:.1f} ms  upstream {tu*1e3:.1f} ms  (rows {ref.shape})")
