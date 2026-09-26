#!/usr/bin/env python
"""Streaming generator: same traces as cot_gen.py, fed continuously to N `vllm serve` engines.

cot_gen.py hands vLLM one chunk at a time via LLM.chat(), and a chunk ends only when its
slowest trace ends. At T=1.6 / low top_p ~7% of traces run to the 40,960-token cap, so every
chunk pays a long low-concurrency tail. Here each engine is kept full continuously: the unit
of work is ONE SAMPLE, and a finished sample is replaced immediately, so the GPUs never drain
until the very end of the run.

Nothing about WHAT is generated changes:
  - prompts are built by cot_gen.py's own build_content();
  - cot_gen.py sends one n=n_samples request per cell with seed=S. vLLM expands that into
    n child requests seeded S+index (v1/engine/parallel_sampling.py). Here each sample is
    sent as its own n=1 request with seed=S+sample_idx -- the identical child request, so
    sampling_cfg (seed=S) is stamped unchanged and analyze.py reads the rows unchanged;
  - --no-request-seed drops the per-request seed. A seeded request carries its own torch
    Generator, and ANY generator in the batch forces vLLM's sampler off FlashInfer onto the
    native path (sort-free Triton top-p over the 248k vocab + a per-row Python RNG loop,
    v1/sample/ops/topk_topp_sampler.py forward_cuda); measured, that made a 183-sequence
    decode step ~48 ms. The sampled DISTRIBUTION is unchanged -- a seed only fixes which
    random numbers are drawn -- and rows are stamped sampling_cfg.seed=None so it is visible;
  - engine-side settings that do not alter the sampling distribution (MTP speculative
    decoding: greedy draft, rejection-sampled against the target's own temperature/top_p
    distribution -- exact) are stamped separately in engine_cfg.

Concurrency is adaptive per engine: KV-bound decode wants as many sequences as the KV cache
holds, but sequences grow for ~40k tokens after admission, so a fixed count either
under-fills early or thrashes (preempt + recompute) later. Every --control-interval the
target in-flight count is raised while the engine has KV headroom and no queue, and cut
when it preempts.

Rows are appended one per sample as it completes; resume (always on) skips samples present.
"""
import argparse, asyncio, collections, glob, gzip, json, random, sys, time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cot_gen import build_content, collect_images, load_options  # identical prompt assembly
from PIL import Image

def scale_images(images, f):
    """Resample every image by factor f (LANCZOS). f=1.0 returns the originals untouched."""
    if f == 1.0:
        return images
    return {k: im.resize((max(1, round(im.size[0] * f)), max(1, round(im.size[1] * f))), Image.LANCZOS)
            for k, im in images.items()}

MAX_IMAGES = 8      # must match --limit-mm-per-prompt on the servers
METRIC_KEYS = ("vllm:generation_tokens_total", "vllm:num_requests_running",
               "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc",
               "vllm:num_preemptions_total", "vllm:spec_decode_num_accepted_tokens_total",
               "vllm:spec_decode_num_draft_tokens_total")


def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_questions(args):
    from datasets import load_dataset
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    N = len(ds)
    k = max(1, round(args.sample_frac * N))
    if args.nest_from:                      # same nested draw as cot_gen.py
        k0 = max(1, round(args.nest_from * N))
        base = random.Random(args.sample_seed).sample(range(N), k0)
        seen = set(base)
        pool = [i for i in range(N) if i not in seen]
        sel = sorted(base + random.Random(args.sample_seed + 1).sample(pool, k - k0))
    else:
        sel = sorted(random.Random(args.sample_seed).sample(range(N), k))
    if args.limit:
        sel = sel[:args.limit]
    scales = sorted(set(args.image_scales))
    qs, over = [], []
    for i in sel:
        row = ds[i]
        opts = load_options(row)
        images = collect_images(row)
        # PERCEPTION DIVERSITY: one prompt per distinct image scale; a chain (sample_idx) uses
        # the scale args.image_scales[sample_idx]. The text is byte-identical across scales.
        msgs = {f: [{"role": "user", "content": build_content(row["question"], opts, scale_images(images, f))}]
                for f in scales}
        content = msgs[scales[0]][0]["content"]
        n_img = sum(1 for c in content if c.get("type") == "image_url")
        if n_img > MAX_IMAGES:
            over.append((row["id"], n_img))
            continue
        qs.append({"id": row["id"], "subject": row.get("subject"), "gold": row["answer"],
                   "n_options": len(opts), "n_images": n_img, "ds_index": i,
                   "messages": msgs})
    if over:
        log(f"[skip] {len(over)} question(s) exceed {MAX_IMAGES} images: "
            + ", ".join(f"{q} ({n})" for q, n in over))
    if args.question_ids:
        keep = set(json.load(open(args.question_ids)))
        qs = [q for q in qs if q["id"] in keep]
        missing = keep - {q["id"] for q in qs}
        if missing:
            raise SystemExit(f"[abort] {len(missing)} ids in {args.question_ids} not in the sample")
        log(f"[sample] restricted to {len(qs)} question ids from {args.question_ids}")
    return N, qs


def env_cfg(path, base):
    """engine_cfg from a serve_gpu*.env file (KEY=VALUE lines, as serve.sh consumes them)."""
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k] = v.strip().strip("'").strip('"') if not v.strip().startswith("'") else v.strip()[1:-1]
    spec = env.get("SPEC") or None
    return {**base, "kv_cache_dtype": env.get("KV_DTYPE", "auto"),
            "spec_decode": json.loads(spec) if spec else None,
            "attention_backend": env.get("ATTN_BACKEND") or "FLASH_ATTN(default)",
            "compilation_cfg": json.loads(env["COMPILATION_CFG"]) if env.get("COMPILATION_CFG") else None}


class Engine:
    def __init__(self, url, target, lo, hi, cfg, env_file=None):
        self.url, self.target, self.lo, self.hi, self._cfg = url, target, lo, hi, cfg
        self.env_file, self._mtime, self._env_cfg = env_file, None, None
        self.active = 0
        self.prev = None            # (t, gen_tokens, preemptions)
        self.cond = asyncio.Condition()

    @property
    def cfg(self):
        if not self.env_file:
            return self._cfg
        m = Path(self.env_file).stat().st_mtime
        if m != self._mtime:
            self._mtime, self._env_cfg = m, env_cfg(self.env_file, self._cfg)
        return self._env_cfg

    async def scrape(self, session):
        vals = collections.defaultdict(float)
        async with session.get(f"{self.url}/metrics", timeout=aiohttp.ClientTimeout(total=10)) as r:
            for line in (await r.text()).splitlines():
                if line.startswith("#"):
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if name in METRIC_KEYS:
                    vals[name] += float(line.rsplit(" ", 1)[1])
        return vals


async def control_loop(session, engines, args, stop):
    """Adapt each engine's in-flight target; log throughput every --metrics-interval."""
    last_log, acc_prev = 0.0, {}
    while not stop.is_set():
        await asyncio.sleep(args.control_interval)
        now, parts, tot = time.time(), [], 0.0
        for e in engines:
            try:
                v = await e.scrape(session)
            except Exception as ex:
                parts.append(f"{e.url.rsplit(':', 1)[1]} DOWN ({type(ex).__name__})")
                e.prev = None
                continue
            g, pre = v["vllm:generation_tokens_total"], v["vllm:num_preemptions_total"]
            kv, wait = v["vllm:kv_cache_usage_perc"], v["vllm:num_requests_waiting"]
            rate = dpre = 0.0
            if e.prev:
                rate = (g - e.prev[1]) / (now - e.prev[0])
                dpre = pre - e.prev[2]
            e.prev = (now, g, pre)
            tot += rate
            old = e.target
            if dpre > args.preempt_tol:                    # thrashing: back off
                e.target = max(e.lo, int(e.target * 0.97))
            elif kv < args.kv_high and wait <= 1 and e.active >= e.target - 1:
                # headroom and nothing queued: add, proportionally to the free KV so a pool that
                # just drained (a cohort of long traces finishing together) refills in ~1 min
                # instead of ~10, and only by --step when close to the wall
                inc = args.step * max(1, int((args.kv_high - kv) * 20))
                e.target = min(e.hi, e.target + inc)
            if e.target > old:
                async with e.cond:
                    e.cond.notify_all()
            acc = ""
            a, d = (v["vllm:spec_decode_num_accepted_tokens_total"],
                    v["vllm:spec_decode_num_draft_tokens_total"])
            if d:
                pa, pd = acc_prev.get(e.url, (0.0, 0.0))
                acc = f" acc={(a - pa) / max(d - pd, 1):.2f}"
                acc_prev[e.url] = (a, d)
            parts.append(f"{e.url.rsplit(':', 1)[1]} {rate:5.0f} tok/s run={v['vllm:num_requests_running']:.0f} "
                         f"wait={wait:.0f} kv={kv:.2f} +pre={dpre:.0f} tgt={e.target} act={e.active}{acc}")
        if now - last_log >= args.metrics_interval:
            last_log = now
            log(f"[metrics] TOTAL {tot:6.0f} tok/s | " + " | ".join(parts))
            if args.metrics_out:
                with open(args.metrics_out, "a") as f:
                    f.write(json.dumps({"t": now, "tok_s": tot, "detail": parts}) + "\n")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--servers", required=True, help="comma-sep base URLs, one per engine")
    ap.add_argument("--inflight", type=int, default=96, help="initial sequences in flight per engine")
    ap.add_argument("--inflight-min", type=int, default=32)
    ap.add_argument("--inflight-max", type=int, default=448)
    ap.add_argument("--kv-high", type=float, default=0.95,
                    help="raise the in-flight target only while KV usage is below this")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--preempt-tol", type=int, default=2,
                    help="preemptions per control interval tolerated before backing off")
    ap.add_argument("--control-interval", type=float, default=10)
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--sample-frac", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=20260706)
    ap.add_argument("--nest-from", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--image-scales", default="",
                    help="comma-sep resize factor per sample_idx (n_samples values), e.g. "
                         "'1,1,1,1,0.75,1.25,0.6,1.5': chain i sees the images resampled by "
                         "factor i. Empty = 1.0 for every chain. Stamped in sampling_cfg and per row.")
    ap.add_argument("--question-ids", default="",
                    help="JSON list of question ids: restrict the run to exactly these (a frozen "
                         "subset of another run, e.g. for a paired baseline)")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--top-ps", required=True)
    ap.add_argument("--temperature", type=float, required=True)
    PROFILES = {"neutral": {"top_k": -1, "presence_penalty": 0.0},
                "qwen-recommended": {"top_k": 20, "presence_penalty": 1.5},
                "neutral-penalty": {"top_k": -1, "presence_penalty": 1.5}}   # neutral + card's presence penalty
    ap.add_argument("--sampling-profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=40960)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--kv-cache-dtype", default="auto")
    ap.add_argument("--seed", type=int, default=1234, help="work-order shuffle + per-request seed base")
    ap.add_argument("--engine-env", default="",
                    help="comma-sep env files, one per --servers entry (serve_gpu*.env). engine_cfg "
                         "is rebuilt from the file whenever a row is written, so an engine can be "
                         "reconfigured and restarted without restarting this client. Stop the "
                         "engine BEFORE editing its file: every sample completing afterwards comes "
                         "from the new engine.")
    ap.add_argument("--no-request-seed", action="store_true",
                    help="omit the per-request seed (keeps sampling on FlashInfer; see docstring)")
    ap.add_argument("--engine-cfg", default="{}",
                    help="JSON stamped into every row as engine_cfg: one object for all engines, "
                         "or a list with one object per --servers entry")
    ap.add_argument("--out", required=True, help="rows appended here (resume is always on)")
    ap.add_argument("--resume-glob", default="", help="also count samples present in these files")
    ap.add_argument("--questions-out", default="")
    ap.add_argument("--max-seconds", type=float, default=0, help="benchmark: stop after this long")
    ap.add_argument("--metrics-interval", type=float, default=60)
    ap.add_argument("--metrics-out", default="")
    args = ap.parse_args()

    prof = PROFILES[args.sampling_profile]
    args.image_scales = ([float(x) for x in args.image_scales.split(",")] if args.image_scales
                         else [1.0] * args.n_samples)
    if len(args.image_scales) != args.n_samples:
        raise SystemExit(f"[abort] --image-scales needs {args.n_samples} values")
    top_ps = [float(x) for x in args.top_ps.split(",")]
    T = args.temperature
    sampling_cfg = {   # identical keys/values to cot_gen.py, so analyze.py and resume guards apply
        "top_k": prof["top_k"], "min_p": args.min_p,
        "presence_penalty": prof["presence_penalty"],
        "repetition_penalty": args.repetition_penalty, "frequency_penalty": 0.0,
        "seed": None if args.no_request_seed else args.seed, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "kv_cache_dtype": args.kv_cache_dtype,
        "model": args.model, "dtype": "bfloat16",
    }
    if any(f != 1.0 for f in args.image_scales):      # prompt content differs -> part of the config
        sampling_cfg["image_scales"] = args.image_scales
    engine_cfg = json.loads(args.engine_cfg)
    n_srv = len(args.servers.split(","))
    engine_cfgs = engine_cfg if isinstance(engine_cfg, list) else [engine_cfg] * n_srv
    if len(engine_cfgs) != n_srv:
        raise SystemExit(f"[abort] --engine-cfg lists {len(engine_cfgs)} configs for {n_srv} servers")

    N, qs = load_questions(args)
    log(f"[sample] {len(qs)} questions ({args.sample_frac:.0%} of {N}) x {len(top_ps)} top_p "
        f"x {args.n_samples} samples @ T={T} | profile={args.sampling_profile} {sampling_cfg}")
    if args.questions_out:
        Path(args.questions_out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({q["id"]: {k: q[k] for k in ("subject", "gold", "n_options", "n_images", "ds_index")}
                   for q in qs}, open(args.questions_out, "w"), indent=2)

    # resume: one row per (id, top_p, T, sample_idx); refuse a config mismatch or a duplicate
    files = sorted(set(glob.glob(args.resume_glob) if args.resume_glob else []) | {args.out})
    have = collections.Counter()
    for rf in files:
        if not Path(rf).exists():
            continue
        for line in (gzip.open(rf, "rt") if rf.endswith(".gz") else open(rf)):
            try:
                r = json.loads(line)
            except Exception:
                continue          # a torn last line from a kill mid-write; that sample reruns
            if r.get("sampling_cfg") != sampling_cfg:
                raise SystemExit(f"[abort] {rf} holds rows with a different sampling_cfg:\n"
                                 f"  {r.get('sampling_cfg')}\n  vs {sampling_cfg}")
            have[(r["id"], r["top_p"], r["temperature"], r["sample_idx"])] += 1
    dup = [k for k, n in have.items() if n > 1]
    if dup:
        raise SystemExit(f"[abort] {len(dup)} sample(s) present more than once, e.g. {dup[:3]}")

    # work order: questions shuffled; a question's 6 x n samples are adjacent (prefix-cache
    # reuse of its images, and an interrupted run leaves whole questions), order within shuffled
    rng = random.Random(args.seed)
    order = list(range(len(qs)))
    rng.shuffle(order)
    work = []
    for j in order:
        units = [(j, p, s) for p in top_ps for s in range(args.n_samples)
                 if (qs[j]["id"], p, T, s) not in have]
        rng.shuffle(units)
        work += units
    total = len(qs) * len(top_ps) * args.n_samples
    log(f"[resume] {len(have)} samples done, {len(work)} to generate ({total} total)")
    queue = collections.deque(work)

    out_f = open(args.out, "a")
    t0 = time.time()
    stats = {"n": 0, "tok": 0, "trunc": 0, "err": 0}
    stop = asyncio.Event()

    async def one(session, eng, j, p, s):
        q = qs[j]
        body = {"model": args.model, "messages": q["messages"][args.image_scales[s]], "n": 1,
                "temperature": T, "top_p": p, "top_k": prof["top_k"], "min_p": args.min_p,
                "presence_penalty": prof["presence_penalty"],
                "repetition_penalty": args.repetition_penalty, "frequency_penalty": 0.0,
                "max_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": True},
                "return_token_ids": True}
        if not args.no_request_seed:
            body["seed"] = args.seed + s
        async with session.post(f"{eng.url}/v1/chat/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=None, sock_read=None)) as r:
            if r.status == 400:
                # the engine rejected the prompt itself (typically prompt + max_tokens exceeds the
                # context window with upscaled multi-image prompts): a permanent failure of this
                # config on this cell, recorded as a spoiled ballot rather than retried forever
                err = (await r.text())[:200]
                log(f"[reject] {q['id']} p={p} s={s}: {err}")
                return {"id": q["id"], "subject": q["subject"], "top_p": p, "temperature": T,
                        "sample_idx": s, "gold": q["gold"], "n_options": q["n_options"],
                        "out_tokens": 0, "finish_reason": "rejected", "reject_reason": err,
                        "sampling_cfg": sampling_cfg, "cfg_profile": args.sampling_profile,
                        "image_scale": args.image_scales[s], "engine_cfg": eng.cfg, "text": ""}
            if r.status != 200:
                raise RuntimeError(f"HTTP {r.status}: {(await r.text())[:300]}")
            resp = await r.json()
        c = resp["choices"][0]
        m = c["message"]
        if m.get("reasoning_content") or m.get("reasoning"):
            raise RuntimeError("server split out reasoning -- run it WITHOUT --reasoning-parser")
        return {"id": q["id"], "subject": q["subject"], "top_p": p, "temperature": T,
                "sample_idx": s, "gold": q["gold"], "n_options": q["n_options"],
                "out_tokens": len(c.get("token_ids") or []), "finish_reason": c["finish_reason"],
                "sampling_cfg": sampling_cfg, "cfg_profile": args.sampling_profile,
                "image_scale": args.image_scales[s],
                "engine_cfg": eng.cfg, "text": m.get("content") or ""}

    async def run_unit(session, eng, unit):
        try:
            row = await one(session, eng, *unit)
            if not stop.is_set():
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                stats["n"] += 1
                stats["tok"] += row["out_tokens"]
                stats["trunc"] += row["finish_reason"] != "stop"
                if stats["n"] % 500 == 0:
                    el = time.time() - t0
                    rate = stats["tok"] / el
                    left_tok = (len(queue) + sum(e.active for e in engines)) * (stats["tok"] / stats["n"])
                    log(f"[gen] {len(have) + stats['n']}/{total} samples | client avg {rate:.0f} tok/s"
                        f" | trunc {stats['trunc']} | err {stats['err']} | queued {len(queue)}"
                        f" | elapsed {el / 3600:.2f}h | eta ~{left_tok / max(rate, 1) / 3600:.1f}h")
        except Exception as ex:
            stats["err"] += 1
            queue.appendleft(unit)             # retry later
            log(f"[error] {eng.url} {qs[unit[0]]['id']} p={unit[1]} s={unit[2]}: "
                f"{type(ex).__name__}: {str(ex)[:200]}")
            await asyncio.sleep(20)            # back off while the engine restarts
        finally:
            async with eng.cond:
                eng.active -= 1
                eng.cond.notify_all()

    async def feeder(session, eng):
        pending = set()
        while not stop.is_set():
            async with eng.cond:
                await eng.cond.wait_for(lambda: eng.active < eng.target or stop.is_set())
                if stop.is_set():
                    break
                if not queue:
                    if eng.active == 0:
                        break
                    await eng.cond.wait()
                    continue
                eng.active += 1
            t = asyncio.create_task(run_unit(session, eng, queue.popleft()))
            pending.add(t)
            t.add_done_callback(pending.discard)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def wait_healthy(session, url):
        while True:
            try:
                async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            log(f"[wait] {url} not healthy yet")
            await asyncio.sleep(15)

    env_files = args.engine_env.split(",") if args.engine_env else [None] * n_srv
    engines = [Engine(u.rstrip("/"), args.inflight, args.inflight_min, args.inflight_max, c, ef)
               for u, c, ef in zip(args.servers.split(","), engine_cfgs, env_files)]
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as session:
        await asyncio.gather(*(wait_healthy(session, e.url) for e in engines))
        ctl = asyncio.create_task(control_loop(session, engines, args, stop))
        feeders = [asyncio.create_task(feeder(session, e)) for e in engines]
        if args.max_seconds:
            await asyncio.wait(feeders, timeout=args.max_seconds)
            stop.set()
            for e in engines:
                async with e.cond:
                    e.cond.notify_all()
            for t in feeders:
                t.cancel()
        else:
            await asyncio.gather(*feeders)
        stop.set()
        ctl.cancel()
    el = time.time() - t0
    log(f"[done] {stats['n']} samples, {stats['tok']} tokens in {el / 3600:.2f}h "
        f"({stats['tok'] / el:.0f} tok/s client-side), trunc {stats['trunc']}, err {stats['err']}")


if __name__ == "__main__":
    asyncio.run(main())
