#!/usr/bin/env python
"""top_p sweep on MMMU-Pro (standard, 10 options) with InternVL3.5-8B at T=1.2.

A second-model arm for the Qwen3.5-9B top_p study (theostoican/eidos,
mmmu_pro/qwen35_9b_cot_diversity). Same design, same prompt assembly, same ballot rule;
what differs is recorded in the data, not in prose:

  - model     OpenGVLab/InternVL3_5-8B  (InternViT-300M + Qwen3-8B; MMMU-Pro unpublished,
              measured here on the same questions as the Qwen arm)
  - T         1.2  -- above 1.0, where top_p truncation does real work (the Qwen arms
              showed no interior optimum at 1.0 and a clear one at 1.6)
  - top_k     -1 (off). The model card ships no sampling defaults; top_p is the ONLY
              truncation knob so the swept axis is not confounded.
  - samples   8 per (question, top_p) cell (the Qwen arms used 16)
  - questions 10% of MMMU-Pro, NESTED from the 5% seed so the Qwen T=1.6 arm's 86-question set is
              retained exactly; it also happens to be a strict subset of the 20% T=1.0 set.

THINKING MODE. InternVL3.5 has no chat-template `enable_thinking` switch. Thinking is
enabled by the model card's R1_SYSTEM_PROMPT (verbatim below), which makes the model wrap
its reasoning in <think>...</think> -- the same tags analyze.py's parser already splits on.
The card recommends T=0.6 in thinking mode "to mitigate undesired repetition"; T=1.2 is
deliberately above that, so a repetition-loop truncation arm at low top_p is expected and
is counted as spoiled ballots, never dropped.

CONTEXT. max_tokens=40960, the SAME nominal cap as the Qwen arm. max_model_len is 40960, the
LLM's declared context (the Qwen arm used 49152; here that value crashes the engine with a
device-side assert when a trace crosses position 40960, because the rotary table is sized to
it). Consequence: a trace whose prompt+output would exceed 40960 is ended by vLLM at the
model length, i.e. 1-8k tokens before the nominal cap, with finish_reason="length" -- the
same spoiled ballot it would have been at 40960. The effective cap differs from Qwen's only
on that already-spoiled tail.

IMAGES. The official MMMU-Pro assembly (text first, images appended in reference order,
"<image N>" -> "[image]") is kept byte-for-byte on the text side. vLLM inserts InternVL's
<image> placeholders per image; tiling is the model's own config (max_dynamic_patch=12
+ thumbnail, up to 3328 tokens/image) applied identically to every image at every top_p.
"""
import argparse, ast, base64, collections, glob, gzip, io, json, random, re, time
from pathlib import Path

def b64_image(img, fmt="PNG"):
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

LETTERS = [chr(ord("A") + i) for i in range(26)]

# --- OFFICIAL MMMU-Pro CoT prompt, verbatim from the paper's repo ---
# MMMU-Benchmark/MMMU  mmmu-pro/prompts.yaml  ->  cot.standard  (fetched, not paraphrased).
# In the official infer code this string is a SUFFIX: prompt = f"{question}\n{options}\n{cot}",
# with each "<image N>" marker replaced by the literal text "[image]" and the actual images
# appended AFTER all text, in the order they were referenced.
PROMPT_STANDARD = (
    "Answer the preceding multiple choice question. The last line of your response should "
    "be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
    "options. Think step by step before answering."
)

# --- InternVL3.5 THINKING-MODE system prompt, verbatim from the model card ---
# (OpenGVLab/InternVL3_5-8B README "Thinking Mode"; VLMEvalKit R1_SYSTEM_PROMPT). This is
# the ONLY way to enable thinking for this model: there is no chat_template kwarg.
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()

IMG_MARK = re.compile(r"<image\s+(\d+)>")

def parse_options(options):
    """Official parse_options: 'A. <opt>\\nB. <opt>\\n...' for len(options) letters."""
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))

def build_content(question, options, images):
    """OpenAI-style content list matching the OFFICIAL MMMU-Pro standard assembly:
    all text first (question + options + the suffix prompt, with <image N> -> '[image]'),
    then the referenced images appended after the text in reference order."""
    text = f"{question}\n{parse_options(options)}\n{PROMPT_STANDARD}"
    order = [int(n) for n in IMG_MARK.findall(text)]      # image_order (may repeat)
    text = IMG_MARK.sub("[image]", text)                  # replace_images_tokens
    content = [{"type": "text", "text": text}]
    for idx in order:                                     # append referenced images, in order
        if idx in images:
            content.append({"type": "image_url", "image_url": {"url": b64_image(images[idx])}})
    return content

def load_options(row):
    return ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]

def collect_images(row):
    images = {}
    for i in range(1, 8):
        im = row.get(f"image_{i}")
        if im is not None:
            images[i] = im
    return images

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--sample-frac", type=float, default=0.10)
    ap.add_argument("--sample-seed", type=int, default=20260706)
    # NESTED SAMPLING (unchanged from the Qwen arm). A plain sample() at a larger fraction is
    # NOT a superset of a smaller one (CPython's sample() is not prefix-stable in k), so the
    # smaller fraction is drawn FIRST with the identical call, then topped up from the
    # complement on a derived seed. With --nest-from 0.05 the Qwen T=1.6 arm's 86-question set is kept
    # exactly (and the result is also a strict subset of the 345-question T=1.0 set).
    ap.add_argument("--nest-from", type=float, default=0.05,
                    help="draw this fraction FIRST with --sample-seed, then top up to "
                         "--sample-frac (0 = plain independent sample)")
    ap.add_argument("--limit", type=int, default=0, help="calibration: cap #questions AFTER sampling (0=all)")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--top-ps", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--temps", default="",
                    help="comma-sep temperatures to SWEEP (2D grid over top_ps x temps). "
                         "Empty = single --temperature.")
    # No argparse default for the flags that decide the result: --sampling-profile must be
    # stated, so the choice is always deliberate and always recorded in every row.
    PROFILES = {"neutral": {"top_k": -1, "presence_penalty": 0.0}}
    ap.add_argument("--sampling-profile", required=True, choices=sorted(PROFILES),
                    help="neutral = top_k off, no presence penalty: top_p is the only "
                         "truncation knob. Explicit --top-k / --presence-penalty override.")
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top-k", type=int, default=None, help="override the profile's top_k")
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="override the profile's presence_penalty")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=40960)
    ap.add_argument("--max-model-len", type=int, default=40960,
                    help="the LLM's declared context. 49152 (the Qwen arm) crashes the engine with a "
                         "device-side assert as soon as a trace crosses position 40960.")
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    # bf16 KV is 144KB/token for this LLM (36 layers x 8 KV heads x 128). On a 32GB card
    # ~12GB is left for KV after weights -> ~85k tokens in flight. 'auto' keeps the sampling
    # numerics exact (the committed Qwen data used auto); fp8 doubles concurrency.
    ap.add_argument("--kv-cache-dtype", default="auto", help="'auto'(bf16) or 'fp8'")
    ap.add_argument("--chunk-questions", type=int, default=8,
                    help="questions per llm.chat call; each chunk covers ALL top_p for those "
                         "questions, so an interrupted run still leaves a balanced dataset.")
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=1234)
    # N-GRAM SPECULATIVE DECODING (engine mechanism, NOT a sampling parameter). vLLM verifies
    # drafted tokens by rejection sampling against the target model's own top-p/top-k
    # distribution, so the output distribution is unchanged; only wall-clock changes. It pays
    # most on repetition loops -- exactly the capped traces that monopolise the KV cache.
    # Recorded per row under "engine_cfg", separate from "sampling_cfg", so the resume guard
    # (which compares sampling_cfg) still allows mixing rows generated with and without it.
    ap.add_argument("--spec-ngram", type=int, default=0,
                    help="num_speculative_tokens for ngram speculation (0 = off)")
    ap.add_argument("--spec-method", default="ngram", choices=["ngram", "ngram_gpu"],
                    help="ngram = CPU draft search over the full history (slow at 40k contexts); "
                         "ngram_gpu = same proposer on the GPU")
    ap.add_argument("--spec-lookup-min", type=int, default=2)
    ap.add_argument("--spec-lookup-max", type=int, default=4)
    # data-parallel sharding: launch N processes, each pinned to one GPU via
    # CUDA_VISIBLE_DEVICES=i, with --num-shards N --shard-id i. Each takes a strided slice
    # sel[i::N] and writes to its own <out>.shard{i}.jsonl. Merge with `cat` after.
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-glob", default="",
                    help="also treat cells complete in these files as done; comma-separated globs")
    ap.add_argument("--questions-out", default="outputs/ivl35_t12_q.json")
    ap.add_argument("--out", default="outputs/ivl35_t12.jsonl")
    args = ap.parse_args()
    prof = PROFILES[args.sampling_profile]
    if args.top_k is None:
        args.top_k = prof["top_k"]
    if args.presence_penalty is None:
        args.presence_penalty = prof["presence_penalty"]
    top_ps = [float(x) for x in args.top_ps.split(",")]
    temps = [float(x) for x in args.temps.split(",")] if args.temps else [args.temperature]

    def shard_path(path):
        # Always suffix, even for a single shard. Returning the bare path when num_shards==1
        # (the TP=4 case, where NSHARDS=4/TP=1) silently breaks everything that globs
        # "<out>.shard*.jsonl": --resume finds no completed cells and REGENERATES the whole arm,
        # appending duplicate cells that analyze.py then refuses. Observed 2026-09-09.
        stem, dot, ext = path.rpartition(".")
        return f"{stem}.shard{args.shard_id}.{ext}" if dot else f"{path}.shard{args.shard_id}"
    out_path, questions_out = shard_path(args.out), shard_path(args.questions_out)

    if args.max_model_len > 40960:
        raise SystemExit("[abort] max_model_len > 40960: InternVL3.5-8B's rotary table is sized to its "
                         "declared 40960 context; the engine dies with a CUDA device-side assert when a "
                         "trace crosses it (observed 2026-09-07). Keep max_model_len=40960.")
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    MAX_IMAGES = 8          # must match limit_mm_per_prompt passed to LLM() below

    print("[load] MMMU/MMMU_Pro standard (10 options) test ...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    N = len(ds)
    k = max(1, round(args.sample_frac * N))
    if args.nest_from:
        k0 = max(1, round(args.nest_from * N))
        if k0 > k:
            raise SystemExit(f"[abort] --nest-from {args.nest_from} selects {k0} questions, "
                             f"more than --sample-frac {args.sample_frac} selects ({k}).")
        base = random.Random(args.sample_seed).sample(range(N), k0)
        seen = set(base)
        pool = [i for i in range(N) if i not in seen]
        sel = sorted(base + random.Random(args.sample_seed + 1).sample(pool, k - k0))
        print(f"[sample] nested: {k0} questions retained from the "
              f"{args.nest_from:.0%} run + {k - k0} new = {k}", flush=True)
    else:
        sel = sorted(random.Random(args.sample_seed).sample(range(N), k))
    if args.limit:
        sel = sel[:args.limit]
    full_k = len(sel)
    if args.num_shards > 1:
        sel = sel[args.shard_id::args.num_shards]
    print(f"[sample] {len(sel)}/{full_k} questions this shard "
          f"(shard {args.shard_id}/{args.num_shards}, {args.sample_frac:.0%} of {N}, "
          f"seed={args.sample_seed})" + (f" [LIMIT {args.limit}]" if args.limit else ""), flush=True)

    # Build the conversation ONCE per question. A question whose image references exceed
    # MAX_IMAGES cannot be represented under this engine config (test_Geography_252 references
    # 35 images) and is SKIPPED and named, never silently re-rendered.
    qconvs, qinfo, qmeta = [], [], {}
    over_limit = []
    for i in sel:
        row = ds[i]
        opts = load_options(row)
        images = collect_images(row)
        content = build_content(row["question"], opts, images)
        n_img = sum(1 for c in content if c.get("type") == "image_url")
        if n_img > MAX_IMAGES:
            over_limit.append((row["id"], n_img))
            continue
        qmeta[row["id"]] = {"subject": row.get("subject"), "gold": row["answer"],
                            "n_options": len(opts), "n_images": len(images), "ds_index": i}
        qconvs.append([{"role": "system", "content": R1_SYSTEM_PROMPT},
                       {"role": "user", "content": content}])
        qinfo.append({"id": row["id"], "subject": row.get("subject"), "gold": row["answer"],
                      "n_options": len(opts)})

    if over_limit:
        print(f"[skip] {len(over_limit)} question(s) exceed limit_mm_per_prompt="
              f"{MAX_IMAGES} and are EXCLUDED from this run: "
              + ", ".join(f"{qid} ({n} images)" for qid, n in over_limit), flush=True)

    Path(questions_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(qmeta, open(questions_out, "w"), indent=2)
    print(f"[sample] wrote {len(qmeta)} question meta -> {questions_out} | "
          f"grid = {len(top_ps)} top_p x {len(temps)} temp", flush=True)

    # resume: which (id, top_p, temp) cells are already complete (all n_samples present)?
    done_cells = set()
    # comma-separated patterns (glob.glob has no brace expansion): e.g.
    # "outputs/x.shard*.jsonl,cots/completed_*.jsonl.gz"
    resume_files = sorted({f for pat in args.resume_glob.split(",") if pat.strip()
                           for f in glob.glob(pat.strip())}) if args.resume_glob else []
    if args.resume and Path(out_path).exists() and out_path not in resume_files:
        resume_files.append(out_path)
    if args.resume and resume_files:
        cnt = collections.Counter()
        for rf in resume_files:
            for line in (gzip.open(rf, "rt") if rf.endswith(".gz") else open(rf)):
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cnt[(r["id"], r["top_p"], r.get("temperature", args.temperature))] += 1
        done_cells = {c for c, n in cnt.items() if n >= args.n_samples}
        print(f"[resume] {len(done_cells)} complete cells across {len(resume_files)} file(s); "
              f"will skip them", flush=True)

    # The sampling config lives IN the data. Every row carries it, so rows generated under
    # different configs (or a different model) can never be silently interleaved.
    sampling_cfg = {
        "top_k": args.top_k, "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": 0.0,
        "seed": args.seed, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "kv_cache_dtype": args.kv_cache_dtype,
        "model": args.model, "dtype": "bfloat16",
        "thinking": "r1_system_prompt",          # how thinking was enabled for this model
        "max_dynamic_patch": 12, "use_thumbnail": True,   # the model's own tiling config
    }
    cfg_tag = ("neutral" if (args.top_k == -1 and args.presence_penalty == 0.0) else "custom")
    print(f"[config] profile={cfg_tag} {sampling_cfg}", flush=True)

    # --resume keys only on (id, top_p, temperature). Refuse to append rows generated under
    # a different sampling config into an existing file.
    for rf in (resume_files if args.resume else []):
        for line in (gzip.open(rf, "rt") if rf.endswith(".gz") else open(rf)):
            try:
                prev = json.loads(line).get("sampling_cfg")
            except Exception:
                continue
            if prev is not None and prev != sampling_cfg:
                raise SystemExit(
                    f"[abort] {rf} was generated with a DIFFERENT sampling config:\n"
                    f"  existing: {prev}\n  this run: {sampling_cfg}\n"
                    f"Appending/merging would interleave incomparable traces. Use a new --out.")
            break

    engine_kwargs = {}
    engine_cfg = {"spec_decode": None}
    if args.spec_ngram:
        engine_kwargs["speculative_config"] = {"method": args.spec_method,
                                              "num_speculative_tokens": args.spec_ngram,
                                              "prompt_lookup_min": args.spec_lookup_min,
                                              "prompt_lookup_max": args.spec_lookup_max}
        engine_cfg["spec_decode"] = dict(engine_kwargs["speculative_config"])
    print(f"[engine] {engine_cfg}", flush=True)
    if args.max_num_batched_tokens:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              kv_cache_dtype=args.kv_cache_dtype,
              tensor_parallel_size=args.tensor_parallel_size,
              limit_mm_per_prompt={"image": MAX_IMAGES, "video": 0}, trust_remote_code=True,
              enable_prefix_caching=True, disable_log_stats=False,
              seed=args.seed, **engine_kwargs)

    # One shuffled work queue over (question x top_p x temp), chunked by QUESTION BLOCK so
    # an interrupted run leaves a complete, balanced dataset over the questions finished
    # so far. Sampling params are per-request, so each cell gets exactly its own params.
    qorder = list(range(len(qinfo)))
    random.Random(args.seed).shuffle(qorder)
    per_chunk = max(1, args.chunk_questions or len(qorder))
    chunks = []
    for i in range(0, len(qorder), per_chunk):
        blk = [(j, p, T) for j in qorder[i:i + per_chunk] for T in temps for p in top_ps
               if (qinfo[j]["id"], p, T) not in done_cells]
        random.Random(args.seed + i).shuffle(blk)
        if blk:
            chunks.append(blk)
    work = [c for blk in chunks for c in blk]
    n_skipped = len(qinfo) * len(top_ps) * len(temps) - len(work)
    CH = per_chunk
    print(f"[gen] {len(work)} cells to generate ({n_skipped} already done) "
          f"in {len(chunks)} chunk(s) of <={CH} questions x {len(top_ps)} top_p "
          f"x {args.n_samples} samples", flush=True)

    t0 = time.time()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = n_trunc_total = 0
    tok_total = 0
    with open(out_path, "a" if args.resume else "w") as f:
        for ci, chunk in enumerate(chunks):
            sps = [SamplingParams(n=args.n_samples, temperature=T, top_p=p,
                                  top_k=args.top_k, min_p=args.min_p,
                                  presence_penalty=args.presence_penalty,
                                  repetition_penalty=args.repetition_penalty,
                                  max_tokens=args.max_tokens, seed=args.seed)
                   for (_, p, T) in chunk]
            tc = time.time()
            # No chat_template_kwargs: thinking is the system message in qconvs.
            outs = llm.chat([qconvs[j] for (j, _, _) in chunk], sps)
            gen_tok = n_trunc = 0
            for (j, p, T), o in zip(chunk, outs):
                qi = qinfo[j]
                for s_idx, comp in enumerate(o.outputs):
                    ntok = len(comp.token_ids)
                    gen_tok += ntok
                    n_trunc += comp.finish_reason != "stop"
                    f.write(json.dumps({
                        "id": qi["id"], "subject": qi["subject"], "top_p": p, "temperature": T,
                        "sample_idx": s_idx, "gold": qi["gold"], "n_options": qi["n_options"],
                        "out_tokens": ntok, "finish_reason": comp.finish_reason,
                        "sampling_cfg": sampling_cfg, "cfg_profile": cfg_tag,
                        "engine_cfg": engine_cfg,
                        "text": comp.text,
                    }) + "\n")
                    n_written += 1
            f.flush()
            dt_chunk = time.time() - tc
            n_trunc_total += n_trunc
            tok_total += gen_tok
            el = time.time() - t0
            eta = el / (ci + 1) * (len(chunks) - ci - 1)
            print(f"[gen] chunk {ci+1}/{len(chunks)} | {len(chunk)}x{args.n_samples} gens | "
                  f"trunc={n_trunc} | chunk={dt_chunk/60:.1f}m | decode={gen_tok/dt_chunk:.0f} tok/s "
                  f"| avg={tok_total/el:.0f} tok/s | elapsed={el/60:.1f}m | eta={eta/60:.1f}m",
                  flush=True)
    dt = time.time() - t0
    summary = {"model": args.model, "n_questions": len(qmeta), "top_ps": top_ps, "temps": temps,
               "n_samples": args.n_samples, "n_generations": n_written,
               "resumed_skipped": n_skipped, "truncated": n_trunc_total,
               "num_shards": args.num_shards, "shard_id": args.shard_id,
               "gen_seconds": round(dt, 1),
               "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
               "cfg_profile": cfg_tag, "sampling_cfg": sampling_cfg, "engine_cfg": engine_cfg,
               "temperature": args.temperature}
    json.dump(summary, open(out_path.replace(".jsonl", "_summary.json"), "w"), indent=2)
    print(f"[done] {dt/60:.1f}m | {n_written} new gens ({n_trunc_total} trunc, "
          f"{n_skipped} skipped) -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
