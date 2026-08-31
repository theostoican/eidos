#!/usr/bin/env python
"""CoT-diversity sweep on a 5% sample of MMMU-Pro (standard, 10 options).

Same nucleus-diversity design as the premise experiment, but:
  - the OFFICIAL MMMU-Pro prompt is used (answer the MC question, think step by step),
  - we keep the FULL generation (the <think> CoT + the final 'Answer: X'),
  - sweep top_p {0.5,0.7,0.9,0.95,1.0} x 16 samples, thinking ON, temp 1.0, top_k off,
    40960-token budget (max_model_len 49152) -- "the model params the same".

Downstream: extract_cot.py pulls the <think> CoT; analyze_cot.py embeds it (Vendi +
cosine), a vision judge rules each CoT's reasoning sound/unsound, and the parsed
'Answer: X' gives per-sample final-answer accuracy. All three are tracked vs top_p.

The 5% question sample is chosen deterministically (--sample-seed) and written to
outputs/questions_5pct.json so judging/analysis reference the exact same set.
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
# In the official infer code (mmmu-pro/infer/infer_transformers.py :: construct_prompt) this
# string is a SUFFIX: prompt = f"{question}\n{parsed_options}\n{cot.standard}", with each
# "<image N>" marker replaced by the literal text "[image]" and the actual images appended
# AFTER all text, in the order they were referenced (origin_mmmu_doc_to_visual).
PROMPT_STANDARD = (
    "Answer the preceding multiple choice question. The last line of your response should "
    "be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
    "options. Think step by step before answering."
)

IMG_MARK = re.compile(r"<image\s+(\d+)>")

def parse_options(options):
    """Official parse_options: 'A. <opt>\\nB. <opt>\\n...' for len(options) letters."""
    return "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))

def build_content(question, options, images):
    """OpenAI-style content list matching the OFFICIAL MMMU-Pro standard assembly exactly:
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
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--sample-frac", type=float, default=0.05)
    ap.add_argument("--sample-seed", type=int, default=20260706)
    # NESTED SAMPLING. A plain sample() at a larger --sample-frac is NOT a superset of the
    # smaller one (CPython's sample() is not prefix-stable in k), so scaling 5% -> 20% would
    # discard every committed trace and leave the two runs non-comparable question-for-question.
    # --nest-from draws the smaller fraction FIRST with the identical call, then tops up from
    # the complement on a derived seed: the pilot's questions are retained exactly, its cells
    # stay reusable via --resume, and its numbers become a literal subset of the new run's.
    ap.add_argument("--nest-from", type=float, default=0.0,
                    help="draw this fraction FIRST with --sample-seed, then top up to "
                         "--sample-frac, making the question set a strict SUPERSET of the "
                         "smaller run's (0 = plain independent sample)")
    ap.add_argument("--limit", type=int, default=0, help="calibration: cap #questions AFTER sampling (0=all)")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--top-ps", default="0.5,0.7,0.9,0.95,1.0")
    ap.add_argument("--temps", default="",
                    help="comma-sep temperatures to SWEEP (2D grid over top_ps x temps). "
                         "Empty = single --temperature. Generation order is top_p-outer, temp-inner.")
    # Qwen3.5-9B RECOMMENDED thinking-mode sampling defaults (from the model card, general
    # tasks): temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5,
    # repetition_penalty=1.0. top_p is the SWEPT variable here; the rest use the recommended
    # values. presence_penalty=1.5 is what prevents the low-top_p repetition-loop degeneration
    # (the earlier run used top_k=-1 / presence_penalty=0 and got 11% truncation at top_p=0.5).
    # HANDOFF 2.1.1(a): the old argparse defaults (top_k=20, presence_penalty=1.5) were a
    # loaded gun -- running cot_gen.py without the full flag list silently produced data in
    # the config that erases the effect, which is how a whole run was invalidated. There is
    # now NO default: --sampling-profile must be stated explicitly, so the choice is always
    # deliberate and always recorded. Explicit --top-k / --presence-penalty still override.
    PROFILES = {"neutral": {"top_k": -1, "presence_penalty": 0.0},
                "qwen-recommended": {"top_k": 20, "presence_penalty": 1.5}}
    ap.add_argument("--sampling-profile", required=True, choices=sorted(PROFILES),
                    help="neutral = HANDOFF 2.1 (top_k=-1, presence_penalty=0), the config the "
                         "committed cots/ use. qwen-recommended = the model card's thinking "
                         "defaults (top_k=20, presence_penalty=1.5).")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None, help="override the profile's top_k")
    ap.add_argument("--presence-penalty", type=float, default=None,
                    help="override the profile's presence_penalty")
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=40960)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--gpu-mem-util", type=float, default=0.93)
    ap.add_argument("--max-num-seqs", type=int, default=8,
                    help="RAISE on a big GPU (e.g. 64-256); 8 is a 24GB-4090 setting")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="set = #GPUs for a multi-GPU node (TP). For data-parallel, run N")
    # fp8 KV cache: bf16 KV is 128KB/token -> one 49k-token seq = 6.4GB, which does not
    # fit alongside 18GB of bf16 weights on a 24GB card. fp8 halves KV (weights stay bf16).
    # ON A LARGER GPU (>=48GB) set --kv-cache-dtype auto to keep params EXACT (bf16 KV).
    ap.add_argument("--kv-cache-dtype", default="fp8", help="'auto'(bf16) or 'fp8'")
    ap.add_argument("--chunk-questions", type=int, default=0,
                    help="questions per llm.chat call; each chunk covers ALL top_p for those "
                         "questions, so an interrupted run still leaves a balanced dataset. "
                         "0 = one single call (max throughput, no intermediate checkpoint).")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0,
                    help="0 = vLLM default. Raise (8192-16384) to speed prefill on a big GPU.")
    # ENGINE-ONLY, never a sampling parameter: vLLM's async scheduler crashes on Kimi-VL
    # under TP ("KeyError: <req_id>" in scheduler.update_from_output via
    # step_with_batch_queue). Off by default so every existing command -- and the committed
    # Qwen results -- reproduce exactly; it changes scheduling, not what is generated.
    ap.add_argument("--disable-async-scheduling", action="store_true",
                    help="run the synchronous scheduler (workaround for a vLLM batch-queue "
                         "KeyError crash seen with Kimi-VL + tensor parallel)")
    ap.add_argument("--seed", type=int, default=1234)
    # data-parallel sharding for a multi-GPU NODE: launch N processes, each pinned to one
    # GPU via CUDA_VISIBLE_DEVICES=i, with --num-shards N --shard-id i. Each takes a strided
    # slice sel[i::N] (interleaved so easy/hard questions balance across GPUs) and writes to
    # its own <out>.shard{i}.jsonl / <questions-out>.shard{i}.json. Merge with `cat` after.
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    # resume: skip (id, top_p) cells already fully present (n_samples rows) in --out, and
    # APPEND rather than overwrite. Lets a long node run survive restarts/preemption.
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--resume-glob", default="",
                    help="also treat cells complete in these files as done (e.g. "
                         "'outputs/nd_gen.shard*.jsonl'), so work can be re-sharded mid-run.")
    ap.add_argument("--questions-out", default="outputs/questions_5pct.json")
    ap.add_argument("--out", default="outputs/cot_gen.jsonl")
    args = ap.parse_args()
    prof = PROFILES[args.sampling_profile]
    if args.top_k is None:
        args.top_k = prof["top_k"]
    if args.presence_penalty is None:
        args.presence_penalty = prof["presence_penalty"]
    top_ps = [float(x) for x in args.top_ps.split(",")]
    temps = [float(x) for x in args.temps.split(",")] if args.temps else [args.temperature]

    def shard_path(path):
        if args.num_shards <= 1:
            return path
        stem, dot, ext = path.rpartition(".")
        return f"{stem}.shard{args.shard_id}.{ext}" if dot else f"{path}.shard{args.shard_id}"
    out_path, questions_out = shard_path(args.out), shard_path(args.questions_out)

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
        # identical call to the --nest-from run, so its exact question set is reproduced
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
    if args.num_shards > 1:                 # data-parallel: strided slice for this GPU
        sel = sel[args.shard_id::args.num_shards]
    print(f"[sample] {len(sel)}/{full_k} questions this shard "
          f"(shard {args.shard_id}/{args.num_shards}, {args.sample_frac:.0%} of {N}, "
          f"seed={args.sample_seed})" + (f" [LIMIT {args.limit}]" if args.limit else ""), flush=True)

    # Build the conversation ONCE per question (prompt is identical across top_p/temp; only
    # sampling params differ). The 2D grid is applied in the generation loop below.
    # build_content appends ONE image per <image N> reference, and references may repeat
    # (image_order, per the official assembly). A question that references its images more
    # than MAX_IMAGES times therefore exceeds the engine's limit_mm_per_prompt and makes
    # llm.chat() raise, killing the whole chunk -- and on restart it fails identically, so
    # supervisor's autorestart turns it into a crash loop. Such a question cannot be
    # represented under this engine config at all (35 images is ~35k prompt tokens, which
    # will not fit alongside max_tokens=40960 in max_model_len=49152 either), so it is
    # SKIPPED and named in the log. De-duplicating instead would silently alter prompt
    # construction for every question with repeated references -- including pilot questions
    # whose 0.6/0.8/0.99 cells are already generated -- leaving one question's cells built
    # two different ways across the swept axis. An excluded question is honest; a silently
    # re-rendered one is not.
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
        qconvs.append([{"role": "user", "content": content}])
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
    # --resume-glob lets a process see cells finished by ANY shard file, not just its own.
    # That makes the run re-shardable mid-flight: the question->shard map is fixed at launch,
    # so if one GPU draws a slice of long traces it finishes hours after the other. With
    # cell-level resume you can kill both, relaunch with a different split over the SAME
    # question set, and only the outstanding cells get generated. Concurrent processes must
    # still hold disjoint question sets or they will duplicate a cell (-> 32 samples).
    done_cells = set()
    resume_files = sorted(glob.glob(args.resume_glob)) if args.resume_glob else []
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

    # HANDOFF 2.1.1(c): the sampling config must live IN the data, not only in prose.
    # Every row carries the complete config so any analysis can assert on it, and so
    # rows generated under different configs can never be silently interleaved.
    sampling_cfg = {
        "top_k": args.top_k, "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
        "frequency_penalty": 0.0,
        "seed": args.seed, "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len, "kv_cache_dtype": args.kv_cache_dtype,
        "model": args.model, "dtype": "bfloat16",
    }
    cfg_tag = ("qwen-recommended" if (args.top_k == 20 and args.presence_penalty == 1.5)
               else "neutral" if (args.top_k == -1 and args.presence_penalty == 0.0)
               else "custom")
    print(f"[config] profile={cfg_tag} {sampling_cfg}", flush=True)

    # HANDOFF 2.1.1(b): --resume keys only on (id, top_p, temperature). Refuse to append
    # rows generated under a different sampling config into an existing file.
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
    if args.max_num_batched_tokens:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.disable_async_scheduling:
        engine_kwargs["async_scheduling"] = False
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              kv_cache_dtype=args.kv_cache_dtype,
              tensor_parallel_size=args.tensor_parallel_size,
              limit_mm_per_prompt={"image": MAX_IMAGES, "video": 0}, trust_remote_code=True,
              enable_prefix_caching=True, disable_log_stats=False,
              seed=args.seed, **engine_kwargs)

    # THROUGHPUT: schedule the whole (question x top_p x temp) grid as ONE shuffled work
    # queue rather than one batch per top_p. A per-top_p batch ends only when its slowest
    # sequence ends, so every batch pays a long low-concurrency tail (measured: ~520 tok/s
    # on an A100 for a 4-question batch, vs the GPU being able to run ~130 sequences at
    # once). Mixing top_p into large chunks keeps the scheduler saturated and pays the
    # tail once. This changes NOTHING about what is generated: sampling params are still
    # per-request, so each cell gets exactly the params it would have gotten.
    # Chunk by QUESTION BLOCK, not by top_p: a chunk holds every top_p for a subset of
    # questions. So whenever the run is interrupted, what is on disk is a COMPLETE,
    # BALANCED dataset over the questions finished so far (every question present at
    # every top_p with all 16 samples) rather than a partial grid that the balancing
    # step would have to throw away. Within a chunk the cells are shuffled so all top_p
    # are in flight together and no batch degenerates to a single-top_p tail.
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
          f"in {len(chunks)} chunk(s) of <={CH} requests x {args.n_samples} samples", flush=True)

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
            outs = llm.chat([qconvs[j] for (j, _, _) in chunk], sps,
                            chat_template_kwargs={"enable_thinking": True})
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
               "cfg_profile": cfg_tag, "sampling_cfg": sampling_cfg,
               "temperature": args.temperature}
    json.dump(summary, open(out_path.replace(".jsonl", "_summary.json"), "w"), indent=2)
    print(f"[done] {dt/60:.1f}m | {n_written} new gens ({n_trunc_total} trunc, "
          f"{n_skipped} skipped) -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
