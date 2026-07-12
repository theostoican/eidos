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
import argparse, ast, base64, collections, io, json, random, re, time
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
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20, help="Qwen3.5 thinking default (was -1 in v1)")
    ap.add_argument("--presence-penalty", type=float, default=1.5,
                    help="Qwen3.5 thinking default; suppresses repetition loops (was 0 in v1)")
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
    ap.add_argument("--questions-out", default="outputs/questions_5pct.json")
    ap.add_argument("--out", default="outputs/cot_gen.jsonl")
    args = ap.parse_args()
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

    print("[load] MMMU/MMMU_Pro standard (10 options) test ...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    N = len(ds)
    k = max(1, round(args.sample_frac * N))
    rng = random.Random(args.sample_seed)
    sel = sorted(rng.sample(range(N), k))
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
    qconvs, qinfo, qmeta = [], [], {}
    for i in sel:
        row = ds[i]
        opts = load_options(row)
        images = collect_images(row)
        content = build_content(row["question"], opts, images)
        qmeta[row["id"]] = {"subject": row.get("subject"), "gold": row["answer"],
                            "n_options": len(opts), "n_images": len(images), "ds_index": i}
        qconvs.append([{"role": "user", "content": content}])
        qinfo.append({"id": row["id"], "subject": row.get("subject"), "gold": row["answer"],
                      "n_options": len(opts)})

    Path(questions_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(qmeta, open(questions_out, "w"), indent=2)
    print(f"[sample] wrote {len(qmeta)} question meta -> {questions_out} | "
          f"grid = {len(top_ps)} top_p x {len(temps)} temp", flush=True)

    # resume: which (id, top_p, temp) cells are already complete (all n_samples present)?
    done_cells = set()
    if args.resume and Path(out_path).exists():
        cnt = collections.Counter()
        for line in open(out_path):
            try:
                r = json.loads(line)
            except Exception:
                continue
            cnt[(r["id"], r["top_p"], r.get("temperature", args.temperature))] += 1
        done_cells = {c for c, n in cnt.items() if n >= args.n_samples}
        print(f"[resume] {len(done_cells)} complete cells already in {out_path}; will skip them",
              flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              kv_cache_dtype=args.kv_cache_dtype,
              tensor_parallel_size=args.tensor_parallel_size,
              limit_mm_per_prompt={"image": 8, "video": 0}, trust_remote_code=True,
              seed=args.seed)

    t0 = time.time()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    n_written = n_trunc_total = n_skipped = 0
    with open(out_path, "a" if args.resume else "w") as f:
        for T in temps:                       # temp-outer, top_p-inner (a temperature's FULL
            for p in top_ps:                  # top_p sweep completes before the next temperature,
                                              # so the inverted-U-vs-p can be checked per temp)
                idxs = [j for j, qi in enumerate(qinfo) if (qi["id"], p, T) not in done_cells]
                n_skipped += sum(1 for qi in qinfo if (qi["id"], p, T) in done_cells)
                if not idxs:
                    print(f"[gen] top_p={p} temp={T} | all cells already done, skipping", flush=True)
                    continue
                sp = SamplingParams(n=args.n_samples, temperature=T, top_p=p,
                                    top_k=args.top_k, min_p=args.min_p,
                                    presence_penalty=args.presence_penalty,
                                    repetition_penalty=args.repetition_penalty,
                                    max_tokens=args.max_tokens, seed=args.seed)
                tc = time.time()
                outs = llm.chat([qconvs[j] for j in idxs], sp,
                                chat_template_kwargs={"enable_thinking": True})
                gen_tok = 0
                for j, o in zip(idxs, outs):
                    qi = qinfo[j]
                    for s_idx, comp in enumerate(o.outputs):
                        ntok = len(comp.token_ids)
                        gen_tok += ntok
                        f.write(json.dumps({
                            "id": qi["id"], "subject": qi["subject"], "top_p": p, "temperature": T,
                            "sample_idx": s_idx, "gold": qi["gold"], "n_options": qi["n_options"],
                            "out_tokens": ntok, "finish_reason": comp.finish_reason,
                            "text": comp.text,
                        }) + "\n")
                        n_written += 1
                f.flush()
                dt_chunk = time.time() - tc
                n_trunc = sum(1 for j, o in zip(idxs, outs) for c in o.outputs if c.finish_reason != "stop")
                n_trunc_total += n_trunc
                print(f"[gen] top_p={p} temp={T} | {len(idxs)}x{args.n_samples} gens | trunc={n_trunc} | "
                      f"chunk={dt_chunk/60:.1f}m | decode={gen_tok/dt_chunk:.0f} tok/s | "
                      f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    dt = time.time() - t0
    summary = {"model": args.model, "n_questions": len(qmeta), "top_ps": top_ps, "temps": temps,
               "n_samples": args.n_samples, "n_generations": n_written,
               "resumed_skipped": n_skipped, "truncated": n_trunc_total,
               "num_shards": args.num_shards, "shard_id": args.shard_id,
               "gen_seconds": round(dt, 1),
               "max_tokens": args.max_tokens, "max_model_len": args.max_model_len}
    json.dump(summary, open(out_path.replace(".jsonl", "_summary.json"), "w"), indent=2)
    print(f"[done] {dt/60:.1f}m | {n_written} new gens ({n_trunc_total} trunc, "
          f"{n_skipped} skipped) -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
