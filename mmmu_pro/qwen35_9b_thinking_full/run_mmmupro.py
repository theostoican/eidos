#!/usr/bin/env python
"""
MMMU-Pro (standard, 10 options) inference pipeline for Qwen/Qwen3.5-9B.

- vLLM offline batched inference (continuous batching) for speed.
- Thinking mode ON (Qwen3.5 default): output contains <think>...</think> then the answer.
- Interleaves question images at their <image N> markers.
- Parses "Answer: X" (A-J), scores accuracy, records per-item output-token counts.

Usage:
  python run_mmmupro.py --limit 25      # calibration
  python run_mmmupro.py                 # full 1730-question run
"""
import argparse, ast, base64, io, json, re, time, sys
from pathlib import Path

def b64_image(img, fmt="PNG"):
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

LETTERS = [chr(ord("A") + i) for i in range(26)]

PROMPT_HEADER = (
    "Answer the following multiple-choice question. The last line of your "
    "response must be exactly of the form: 'Answer: $LETTER' (without quotes) "
    "where $LETTER is one of the option letters. Think step by step before answering.\n\n"
)

IMG_MARK = re.compile(r"<image\s+(\d+)>")

def build_content(question, options, images):
    """Return OpenAI-style content list, interleaving images at <image N> markers."""
    content = [{"type": "text", "text": PROMPT_HEADER}]
    # Split the question on <image N> markers and interleave images in order.
    pos = 0
    for m in IMG_MARK.finditer(question):
        pre = question[pos:m.start()]
        if pre.strip():
            content.append({"type": "text", "text": pre})
        idx = int(m.group(1))
        if idx in images:
            content.append({"type": "image_url",
                            "image_url": {"url": b64_image(images[idx])}})
        else:
            content.append({"type": "text", "text": m.group(0)})
        pos = m.end()
    tail = question[pos:]
    if tail.strip():
        content.append({"type": "text", "text": tail})
    # Append any images that were not referenced by a marker (rare).
    referenced = {int(x) for x in IMG_MARK.findall(question)}
    for idx in sorted(images):
        if idx not in referenced:
            content.append({"type": "image_url",
                            "image_url": {"url": b64_image(images[idx])}})
    # Options block
    opt_lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    content.append({"type": "text", "text": "\n" + opt_lines})
    return content

ANS_RE = re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)

def parse_answer(text, n_options):
    """Extract the predicted option letter from generated text."""
    valid = set(LETTERS[:n_options])
    # Look only after the closing think tag if present
    if "</think>" in text:
        text_after = text.split("</think>")[-1]
    else:
        text_after = text
    cands = ANS_RE.findall(text_after) or ANS_RE.findall(text)
    for c in reversed(cands):  # last stated answer wins
        if c.upper() in valid:
            return c.upper()
    # Fallback: last standalone capital letter that is a valid option
    for ch in reversed(re.findall(r"\b([A-J])\b", text_after)):
        if ch in valid:
            return ch
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--limit", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--max-tokens", type=int, default=40960)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--max-num-seqs", type=int, default=0, help="0 = vLLM default")
    ap.add_argument("--max-num-batched-tokens", type=int, default=0, help="0 = vLLM default")
    ap.add_argument("--max-pixels", type=int, default=0, help="cap image pixels for speed; 0=processor default")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=0, help="0 = all at once; else checkpoint per chunk")
    ap.add_argument("--out", default="/workspace/mmmupro_qwen3vl/outputs/results.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    print(f"[load] dataset MMMU/MMMU_Pro standard(10 options) ...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[load] {len(ds)} questions", flush=True)

    # Build prompts
    conversations, meta = [], []
    for row in ds:
        opts = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
        images = {}
        for i in range(1, 8):
            im = row.get(f"image_{i}")
            if im is not None:
                images[i] = im
        content = build_content(row["question"], opts, images)
        conversations.append([{"role": "user", "content": content}])
        meta.append({"id": row["id"], "answer": row["answer"], "n_options": len(opts),
                     "n_images": len(images), "subject": row.get("subject")})

    mm_kwargs = {}
    if args.max_pixels:
        mm_kwargs["max_pixels"] = args.max_pixels

    llm_kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 8, "video": 0},
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    if args.max_num_seqs:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    if args.max_num_batched_tokens:
        llm_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if mm_kwargs:
        llm_kwargs["mm_processor_kwargs"] = mm_kwargs

    print(f"[init] loading vLLM engine for {args.model} ...", flush=True)
    t_load = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[init] engine ready in {time.time()-t_load:.1f}s", flush=True)

    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                        max_tokens=args.max_tokens, seed=args.seed)

    n = len(conversations)
    chunk = args.chunk_size if args.chunk_size > 0 else n
    print(f"[gen] generating for {n} questions (max_tokens={args.max_tokens}, chunk={chunk}) ...", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    correct = total_out_tok = truncated = no_answer = 0
    tok_counts = []
    t0 = time.time()
    with open(args.out, "w") as f:
        for start in range(0, n, chunk):
            conv_c = conversations[start:start + chunk]
            meta_c = meta[start:start + chunk]
            tc = time.time()
            outputs = llm.chat(conv_c, sp)
            for o, m in zip(outputs, meta_c):
                text = o.outputs[0].text
                n_tok = len(o.outputs[0].token_ids)
                fr = o.outputs[0].finish_reason
                pred = parse_answer(text, m["n_options"])
                is_correct = (pred == m["answer"])
                correct += int(is_correct)
                total_out_tok += n_tok
                tok_counts.append(n_tok)
                if fr == "length":
                    truncated += 1
                if pred is None:
                    no_answer += 1
                f.write(json.dumps({
                    "id": m["id"], "pred": pred, "gold": m["answer"], "correct": is_correct,
                    "finish_reason": fr, "out_tokens": n_tok, "n_images": m["n_images"],
                    "subject": m["subject"], "text": text,
                }) + "\n")
            f.flush()
            done = start + len(meta_c)
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed else 0
            eta = (n - done) / rate if rate else 0
            print(f"[chunk] {done}/{n} done | run_acc={correct/done:.3f} | "
                  f"chunk_t={time.time()-tc:.0f}s | elapsed={elapsed/60:.1f}m | "
                  f"q/s={rate:.4f} | agg_decode={total_out_tok/elapsed:.0f} tok/s | "
                  f"ETA={eta/60:.1f}m", flush=True)
    dt = time.time() - t0
    tok_counts.sort()
    median_tok = tok_counts[n // 2] if n else 0
    summary = {
        "model": args.model, "n_questions": n,
        "accuracy": round(correct / n, 4) if n else 0,
        "correct": correct, "no_answer": no_answer, "truncated_at_max": truncated,
        "gen_seconds": round(dt, 1),
        "questions_per_sec": round(n / dt, 3) if dt else 0,
        "total_output_tokens": total_out_tok,
        "mean_output_tokens": round(total_out_tok / n, 1) if n else 0,
        "median_output_tokens": median_tok,
        "max_output_tokens": tok_counts[-1] if n else 0,
        "decode_tokens_per_sec": round(total_out_tok / dt, 1) if dt else 0,
    }
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2), flush=True)
    with open(args.out.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
