#!/usr/bin/env python
"""Stage 2: extract VISUAL PREMISES from each T=1.0 <think> trace (Qwen3.5-9B, temp 0).

A visual premise is an atomic claim about what the model SAW -- "the far-left inductor is
labelled j20", "the left bar is tallest". Not arithmetic, not inference, not option
elimination: only claims a person could check by looking at the picture.

Why per-premise at all: whole-CoT soundness is one coarse bit per trace and comes out flat.
A long trace makes many visual reads, some right, some wrong; collapsing them to one bit
throws the signal away.

Why the GENERATOR model extracts: this is text-to-text rewriting (pull the sentences out,
keep their wording), not judgement. It never sees the image and never decides truth.

Rows are read from vp_prep.py's layer files and appended to --out; --resume skips any
(id, top_p, sample_idx) already present, so the run survives restarts.
"""
import argparse, glob, gzip, json, re, time
from pathlib import Path

HEADER = """You extract VISUAL PREMISES from a reasoning trace written by a model that was looking at an image while answering a question.

A VISUAL PREMISE is a claim about what is DEPICTED in the image — something you could confirm or refute by LOOKING AT THE IMAGE ALONE, with no math and no reading of the reasoning. Examples:
- a value/label read off a chart, table, gauge, or diagram ("Machine A's initial cost is $40,000", "the resistor is labelled 4 ohms")
- a shape, curve, count, color, or position ("the graph is a straight line from (0, v_o) to (b, 0)", "there are three people seated", "the left bar is tallest")
- text or a symbol visible in the image ("the axis is labelled t")

The trace will contain MANY sentences that are NOT visual premises. You MUST exclude them:
- arithmetic, algebra, calculus, any calculation or identity ("the integral evaluates to X", "(jw)^2 = -w^2", "40000/2.673 = 14964")
- derivations or formulas the model builds up ("so f(t) = v_o(1 - t/b)", "using integration by parts...")
- goals, plans, meta-talk ("the goal is to find...", "let me compute...", "I need to check...")
- inference, elimination, conclusions ("therefore the answer is C", "option B is impossible")

Most traces yield only a HANDFUL of visual premises (often 2-8), even if the trace is long. Do NOT pad the list with reasoning steps.

--- WORKED EXAMPLE ---
TRACE (excerpt): "The graph shows a straight line starting at v_o when t=0 and dropping to 0 at t=b. So f(t) = v_o(1 - t/b) on [0,b]. The Fourier transform is the integral of f(t)e^{-jwt}. Splitting the integral, the first part gives (1-e^{-jwb})/jw. By integration by parts the second part is ... Therefore the answer is C."
CORRECT OUTPUT:
P: the graph shows a straight line
P: the line starts at v_o when t=0
P: the line drops to 0 at t=b
(NOTE: the formula f(t)=v_o(1-t/b), the Fourier integral, the split, integration by parts, and "the answer is C" are ALL excluded — they are math/derivation/inference, not things visible in the image.)
--- END EXAMPLE ---

Now do the same for the trace below.
Rules: restate each visual premise as a short standalone sentence, keep the trace's own wording and numbers (do NOT correct them even if you think they are wrong), list each distinct visual fact ONCE. If the trace makes NO checkable visual claims, output exactly: NONE

Output — one premise per line, prefixed "P: ", nothing else:
P: <premise>
P: <premise>
"""

PREM_RE = re.compile(r"^\s*P:\s*(.+?)\s*$", re.MULTILINE)


def parse_premises(text, max_premises):
    """Pull 'P: ...' lines out, drop degenerate ones, DEDUPLICATE, cap.

    Dedup is load-bearing, not tidying: at top_p=0.5, 11% of traces are repetition loops and
    the extractor faithfully echoes every restatement. Without it a single looping trace
    contributes 40 copies of one read, which would inflate premise counts exactly where the
    loops live (low top_p), let one claim dominate a cell's soundness, and collapse the Vendi
    diversity score at the same end of the axis -- manufacturing the diversity gradient this
    experiment reports. Residual near-duplicates (Jaccard>=0.8) run 1.1% and, unlike exact
    ones, do not trend along top_p.
    """
    if "NONE" in text[:40].upper() and "P:" not in text:
        return []
    out, seen = [], set()
    for m in PREM_RE.findall(text):
        p = m.strip().strip("-*").strip()
        if len(p) < 8:
            continue
        key = re.sub(r"[^a-z0-9]+", "", p.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= max_premises:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--layers", default="outputs/vp_layers/layer*.jsonl.gz")
    ap.add_argument("--out", default="outputs/vp_premises.jsonl")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    # THE CAP IS IN TOKENS, NOT CHARACTERS. A character cap is not a context guarantee: 24k
    # chars of dense table/CJK/math text tokenises past 12k tokens, the engine REJECTS the
    # request, and one such trace killed an entire layer. Clipping with the model's own
    # tokenizer bounds the prompt exactly. --max-cot-chars is a cheap pre-clip so the
    # tokenizer never sees a 400KB repetition loop.
    ap.add_argument("--max-cot-tokens", type=int, default=6000)
    ap.add_argument("--max-cot-chars", type=int, default=60000)
    ap.add_argument("--max-premises", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--gpu-mem-util", type=float, default=0.93)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--kv-cache-dtype", default="fp8",
                    help="fp8 halves KV so more sequences fit beside 18GB of bf16 weights on "
                         "a 24GB card. Affects the EXTRACTOR only, never the traces.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(a.layers))
    if not files:
        raise SystemExit(f"[extract] no layer files match {a.layers!r} -- run vp_prep.py first")
    done = set()
    if a.resume and Path(a.out).exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["id"], r["top_p"], r["sample_idx"]))
        print(f"[extract] resume: {len(done)} traces already extracted")

    rows = []
    for f in files:
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            if (r["id"], r["top_p"], r["sample_idx"]) in done or not r.get("cot"):
                continue
            rows.append(r)
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        print("[extract] nothing to do")
        return
    layers = sorted({r["sample_idx"] for r in rows})
    print(f"[extract] {len(rows)} traces | layers {layers[0]}-{layers[-1]} "
          f"| cap {a.max_cot_tokens} tokens", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    n_clip = 0
    for i in range(0, len(rows), 512):
        blk = rows[i:i + 512]
        ids = tok([r["cot"][:a.max_cot_chars] for r in blk], add_special_tokens=False)["input_ids"]
        for r, tid in zip(blk, ids):
            r["cot_tokens_kept"] = min(len(tid), a.max_cot_tokens)
            if len(tid) > a.max_cot_tokens:
                r["cot"] = tok.decode(tid[:a.max_cot_tokens])
                n_clip += 1
            else:
                r["cot"] = r["cot"][:a.max_cot_chars]
    print(f"[extract] {n_clip}/{len(rows)} traces clipped ({100*n_clip/len(rows):.1f}%)", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
              max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len,
              kv_cache_dtype=a.kv_cache_dtype, trust_remote_code=True,
              enable_prefix_caching=True, seed=1234)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0, n_prem, n_empty, n_done = time.time(), 0, 0, 0
    with open(a.out, "a" if a.resume else "w") as fout:
        for s in range(0, len(rows), a.batch):
            chunk = rows[s:s + a.batch]
            convs = [[{"role": "user", "content": HEADER + "\n=== REASONING TRACE ===\n"
                       + r["cot"] + "\n=== END TRACE ===\n"}] for r in chunk]
            try:
                outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
            except Exception as e:
                # one rejected prompt must not cost the whole layer
                print(f"[extract] batch failed ({type(e).__name__}); retrying one-by-one", flush=True)
                outs = []
                for cv in convs:
                    try:
                        outs.append(llm.chat([cv], sp,
                                             chat_template_kwargs={"enable_thinking": False})[0])
                    except Exception as e2:
                        print(f"[extract]   skipping one trace: {type(e2).__name__}", flush=True)
                        outs.append(None)
            for r, o in zip(chunk, outs):
                prem = parse_premises(o.outputs[0].text, a.max_premises) if o is not None else []
                n_prem += len(prem)
                n_empty += int(not prem)
                fout.write(json.dumps({
                    "id": r["id"], "subject": r.get("subject"), "top_p": r["top_p"],
                    "sample_idx": r["sample_idx"], "gold": r["gold"],
                    "n_options": r.get("n_options", 10), "pred": r.get("pred"),
                    "answer_correct": r.get("answer_correct"),
                    "finish_reason": r.get("finish_reason"), "cot_chars": r.get("cot_chars"),
                    "cot_tokens_kept": r.get("cot_tokens_kept"), "cot_token_cap": a.max_cot_tokens,
                    "premises": prem, "n_premises": len(prem), "extract_failed": o is None,
                }) + "\n")
                n_done += 1
            fout.flush()
            el = time.time() - t0
            print(f"[extract] {n_done}/{len(rows)} | {n_prem} premises "
                  f"(mean {n_prem/max(n_done,1):.1f}, {n_empty} empty) | "
                  f"{n_done/el*3600:.0f} traces/h | "
                  f"eta {(len(rows)-n_done)/max(n_done,1)*el/3600:.1f}h", flush=True)
    print(f"[extract] done: {n_done} traces -> {n_prem} premises -> {a.out}")


if __name__ == "__main__":
    main()
