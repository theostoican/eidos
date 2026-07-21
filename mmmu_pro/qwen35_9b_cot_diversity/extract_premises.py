#!/usr/bin/env python
"""Extract VISUAL PREMISES from each <think> trace.

A "visual premise" is an atomic statement the trace asserts about what it SEES in the
image -- "the far-left inductor is labelled j20", "the woman holds a wax tablet",
"curve C peaks at t=b". NOT arithmetic, NOT inference, NOT option elimination: only
claims that could be checked by looking at the picture.

Motivation (HANDOFF_NEXT_visual_premises.md): whole-CoT soundness is one coarse binary
per trace and comes out flat (~0.61) across top_p. A long trace makes MANY visual reads,
some right some wrong; collapsing them to one bit throws away the signal. Judging each
premise separately gives a GRADED soundness (fraction-of-premises-correct).

Input : outputs/vp_extracted.jsonl  (from extract_cot.py -- needs the "cot" field)
Output: outputs/vp_premises.jsonl   one line per (id, top_p, sample_idx):
  {id, subject, top_p, sample_idx, gold, pred, answer_correct, finish_reason,
   premises: [str, ...], n_premises}

Extraction is done by Qwen3.5-9B in NON-thinking mode at temperature 0 -- this is a
text-only rewriting task (pull sentences out of a trace), not a judgement, so the
generator model is fine here. Soundness judging is a separate, harder job and uses
InternVL3-38B against the actual image (judge_premises_internvl.py).

Only sample_idx < --judge-samples are processed: per-premise judging is ~5-15x more
calls than whole-CoT, so the premise arm subsamples while majority-vote accuracy
still uses all 16 samples.
"""
import argparse, json, re
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
    """Pull 'P: ...' lines out of the extractor output, dedup, cap."""
    if "NONE" in text[:40].upper() and "P:" not in text:
        return []
    out, seen = [], set()
    for m in PREM_RE.findall(text):
        p = m.strip().strip("-*").strip()
        # drop degenerate / numbering-only lines
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
    ap.add_argument("--extracted", default="outputs/vp_extracted.jsonl")
    ap.add_argument("--out", default="outputs/vp_premises.jsonl")
    ap.add_argument("--judge-samples", type=int, default=4,
                    help="only extract premises for sample_idx < this (judging cost control)")
    ap.add_argument("--max-cot-chars", type=int, default=24000)
    ap.add_argument("--max-premises", type=int, default=40)
    ap.add_argument("--skip-truncated", action="store_true", default=True,
                    help="drop finish_reason!='stop' (truncated traces are repetition loops)")
    ap.add_argument("--keep-truncated", dest="skip_truncated", action="store_false")
    ap.add_argument("--max-tokens", type=int, default=1536)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    for line in open(args.extracted):
        r = json.loads(line)
        if r["sample_idx"] >= args.judge_samples:
            continue
        if args.skip_truncated and r.get("finish_reason") != "stop":
            continue
        if not r.get("cot"):
            continue
        rows.append(r)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[premises] {len(rows)} traces to extract from "
          f"(sample_idx < {args.judge_samples}, truncated {'dropped' if args.skip_truncated else 'kept'})")

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs,
              max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size,
              trust_remote_code=True, seed=1234)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_prem = n_empty = 0
    with open(args.out, "w") as fout:
        for s in range(0, len(rows), args.batch):
            chunk = rows[s:s + args.batch]
            convs = [[{"role": "user", "content": HEADER +
                       "\n=== REASONING TRACE ===\n" + r["cot"][:args.max_cot_chars] +
                       "\n=== END TRACE ===\n"}] for r in chunk]
            outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
            for r, o in zip(chunk, outs):
                prem = parse_premises(o.outputs[0].text, args.max_premises)
                n_prem += len(prem)
                n_empty += int(not prem)
                fout.write(json.dumps({
                    "id": r["id"], "subject": r.get("subject"), "top_p": r["top_p"],
                    "sample_idx": r["sample_idx"], "gold": r["gold"], "pred": r.get("pred"),
                    "answer_correct": r.get("answer_correct"),
                    "finish_reason": r.get("finish_reason"),
                    "premises": prem, "n_premises": len(prem),
                }) + "\n")
            fout.flush()
            print(f"[premises] {min(s+args.batch, len(rows))}/{len(rows)} traces | "
                  f"{n_prem} premises so far", flush=True)

    n = len(rows)
    print(f"[premises] done: {n} traces -> {n_prem} premises "
          f"(mean {n_prem/max(n,1):.1f}/trace, {n_empty} traces yielded none) -> {args.out}")


if __name__ == "__main__":
    main()
