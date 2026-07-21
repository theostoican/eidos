#!/usr/bin/env python
"""Judge each VISUAL PREMISE against the image with InternVL3-38B. NO GOLD ANSWER.

Why no gold (HANDOFF_RESULTS.md / HANDOFF_NEXT_visual_premises.md -- do not relearn this
the hard way): showing the judge the correct answer makes it anchor on outcome rather than
on the image. The prior Qwen self-judge went 96.5% "sound" on correct-answer traces vs 70.3%
on wrong-answer ones -- circular. This judge sees the image, the question, the options and
ONE premise at a time-worth of context; it never sees which option is right.

Why InternVL3-38B: it is a different model family from the generator (Qwen3.5-9B), so this
is not a self-judge, and it is the only complete InternVL checkpoint in the local cache
(InternVL3-9B's weight shards are missing).

Batching: all premises of a single trace go in ONE call, numbered, so the image is encoded
once instead of once per premise. The model returns one verdict line per premise.

Input : outputs/vp_premises.jsonl  (from extract_premises.py)
Output: outputs/vp_verdicts.jsonl  one line per (id, top_p, sample_idx):
  {id, top_p, sample_idx, n_premises, verdicts: [bool|null, ...], n_sound, n_judged,
   frac_sound, raw}
frac_sound over the PARSED verdicts is the graded soundness the experiment is after.
"""
import argparse, ast, base64, io, json, re
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]
IMG_MARK = re.compile(r"<image\s+(\d+)>")
VERDICT_RE = re.compile(r"^\s*(\d+)\s*[:.\)]\s*(SOUND|UNSOUND|UNVERIFIABLE)\b", re.MULTILINE | re.IGNORECASE)


def b64_image(img, fmt="PNG"):
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def load_options(row):
    return ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]


RUBRIC = """You are a strict visual fact-checker. You see an image (or several), the question that was asked about it, and a numbered list of statements a model made while reasoning. Classify EACH statement.

FIRST decide if the statement is even a claim about the IMAGE:
- If it is a calculation, algebra, integral, formula, derivation, goal, plan, or inference/conclusion — NOT a direct observation of the image — answer UNVERIFIABLE, EVEN IF THE MATH IS CORRECT. You are not checking math. Example UNVERIFIABLE: "the integral evaluates to X", "(jw)^2 = -w^2", "the answer is C", "the goal is to find the rent".

If it IS a claim about what the image depicts (a value/label read from a chart or table, a shape, curve, count, color, position, or visible text), then:
- SOUND   — the image clearly supports it.
- UNSOUND — the image contradicts it, or it asserts a detail that is not there (a fabricated label, invented object, misread number/axis/color).

Be STRICT on SOUND/UNSOUND: a specific value, label, count, or position is SOUND only if you can confirm that exact detail in the image. Approximately-right-but-wrong-in-the-specific is UNSOUND. Do not give the benefit of the doubt, and do not try to solve the question yourself.

Output EXACTLY one line per statement, in order, nothing else:
<number>: SOUND
<number>: UNSOUND
<number>: UNVERIFIABLE
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-38B-AWQ")
    ap.add_argument("--quantization", default="awq",
                    help="AWQ config is nested under llm_config.quantization_config in this repo, "
                         "which vLLM may not auto-detect -- pass it explicitly. '' to auto-detect.")
    ap.add_argument("--premises", default="outputs/vp_premises.jsonl")
    ap.add_argument("--questions", default="outputs/vp_q.json",
                    help="qid -> {subject, gold, n_options, n_images, ds_index}; shards auto-merged")
    ap.add_argument("--out", default="outputs/vp_verdicts.jsonl")
    ap.add_argument("--max-premises-per-call", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--tensor-parallel-size", type=int, default=2,
                    help="InternVL3-38B-AWQ is ~29GB (int4 LLM + bf16 vision tower). TP=1 fits a "
                         "40GB card but leaves only ~8GB for KV + dynamic-tile vision activations; "
                         "TP=2 is comfortable. GPU 1 is DEAD (ECC) -- use healthy GPUs only.")
    ap.add_argument("--max-dynamic-patch", type=int, default=6,
                    help="cap InternVL dynamic image tiles (config default 12). Each tile ~256 "
                         "tokens; 12 tiles x multi-image can exceed max_model_len. 6 keeps prompts "
                         "in-context at TP=1 with modest visual-acuity loss.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import glob
    qmeta = {}
    for f in sorted(glob.glob(args.questions.replace(".json", ".shard*.json")) or [args.questions]):
        qmeta.update(json.load(open(f)))
    print(f"[judge] {len(qmeta)} questions in metadata")

    rows = [json.loads(l) for l in open(args.premises)]
    rows = [r for r in rows if r["n_premises"] > 0]
    if args.limit:
        rows = rows[:args.limit]
    print(f"[judge] {len(rows)} traces with premises | "
          f"{sum(r['n_premises'] for r in rows)} premises total")

    from datasets import load_dataset
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")

    # cache per-question prompt scaffolding (question text, options, encoded images)
    qcache = {}
    for qid in sorted({r["id"] for r in rows}):
        row = ds[qmeta[qid]["ds_index"]]
        text = row["question"]
        order = [int(n) for n in IMG_MARK.findall(text)]
        text = IMG_MARK.sub("[image]", text)
        imgs = {}
        for i in range(1, 8):
            im = row.get(f"image_{i}")
            if im is not None:
                imgs[i] = im
        if not order:
            order = sorted(imgs)
        opts = load_options(row)
        qcache[qid] = {
            "question": text,
            "options": "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)),
            "b64": [b64_image(imgs[i]) for i in order if i in imgs],
        }

    from vllm import LLM, SamplingParams
    llm_kwargs = {}
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(model=args.model, dtype="auto",
              gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs,
              max_model_len=args.max_model_len,
              tensor_parallel_size=args.tensor_parallel_size,
              limit_mm_per_prompt={"image": 8, "video": 0},
              trust_remote_code=True, seed=1234, **llm_kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_sound = n_judged = n_unparsed = 0
    with open(args.out, "w") as fout:
        for s in range(0, len(rows), args.batch):
            chunk = rows[s:s + args.batch]
            convs = []
            for r in chunk:
                q = qcache[r["id"]]
                prem = r["premises"][:args.max_premises_per_call]
                plist = "\n".join(f"{i+1}. {p}" for i, p in enumerate(prem))
                text = (RUBRIC +
                        f"\n=== QUESTION ===\n{q['question']}\n\n=== OPTIONS ===\n{q['options']}\n\n"
                        f"=== PREMISES TO CHECK ({len(prem)}) ===\n{plist}\n")
                content = [{"type": "text", "text": text}]
                for b in q["b64"]:
                    content.append({"type": "image_url", "image_url": {"url": b}})
                convs.append([{"role": "user", "content": content}])

            try:
                outs = llm.chat(convs, sp)
            except Exception as e:
                print(f"[judge] batch failed ({type(e).__name__}); retrying one-by-one", flush=True)
                outs = []
                for cv in convs:
                    try:
                        outs.append(llm.chat([cv], sp)[0])
                    except Exception as e2:
                        print(f"[judge]   skipping one trace: {type(e2).__name__}", flush=True)
                        outs.append(None)
            for r, o in zip(chunk, outs):
                if o is None:
                    prem = r["premises"][:args.max_premises_per_call]
                    fout.write(json.dumps({
                        "id": r["id"], "top_p": r["top_p"], "sample_idx": r["sample_idx"],
                        "subject": r.get("subject"), "gold": r.get("gold"), "pred": r.get("pred"),
                        "answer_correct": r.get("answer_correct"),
                        "n_premises": len(prem), "verdicts": [None]*len(prem),
                        "n_sound": 0, "n_judged": 0, "frac_sound": None, "raw": "SKIPPED_OVERLENGTH",
                    }) + "\n")
                    n_unparsed += len(prem)
                    continue
                raw = o.outputs[0].text
                prem = r["premises"][:args.max_premises_per_call]
                parsed = {}
                for num, verd in VERDICT_RE.findall(raw):
                    k = int(num) - 1
                    if 0 <= k < len(prem):
                        v = verd.upper()
                        parsed[k] = True if v == "SOUND" else (False if v == "UNSOUND" else None)
                verdicts = [parsed.get(k) for k in range(len(prem))]
                ok = [v for v in verdicts if v is not None]
                n_sound += sum(ok)
                n_judged += len(ok)
                n_unparsed += len(prem) - len(ok)
                fout.write(json.dumps({
                    "id": r["id"], "top_p": r["top_p"], "sample_idx": r["sample_idx"],
                    "subject": r.get("subject"), "gold": r.get("gold"), "pred": r.get("pred"),
                    "answer_correct": r.get("answer_correct"),
                    "n_premises": len(prem), "verdicts": verdicts,
                    "n_sound": sum(ok), "n_judged": len(ok),
                    "frac_sound": (sum(ok) / len(ok)) if ok else None,
                    "raw": raw if not ok else "",
                }) + "\n")
            fout.flush()
            print(f"[judge] {min(s+args.batch, len(rows))}/{len(rows)} traces | "
                  f"{n_judged} premises judged | sound {n_sound/max(n_judged,1):.1%} | "
                  f"unparsed {n_unparsed}", flush=True)

    print(f"[judge] done: {n_judged} premises judged, {n_sound} sound "
          f"({n_sound/max(n_judged,1):.1%}), {n_unparsed} unparsed -> {args.out}")


if __name__ == "__main__":
    main()
