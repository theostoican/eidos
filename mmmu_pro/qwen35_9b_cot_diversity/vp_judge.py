#!/usr/bin/env python
"""Stage 3: fact-check every extracted premise against the actual image (InternVL3-8B, temp 0).

NO GOLD ANSWER IS SHOWN. The judge sees the image, the question and the options, and one
trace's premises -- never which option is correct, and never the trace's conclusion. A prior
self-judge that did see outcomes scored 96.5% sound on correct-answer traces vs 70.3% on
wrong-answer ones, which is a judge reading the answer key.

DIFFERENT MODEL FAMILY from the generator (Qwen3.5-9B), so this is not a self-judge. On a
24GB card a 38B judge does not fit; InternVL3-8B-AWQ does, at the cost of a weaker judge.
The absolute soundness level is judge-dependent; only the SHAPE across top_p is claimed, and
the judge is identical at every top_p.

PROMPT ORDER IS DELIBERATE: rubric + question + options, then the images, then the premise
list. Everything before the premise list is identical for every trace of a question, so with
prefix caching the image tokens are encoded ONCE per question rather than once per trace --
worth ~10x here. Rows are therefore judged grouped by question.

KNOWN LIMIT: re-judging the same premises in a shuffled order flips 43% of UNSOUND verdicts
(Cohen's kappa 0.36). Aggregate means over ~10^5 premises survive that; individual trace
verdicts do not. See vp_judge_shuffle.py.
"""
import argparse, ast, base64, collections, io, json, re, time
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]
IMG_MARK = re.compile(r"<image\s+(\d+)>")
VERDICT_RE = re.compile(r"^\s*(\d+)\s*[:.\)]\s*(SOUND|UNSOUND|UNVERIFIABLE)\b",
                        re.MULTILINE | re.IGNORECASE)

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


def b64_image(img, fmt="PNG"):
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def load_options(row):
    return ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-8B-AWQ")
    ap.add_argument("--quantization", default="awq")
    ap.add_argument("--premises", default="outputs/vp_premises.jsonl")
    ap.add_argument("--questions", default="outputs/vp_q.json")
    ap.add_argument("--out", default="outputs/vp_verdicts.jsonl")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--max-premises-per-call", type=int, default=25)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=16384,
                    help="a 5-image question at max_dynamic_patch=6 is ~9k image tokens; "
                         "12288 leaves too little headroom once rubric and premises land")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-dynamic-patch", type=int, default=6,
                    help="cap InternVL dynamic tiles (config default 12); each tile ~256 tokens. "
                         "Applied via hf_overrides, NOT mm_processor_kwargs: the latter is "
                         "forwarded to the video processor too, which rejects the argument.")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    qmeta = json.load(open(a.questions))
    done = set()
    if a.resume and Path(a.out).exists():
        for line in open(a.out):
            try:
                r = json.loads(line)
            except Exception:
                continue
            done.add((r["id"], r["top_p"], r["sample_idx"]))
        print(f"[judge] resume: {len(done)} traces already judged")

    rows = []
    for line in open(a.premises):
        r = json.loads(line)
        if r["n_premises"] == 0 or (r["id"], r["top_p"], r["sample_idx"]) in done:
            continue
        rows.append(r)
    rows.sort(key=lambda r: (r["id"], r["top_p"], r["sample_idx"]))   # group by question
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        print("[judge] nothing to do")
        return
    print(f"[judge] {len(rows)} traces | {sum(r['n_premises'] for r in rows)} premises | "
          f"{len({r['id'] for r in rows})} questions", flush=True)

    from datasets import load_dataset
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")

    from vllm import LLM, SamplingParams
    kw = {"quantization": a.quantization} if a.quantization else {}
    llm = LLM(model=a.model, dtype="auto", gpu_memory_utilization=a.gpu_mem_util,
              max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len,
              limit_mm_per_prompt={"image": 8, "video": 0}, trust_remote_code=True,
              enable_prefix_caching=True, seed=1234,
              hf_overrides={"max_dynamic_patch": a.max_dynamic_patch}, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    qcache = {}

    def scaffold(qid):
        if qid not in qcache:
            row = ds[qmeta[qid]["ds_index"]]
            text = row["question"]
            order = [int(n) for n in IMG_MARK.findall(text)]
            text = IMG_MARK.sub("[image]", text)
            imgs = {i: row.get(f"image_{i}") for i in range(1, 8)
                    if row.get(f"image_{i}") is not None}
            if not order:
                order = sorted(imgs)
            seen, uniq = set(), []
            for i in order:                       # a repeated <image N> is one image
                if i in imgs and i not in seen:
                    seen.add(i)
                    uniq.append(i)
            opts = load_options(row)
            qcache.clear()                        # rows are question-sorted: keep one alive
            qcache[qid] = {
                "head": (RUBRIC + f"\n=== QUESTION ===\n{text}\n\n=== OPTIONS ===\n"
                         + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + "\n"),
                "b64": [b64_image(imgs[i]) for i in uniq]}
        return qcache[qid]

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0, n_sound, n_unsound, n_unver, n_unparsed, n_done = time.time(), 0, 0, 0, 0, 0
    with open(a.out, "a" if a.resume else "w") as fout:
        for s in range(0, len(rows), a.batch):
            chunk = rows[s:s + a.batch]
            convs = []
            for r in chunk:
                q = scaffold(r["id"])
                prem = r["premises"][:a.max_premises_per_call]
                plist = "\n".join(f"{i+1}. {p}" for i, p in enumerate(prem))
                content = [{"type": "text", "text": q["head"]}]
                for b in q["b64"]:
                    content.append({"type": "image_url", "image_url": {"url": b}})
                content.append({"type": "text",
                                "text": f"\n=== PREMISES TO CHECK ({len(prem)}) ===\n{plist}\n"})
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
                prem = r["premises"][:a.max_premises_per_call]
                if o is None:
                    verdicts, raw = [None] * len(prem), "SKIPPED_ENGINE_ERROR"
                else:
                    raw = o.outputs[0].text
                    parsed = {}
                    for num, verd in VERDICT_RE.findall(raw):
                        k = int(num) - 1
                        if 0 <= k < len(prem):
                            v = verd.upper()
                            parsed[k] = True if v == "SOUND" else (False if v == "UNSOUND" else "U")
                    verdicts = [parsed.get(k) for k in range(len(prem))]
                ns = sum(v is True for v in verdicts)
                nu = sum(v is False for v in verdicts)
                nv = sum(v == "U" for v in verdicts)
                nmiss = sum(v is None for v in verdicts)
                n_sound += ns; n_unsound += nu; n_unver += nv; n_unparsed += nmiss
                fout.write(json.dumps({
                    "id": r["id"], "subject": r.get("subject"), "top_p": r["top_p"],
                    "sample_idx": r["sample_idx"], "gold": r.get("gold"), "pred": r.get("pred"),
                    "answer_correct": r.get("answer_correct"),
                    "finish_reason": r.get("finish_reason"), "n_premises": len(prem),
                    "orig_pos": r.get("orig_pos"),
                    "verdicts": [("SOUND" if v is True else "UNSOUND" if v is False
                                  else "UNVERIFIABLE" if v == "U" else None) for v in verdicts],
                    "n_sound": ns, "n_unsound": nu, "n_unverifiable": nv, "n_unparsed": nmiss,
                    "frac_sound": (ns / (ns + nu)) if (ns + nu) else None,
                    "raw": "" if (ns + nu + nv) else raw[:400],
                }) + "\n")
                n_done += 1
            fout.flush()
            el = time.time() - t0
            den = max(n_sound + n_unsound, 1)
            print(f"[judge] {n_done}/{len(rows)} | sound {n_sound/den:.1%} | "
                  f"unverifiable {n_unver} | unparsed {n_unparsed} | "
                  f"{n_done/el*3600:.0f} traces/h | "
                  f"eta {(len(rows)-n_done)/max(n_done,1)*el/3600:.1f}h", flush=True)
    print(f"[judge] done: {n_done} traces -> {a.out}")


if __name__ == "__main__":
    main()
