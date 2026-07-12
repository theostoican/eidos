#!/usr/bin/env python
"""Local Qwen3.5-9B judge for CoT reasoning soundness (the handoff's "local self-judge").

Reads each generated CoT + the question image(s) and rules the REASONING sound/unsound
under a STRICT rubric, writing one line per sample to outputs/verdicts_cot.jsonl:

    {"id": ..., "top_p": ..., "sample_idx": ..., "sound": true|false, "raw": "SOUND"|"UNSOUND"}

which is exactly what analyze_cot.py consumes.

Design notes / caveats:
  - This is a SELF-judge: the same model family that produced the CoT grades it, so it is
    weaker and somewhat circular vs an independent judge (Sonnet). Documented on purpose.
  - The gold answer IS shown to the judge to anchor the visual facts, but the rubric is
    explicit that reaching the right letter by a misread or an invalid step is UNSOUND.
  - CoT = text before </think> (Qwen emits the opening <think> in the prompt; the closing
    tag is the delimiter). Truncated traces (no </think>) are judged on what they produced.

Runs on ONE GPU (pin with CUDA_VISIBLE_DEVICES) so it can share a node with the 3-GPU
generator. --watch streams over generations as they are appended and resumes cleanly:
already-judged (id, top_p, sample_idx) triples in verdicts_cot.jsonl are skipped.
"""
import argparse, ast, base64, glob, io, json, os, re, time
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]

def b64_image(img, fmt="PNG"):
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

def load_options(row):
    return ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]

def split_cot(text):
    if "</think>" in text:
        return text.split("</think>", 1)[0].replace("<think>", "").strip()
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip()
    return text.strip()

def rubric(reveal_gold):
    """The grading rubric. With reveal_gold=False the CORRECT answer is NOT shown, so the
    judge must verify the reasoning against the image on its own merits -- this removes the
    gold-anchoring confound (Qwen otherwise rates a CoT sound 96% when the answer is right vs
    62% when wrong). The de-anchored judge must independently check the visual reads."""
    intro = ("its image(s), the answer options, the CORRECT answer, and" if reveal_gold
             else "its image(s) and the answer options, and")
    extra = ("" if reveal_gold else
             "You are NOT told the correct answer -- verify the reasoning against the IMAGE "
             "yourself. ")
    return (
        "You are a STRICT grader of visual reasoning. Below is a multiple-choice question with "
        + intro + " a step-by-step REASONING "
        "TRACE produced by another model. Judge ONLY whether the reasoning is SOUND.\n\n"
        + extra +
        "The trace is SOUND only if BOTH of these hold:\n"
        "  1. Every fact it reads off the image is accurate (no misread value, label, shape, "
        "connection, axis, or count).\n"
        "  2. Every inferential / mathematical step that follows is valid.\n\n"
        "If it misreads the image even once in a way that matters, OR makes an invalid step, it "
        "is UNSOUND -- EVEN IF it still arrives at the correct letter (reaching the right answer "
        "by a lucky guess, a compensating error, or wrong reasoning is UNSOUND). A trace that is "
        "truncated before finishing is UNSOUND unless the reasoning it did produce is fully "
        "correct and already determines the answer.\n\n"
        "Give your reasoning in AT MOST 3 short sentences, then end with EXACTLY one final line "
        "that is either 'VERDICT: SOUND' or 'VERDICT: UNSOUND'. The VERDICT line is mandatory."
    )

VERDICT_RE = re.compile(r"VERDICT:\s*(SOUND|UNSOUND)", re.IGNORECASE)

def parse_verdict(text):
    m = list(VERDICT_RE.finditer(text))
    if m:
        return m[-1].group(1).upper() == "SOUND", m[-1].group(1).upper()
    # fallback: last standalone SOUND/UNSOUND token
    toks = re.findall(r"\b(UNSOUND|SOUND)\b", text, re.IGNORECASE)
    if toks:
        return toks[-1].upper() == "SOUND", toks[-1].upper()
    return None, "UNPARSED"

def build_judge_conv(row_cache, qid, gold, cot, max_cot_chars, reveal_gold=True):
    q_text, options, images = row_cache[qid]
    if max_cot_chars and len(cot) > max_cot_chars:            # guard context blowups
        cot = cot[:max_cot_chars] + "\n...[trace truncated for judging]..."
    opt_lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    content = [{"type": "text", "text": rubric(reveal_gold) + "\n\n--- QUESTION ---\n" + q_text.strip()}]
    for im in images:
        content.append({"type": "image_url", "image_url": {"url": im}})
    tail = "\n--- OPTIONS ---\n" + opt_lines
    if reveal_gold:
        gi = LETTERS.index(gold) if gold in LETTERS else None
        gold_line = f"{gold}. {options[gi]}" if gi is not None and gi < len(options) else gold
        tail += f"\n\n--- CORRECT ANSWER ---\n{gold_line}"
    tail += "\n\n--- REASONING TRACE TO GRADE ---\n" + cot + "\n\n--- END TRACE ---\nGrade the reasoning now."
    content.append({"type": "text", "text": tail})
    return [{"role": "user", "content": content}]

def read_gen_rows(gen_glob):
    seen_files = sorted(set(glob.glob(gen_glob) + glob.glob(gen_glob.replace(".jsonl", ".shard*.jsonl"))))
    for f in seen_files:
        with open(f) as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except Exception:
                    continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--gen", default="outputs/cot_gen.jsonl",
                    help="generations to judge; shards (<gen>.shard*.jsonl) are auto-included")
    ap.add_argument("--questions", default="outputs/questions.json")
    ap.add_argument("--verdicts", default="outputs/verdicts_cot.jsonl")
    ap.add_argument("--max-cot-chars", type=int, default=120000,
                    help="hard cap on CoT chars fed to the judge (~30k tokens)")
    ap.add_argument("--max-model-len", type=int, default=49152)
    ap.add_argument("--gpu-mem-util", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-judge-tokens", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=256, help="judge prompts per llm.chat call")
    ap.add_argument("--no-gold", action="store_true",
                    help="do NOT show the judge the gold answer (de-anchored judge; removes the "
                         "gold-anchoring confound so soundness is judged independently)")
    ap.add_argument("--watch", action="store_true", help="stream: keep judging new gens as they appear")
    ap.add_argument("--poll", type=float, default=30.0, help="watch: seconds between rescans")
    ap.add_argument("--stop-file", default="outputs/.judge_stop",
                    help="watch: create this file to end after the next drain")
    args = ap.parse_args()

    # image/question cache built lazily from the dataset via ds_index in questions.json
    qmeta = json.load(open(args.questions))
    from datasets import load_dataset
    print("[judge] loading MMMU_Pro standard (10 options) test for images ...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    row_cache = {}
    def ensure_row(qid):
        if qid in row_cache:
            return
        row = ds[qmeta[qid]["ds_index"]]
        imgs = []
        for i in range(1, 8):
            im = row.get(f"image_{i}")
            if im is not None:
                imgs.append(b64_image(im))
        row_cache[qid] = (row["question"], load_options(row), imgs)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 8, "video": 0}, trust_remote_code=True, seed=1234)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_judge_tokens)

    def load_done():
        done = set()
        if Path(args.verdicts).exists():
            for line in open(args.verdicts):
                try:
                    v = json.loads(line)
                    done.add((v["id"], v["top_p"], v["sample_idx"]))
                except Exception:
                    continue
        return done

    def drain():
        done = load_done()
        pending = []
        for r in read_gen_rows(args.gen):
            key = (r["id"], r["top_p"], r["sample_idx"])
            if key in done or r["id"] not in qmeta:
                continue
            pending.append(r)
            done.add(key)  # dedupe within this scan
        if not pending:
            return 0
        n_done = 0
        with open(args.verdicts, "a") as vf:
            for b in range(0, len(pending), args.batch):
                chunk = pending[b:b + args.batch]
                convs = []
                for r in chunk:
                    ensure_row(r["id"])
                    cot = split_cot(r["text"])
                    convs.append(build_judge_conv(row_cache, r["id"], r["gold"], cot,
                                                  args.max_cot_chars, reveal_gold=not args.no_gold))
                outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False})
                for r, o in zip(chunk, outs):
                    sound, raw = parse_verdict(o.outputs[0].text)
                    rec = {"id": r["id"], "top_p": r["top_p"],
                           "sample_idx": r["sample_idx"], "sound": sound, "raw": raw}
                    if raw == "UNPARSED":  # keep a tail for diagnosis (rare)
                        rec["fr"] = o.outputs[0].finish_reason
                        rec["tail"] = o.outputs[0].text[-200:]
                    vf.write(json.dumps(rec) + "\n")
                    n_done += 1
                vf.flush()
                print(f"[judge] +{len(chunk)} verdicts (total this drain {n_done}/{len(pending)})",
                      flush=True)
        return n_done

    if not args.watch:
        t0 = time.time()
        n = drain()
        print(f"[judge] done: {n} verdicts in {(time.time()-t0)/60:.1f}m -> {args.verdicts}",
              flush=True)
        return

    print(f"[judge] WATCH mode: polling every {args.poll}s; touch {args.stop_file} to finish",
          flush=True)
    idle = 0
    while True:
        n = drain()
        if n == 0:
            idle += 1
            if Path(args.stop_file).exists():
                print("[judge] stop-file present and nothing left to judge -> exiting", flush=True)
                break
            time.sleep(args.poll)
        else:
            idle = 0
    print(f"[judge] watch complete -> {args.verdicts}", flush=True)

if __name__ == "__main__":
    main()
