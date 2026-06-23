#!/usr/bin/env python
"""Comprehensive single-premise sweep: ONE premise per sample that incorporates
ALL visual facts in the image (vs. the numbered-list enumeration, or a single
narrow fact). Same sweep: top_p {0.5,0.7,0.9,0.95,1.0} x 16, thinking ON,
temp 1.0, top_k off, 16384-token budget.
"""
import argparse, base64, io, json, re, time
from pathlib import Path
from PIL import Image

IMG_MARK = re.compile(r"<image\s+(\d+)>")

def b64_image(img, fmt="PNG"):
    buf = io.BytesIO()
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

PREMISE_HEADER = (
    "You are given an image and the question asked about it (below). Read the image "
    "carefully and write ONE comprehensive visual premise that incorporates EVERY fact "
    "you can read directly from the image — every element, its identity, any "
    "text/label/number/symbol (read exactly, character-by-character), its position, and "
    "how it relates to the others — all combined into a single premise statement.\n"
    "IMPORTANT: Do NOT answer, solve, compute, or work toward the answer. Do NOT pick an "
    "option. Only state the visual premise.\n"
    "Reason briefly if needed, then write the premise on ONE line (it may be long) as: "
    "'Premise: <a single statement capturing all the visual facts>'.\n\n"
)

def build(question, image_paths):
    images = {i + 1: Image.open(p) for i, p in enumerate(image_paths)}
    content = [{"type": "text", "text": PREMISE_HEADER}, {"type": "text", "text": "Question: "}]
    pos = 0
    for m in IMG_MARK.finditer(question):
        pre = question[pos:m.start()]
        if pre.strip():
            content.append({"type": "text", "text": pre})
        idx = int(m.group(1))
        if idx in images:
            content.append({"type": "image_url", "image_url": {"url": b64_image(images[idx])}})
        else:
            content.append({"type": "text", "text": m.group(0)})
        pos = m.end()
    tail = question[pos:]
    if tail.strip():
        content.append({"type": "text", "text": tail})
    for idx in sorted(images):
        if idx not in {int(x) for x in IMG_MARK.findall(question)}:
            content.append({"type": "image_url", "image_url": {"url": b64_image(images[idx])}})
    return content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="outputs/questions.json")
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--top-ps", default="0.5,0.7,0.9,0.95,1.0")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--gpu-mem-util", type=float, default=0.93)
    ap.add_argument("--max-num-seqs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="outputs/premises_comp.jsonl")
    args = ap.parse_args()
    top_ps = [float(x) for x in args.top_ps.split(",")]

    from vllm import LLM, SamplingParams
    qmeta = json.load(open(args.questions))
    qids = sorted(qmeta)
    print(f"[select] {len(qids)} questions:", qids, flush=True)

    convs, meta = [], []
    for qid in qids:
        q = qmeta[qid]
        content = build(q["question"], q["image_paths"])
        for p in top_ps:
            convs.append([{"role": "user", "content": content}])
            meta.append({"id": qid, "subject": q["subject"], "gold": q["gold"], "top_p": p})

    llm = LLM(model="Qwen/Qwen3.5-9B", dtype="bfloat16",
              gpu_memory_utilization=args.gpu_mem_util,
              max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 8, "video": 0}, trust_remote_code=True,
              seed=args.seed)

    t0 = time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in top_ps:
            idxs = [i for i, m in enumerate(meta) if m["top_p"] == p]
            sp = SamplingParams(n=args.n_samples, temperature=args.temperature,
                                top_p=p, top_k=-1, max_tokens=args.max_tokens, seed=args.seed)
            outs = llm.chat([convs[i] for i in idxs], sp,
                            chat_template_kwargs={"enable_thinking": True})
            for i, o in zip(idxs, outs):
                m = meta[i]
                for s_idx, comp in enumerate(o.outputs):
                    f.write(json.dumps({
                        "id": m["id"], "subject": m["subject"], "top_p": p,
                        "sample_idx": s_idx, "gold": m["gold"],
                        "out_tokens": len(comp.token_ids), "finish_reason": comp.finish_reason,
                        "text": comp.text,
                    }) + "\n")
            f.flush()
            n_trunc = sum(1 for i, o in zip(idxs, outs) for c in o.outputs if c.finish_reason != "stop")
            print(f"[gen] top_p={p} done | {len(idxs)}x{args.n_samples} | trunc={n_trunc} | "
                  f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
    print(f"[done] {time.time()-t0:.0f}s -> {args.out}", flush=True)

if __name__ == "__main__":
    main()
