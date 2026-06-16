#!/usr/bin/env python
"""Premise sweep v6: THINKING ENABLED, but bounded so the whole output
(<think> trace + premise) fits well inside max_tokens -> no truncated/rejected
samples. 24 examples, otherwise identical config to v3/v4/v5.

Difference vs v5: thinking is ON (enable_thinking=True). The prompt asks for a
brief reasoning trace and a short final premise so generations finish naturally
(finish_reason='stop') below max_tokens=2048.
"""
import argparse, ast, json, time
from pathlib import Path
from run_mmmupro import b64_image, IMG_MARK, LETTERS

PREMISE_HEADER = (
    "Given the image and the question, identify ONE visual premise that could help "
    "solve the question.\n"
    "Reason VERY briefly: at most 2-3 short sentences. Do NOT compute, derive, or "
    "work toward the final answer, and do NOT restate the question or list every "
    "value in the image.\n"
    "Then write the premise on its own line as: 'Premise: <one short sentence "
    "stating a single visual fact read directly from the image>'.\n"
    "Keep your ENTIRE reply short (well under 200 words total, reasoning included).\n\n"
)

def build_premise_content(question, images):
    content = [{"type": "text", "text": PREMISE_HEADER}]
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
    referenced = {int(x) for x in IMG_MARK.findall(question)}
    for idx in sorted(images):
        if idx not in referenced:
            content.append({"type": "image_url", "image_url": {"url": b64_image(images[idx])}})
    return content

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/workspace/mmmupro_qwen3vl/outputs/results.jsonl")
    ap.add_argument("--n-questions", type=int, default=24)
    ap.add_argument("--n-samples", type=int, default=16)
    ap.add_argument("--top-ps", default="0.5,0.7,0.9,0.95,1.0")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--max-model-len", type=int, default=14336)
    ap.add_argument("--min-len", type=int, default=1500)
    ap.add_argument("--max-len", type=int, default=7000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="/workspace/mmmupro_qwen3vl/outputs/premises_v6.jsonl")
    ap.add_argument("--img-dir", default="/workspace/mmmupro_qwen3vl/outputs/images")
    ap.add_argument("--meta-out", default="/workspace/mmmupro_qwen3vl/outputs/questions_v6.json")
    args = ap.parse_args()
    top_ps = [float(x) for x in args.top_ps.split(",")]

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in open(args.results)]
    fails = [r for r in rows if (not r["correct"]) and r["finish_reason"] == "stop"
             and r["pred"] is not None and args.min_len <= r["out_tokens"] <= args.max_len]
    fails.sort(key=lambda r: r["id"])
    picked, seen = [], set()
    for r in fails:
        if r["subject"] not in seen:
            picked.append(r); seen.add(r["subject"])
        if len(picked) >= args.n_questions:
            break
    print(f"[select] {len(picked)} questions:", [r["id"] for r in picked], flush=True)

    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
    by_id = {row["id"]: row for row in ds}

    img_dir = Path(args.img_dir); img_dir.mkdir(parents=True, exist_ok=True)
    convs, meta, qmeta = [], [], {}
    for r in picked:
        row = by_id[r["id"]]
        opts = ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]
        images = {i: row[f"image_{i}"] for i in range(1, 8) if row.get(f"image_{i}") is not None}
        img_paths = []
        for idx, im in images.items():
            p = img_dir / f"{r['id']}_img{idx}.png"
            im.convert("RGB").save(p)
            img_paths.append(str(p))
        qmeta[r["id"]] = {"subject": r["subject"], "question": row["question"],
                          "options": {LETTERS[i]: o for i, o in enumerate(opts)},
                          "gold": r["gold"], "image_paths": img_paths}
        content = build_premise_content(row["question"], images)
        for p in top_ps:
            convs.append([{"role": "user", "content": content}])
            meta.append({"id": r["id"], "subject": r["subject"], "gold": r["gold"],
                         "n_options": len(opts), "top_p": p})
    json.dump(qmeta, open(args.meta_out, "w"), indent=2)
    print(f"[meta] wrote {args.meta_out} ({len(qmeta)} questions)", flush=True)

    llm = LLM(model="Qwen/Qwen3.5-9B", dtype="bfloat16", gpu_memory_utilization=0.95,
              max_num_seqs=32, max_num_batched_tokens=4096, max_model_len=args.max_model_len,
              limit_mm_per_prompt={"image": 8, "video": 0}, trust_remote_code=True, seed=args.seed)

    t0 = time.time()
    with open(args.out, "w") as f:
        for p in top_ps:
            idxs = [i for i, m in enumerate(meta) if m["top_p"] == p]
            sp = SamplingParams(n=args.n_samples, temperature=args.temperature,
                                top_p=p, top_k=-1, max_tokens=args.max_tokens, seed=args.seed)
            # thinking ON (Qwen3.5 default); be explicit
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
