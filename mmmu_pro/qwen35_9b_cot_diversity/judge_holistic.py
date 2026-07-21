#!/usr/bin/env python
"""HOLISTIC premise judging — replicates the sibling qwen35_9b_premise_diversity design.

Why this exists. Judging ATOMIC premises individually saturates: each claim ("the waveform
is a triangle pulse", "at t=0 the value is v_o") is trivially easy, so soundness pins at
97-100% and top_p cannot move it. The sibling experiment avoided this by judging ONE
COMPREHENSIVE premise per generation -- a dense description where ANY wrong detail fails
the whole thing -- and got 56% overall with a real top_p trend (pearson -0.15).

Here we reconstruct that: all visual premises of a trace are presented together as ONE
composite description, and the judge must confirm EVERY detail. Any single error => INCORRECT.
This restores dynamic range without regenerating anything.

NO GOLD ANSWER is shown (avoids the outcome-anchoring that made the Qwen self-judge circular).

Input : outputs/u_premises.jsonl
Output: outputs/u_verdicts_holistic.jsonl  {id, top_p, sample_idx, n_premises, correct, wrong, raw}
"""
import argparse, ast, base64, collections, glob, io, json, re
from pathlib import Path

LETTERS=[chr(ord("A")+i) for i in range(26)]
IMG_MARK=re.compile(r"<image\s+(\d+)>")
VERD=re.compile(r"VERDICT:\s*(CORRECT|INCORRECT)", re.IGNORECASE)

def b64(img, fmt="PNG"):
    b=io.BytesIO()
    if img.mode not in ("RGB","L"): img=img.convert("RGB")
    img.save(b,format=fmt)
    return f"data:image/{fmt.lower()};base64,"+base64.b64encode(b.getvalue()).decode()

def load_options(row):
    return ast.literal_eval(row["options"]) if isinstance(row["options"],str) else row["options"]

RUBRIC="""You are a strict visual fact-checker. You are shown an image (or several) and a numbered list of claims that another model made about what it saw while reasoning about the image.

Treat the list as ONE composite description. Verify EVERY claim against the image.

- If EVERY claim is accurate, answer CORRECT.
- If ANY SINGLE claim is wrong -- a misread number, a mislabeled part, an invented object, a wrong count/position/colour, a detail not actually present -- answer INCORRECT.

Be strict and literal. A claim asserting a specific value, label, count or position is only accurate if you can confirm that exact detail in the image. "Roughly right" is wrong. Ignore claims that are pure arithmetic or inference (they are not about the image) -- judge only the visual ones. Do NOT try to answer the original question.

Reply in exactly this format:
WRONG: <the first claim number that is wrong, and what is actually shown -- or "none">
VERDICT: CORRECT or INCORRECT
"""

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-38B-AWQ")
    ap.add_argument("--quantization", default="awq")
    ap.add_argument("--premises", default="outputs/u_premises.jsonl")
    ap.add_argument("--questions", default="outputs/u_q.json")
    ap.add_argument("--out", default="outputs/u_verdicts_holistic.jsonl")
    ap.add_argument("--max-premises", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0)
    args=ap.parse_args()

    qmeta={}
    for f in sorted(glob.glob(args.questions.replace(".json",".shard*.json")) or [args.questions]):
        qmeta.update(json.load(open(f)))
    rows=[json.loads(l) for l in open(args.premises)]
    rows=[r for r in rows if r["n_premises"]>0]
    if args.limit: rows=rows[:args.limit]
    print(f"[holistic] {len(rows)} traces | {len(qmeta)} questions", flush=True)

    from datasets import load_dataset
    ds=load_dataset("MMMU/MMMU_Pro","standard (10 options)",split="test")
    qc={}
    for qid in sorted({r["id"] for r in rows}):
        row=ds[qmeta[qid]["ds_index"]]
        text=IMG_MARK.sub("[image]", row["question"])
        order=[int(n) for n in IMG_MARK.findall(row["question"])]
        imgs={i:row.get(f"image_{i}") for i in range(1,8) if row.get(f"image_{i}") is not None}
        if not order: order=sorted(imgs)
        qc[qid]={"q":text,"b64":[b64(imgs[i]) for i in order if i in imgs]}

    from vllm import LLM, SamplingParams
    kw={"quantization":args.quantization} if args.quantization else {}
    llm=LLM(model=args.model,dtype="auto",gpu_memory_utilization=args.gpu_mem_util,
            max_num_seqs=args.max_num_seqs,max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            limit_mm_per_prompt={"image":8,"video":0},trust_remote_code=True,seed=1234,**kw)
    sp=SamplingParams(temperature=0.0,max_tokens=args.max_tokens)

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    nc=nt=0
    with open(args.out,"w") as fo:
        for s in range(0,len(rows),args.batch):
            ch=rows[s:s+args.batch]; convs=[]
            for r in ch:
                q=qc[r["id"]]; prem=r["premises"][:args.max_premises]
                txt=RUBRIC+f"\n=== QUESTION (context only) ===\n{q['q']}\n\n=== CLAIMS ABOUT THE IMAGE ===\n"+ \
                    "\n".join(f"{i+1}. {p}" for i,p in enumerate(prem))+"\n"
                cont=[{"type":"text","text":txt}]+[{"type":"image_url","image_url":{"url":b}} for b in q["b64"]]
                convs.append([{"role":"user","content":cont}])
            try: outs=llm.chat(convs,sp)
            except Exception as e:
                print(f"[holistic] batch failed ({type(e).__name__}); one-by-one",flush=True)
                outs=[]
                for cv in convs:
                    try: outs.append(llm.chat([cv],sp)[0])
                    except Exception: outs.append(None)
            for r,o in zip(ch,outs):
                if o is None:
                    fo.write(json.dumps({"id":r["id"],"top_p":r["top_p"],"sample_idx":r["sample_idx"],
                        "n_premises":r["n_premises"],"correct":None,"wrong":"","raw":"SKIPPED"})+"\n"); continue
                raw=o.outputs[0].text; m=VERD.search(raw)
                corr=(m.group(1).upper()=="CORRECT") if m else None
                wrong=""
                wm=re.search(r"WRONG:\s*(.+)",raw)
                if wm: wrong=wm.group(1).strip()[:200]
                if corr is not None: nt+=1; nc+=int(corr)
                fo.write(json.dumps({"id":r["id"],"top_p":r["top_p"],"sample_idx":r["sample_idx"],
                    "n_premises":r["n_premises"],"correct":corr,"wrong":wrong,
                    "raw":"" if corr is not None else raw[:300]})+"\n")
            fo.flush()
            print(f"[holistic] {min(s+args.batch,len(rows))}/{len(rows)} | correct {nc}/{nt} = "
                  f"{nc/max(nt,1):.1%}",flush=True)
    print(f"[holistic] done: {nc}/{nt} = {nc/max(nt,1):.1%} -> {args.out}")

if __name__=="__main__": main()
