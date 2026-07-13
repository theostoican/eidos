#!/usr/bin/env python
"""Prep for the independent Sonnet soundness judge. For each (id, top_p) at sample_idx=0
(paired across top_p), writes a SELF-CONTAINED task file the Sonnet subagent will read:
  outputs/sonnet_judge/task_<i>.txt  ->  ID/TOPP header, question, options, IMAGE path(s), CoT
GOLD IS NOT INCLUDED (independent judgement, avoids the Qwen anchoring). Images are saved
alongside; each task file names its image path(s) for the agent to Read."""
import json, ast, glob
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]

def load_options(row):
    return ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]

def split_cot(text):
    if "</think>" in text: return text.split("</think>", 1)[0].replace("<think>", "").strip()
    if "<think>" in text: return text.split("<think>", 1)[1].strip()
    return text.strip()

qmeta = {}
for f in glob.glob("outputs/sonnet_q.shard*.json"):
    qmeta.update(json.load(open(f)))

rows = []
for f in glob.glob("outputs/sonnet_gen.shard*.jsonl"):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass

from datasets import load_dataset
ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")

outdir = Path("outputs/sonnet_judge"); (outdir / "img").mkdir(parents=True, exist_ok=True)

qinfo = {}
for qid, m in qmeta.items():
    row = ds[m["ds_index"]]
    imgs = []
    for k in range(1, 8):
        im = row.get(f"image_{k}")
        if im is None: continue
        if im.mode not in ("RGB", "L"): im = im.convert("RGB")
        p = outdir / "img" / f"{qid}_{k}.png"; im.save(p); imgs.append(str(p.resolve()))
    opts = load_options(row)
    qinfo[qid] = {"question": row["question"],
                  "options": "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)),
                  "images": imgs}

index = []
i = 0
for r in sorted(rows, key=lambda r: (r["id"], r["top_p"])):
    if r["sample_idx"] != 0: continue
    qid = r["id"]; qi = qinfo[qid]
    cot = split_cot(r["text"])[:38000]
    body = (f"ID: {qid}\nTOPP: {r['top_p']}\n\n"
            f"=== QUESTION ===\n{qi['question']}\n\n=== OPTIONS ===\n{qi['options']}\n\n"
            f"=== IMAGE FILE(S) TO READ ===\n" + "\n".join(qi["images"]) +
            f"\n\n=== REASONING TRACE TO GRADE ===\n{cot}\n=== END TRACE ===\n")
    (outdir / f"task_{i}.txt").write_text(body)
    index.append({"i": i, "id": qid, "top_p": r["top_p"]})
    i += 1

json.dump(index, open(outdir / "index.json", "w"))
print(f"[sonnet_prep] wrote {i} task files ({len(qmeta)} questions x "
      f"{len(set(x['top_p'] for x in index))} top_p) -> {outdir}/task_*.txt  (+ index.json)")
