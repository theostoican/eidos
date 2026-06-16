#!/usr/bin/env python
"""Split a premises_v*.jsonl into one judge packet per question.

Each packet (outputs/judge_v5/<id>.json) holds everything a judge needs:
question text, options, gold letter, image paths, and the list of premise
samples ({top_p, sample_idx, text}). A judge labels each sample correct/
incorrect; verdicts are later merged by judge_merge.py.
"""
import argparse, json, collections
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", default="/workspace/mmmupro_qwen3vl/outputs/premises_v5.jsonl")
    ap.add_argument("--questions", default="/workspace/mmmupro_qwen3vl/outputs/questions_v5.json")
    ap.add_argument("--out-dir", default="/workspace/mmmupro_qwen3vl/outputs/judge_v5")
    args = ap.parse_args()

    qmeta = json.load(open(args.questions))
    rows = [json.loads(l) for l in open(args.premises)]
    by_q = collections.defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    index = []
    for qid, samp in by_q.items():
        q = qmeta[qid]
        samp.sort(key=lambda r: (r["top_p"], r["sample_idx"]))
        packet = {
            "id": qid, "subject": q["subject"], "gold": q["gold"],
            "question": q["question"], "options": q["options"],
            "image_paths": q["image_paths"],
            "premises": [{"top_p": r["top_p"], "sample_idx": r["sample_idx"],
                          "text": r["text"].strip()} for r in samp],
        }
        p = out / f"{qid}.json"
        json.dump(packet, open(p, "w"), indent=2)
        index.append({"id": qid, "subject": q["subject"], "n_premises": len(samp),
                      "packet": str(p), "image_paths": q["image_paths"]})
    json.dump(index, open(out / "_index.json", "w"), indent=2)
    print(f"[prep] {len(index)} packets -> {out}")
    for it in index:
        print(f"  {it['id']:42s} {it['n_premises']:3d} premises  imgs={len(it['image_paths'])}")

if __name__ == "__main__":
    main()
