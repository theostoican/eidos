#!/usr/bin/env python
"""One judge packet per question: all comprehensive premises (top_p, sample_idx,
text) + image/question/options/gold. A vision judge scores each premise STRICT-
BINARY (correct only if every fact in it is accurate)."""
import argparse, json, collections, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", default="outputs/premises_comp_extracted.jsonl")
    ap.add_argument("--questions", default="outputs/questions.json")
    ap.add_argument("--out-dir", default="outputs/judge_comp")
    args = ap.parse_args()
    qmeta = json.load(open(args.questions))
    rows = [json.loads(l) for l in open(args.premises)]
    byq = collections.defaultdict(list)
    for r in rows:
        byq[r["id"]].append(r)
    os.makedirs(args.out_dir, exist_ok=True)
    idx = []
    for qid in sorted(byq):
        q = qmeta[qid]; samp = sorted(byq[qid], key=lambda r: (r["top_p"], r["sample_idx"]))
        packet = {"id": qid, "subject": q["subject"], "gold": q["gold"], "question": q["question"],
                  "options": q["options"], "image_path": q["image_paths"][0],
                  "premises": [{"top_p": r["top_p"], "sample_idx": r["sample_idx"], "text": r["premise"]} for r in samp]}
        json.dump(packet, open(f"{args.out_dir}/{qid}.json", "w"), indent=2, ensure_ascii=False)
        idx.append({"id": qid, "n": len(samp)})
    json.dump(idx, open(f"{args.out_dir}/_index.json", "w"), indent=2)
    print(f"[prep] {len(idx)} packets -> {args.out_dir}")

if __name__ == "__main__":
    main()
