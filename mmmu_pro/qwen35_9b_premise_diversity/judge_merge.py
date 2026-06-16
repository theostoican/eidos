#!/usr/bin/env python
"""Merge per-question judge verdict files into one verdicts_v5.jsonl.

Each judge writes outputs/judge_v5/<id>.verdicts.json:
  {"id":..., "verdicts":[{"top_p":..., "sample_idx":..., "correct":true/false}, ...]}
This flattens them to the jsonl schema premise_analyze.py expects.
"""
import argparse, json, glob, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/workspace/mmmupro_qwen3vl/outputs/judge_v5")
    ap.add_argument("--out", default="/workspace/mmmupro_qwen3vl/outputs/verdicts_v5.jsonl")
    args = ap.parse_args()

    n, files = 0, sorted(glob.glob(os.path.join(args.dir, "*.verdicts.json")))
    with open(args.out, "w") as f:
        for fp in files:
            d = json.load(open(fp))
            for v in d["verdicts"]:
                f.write(json.dumps({"id": d["id"], "top_p": float(v["top_p"]),
                                    "sample_idx": int(v["sample_idx"]),
                                    "correct": bool(v["correct"])}) + "\n")
                n += 1
    print(f"[merge] {len(files)} files, {n} verdicts -> {args.out}")

if __name__ == "__main__":
    main()
