#!/usr/bin/env python
"""Merge per-question comprehensive verdict files -> verdicts_comp.jsonl."""
import argparse, json, glob, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/judge_comp")
    ap.add_argument("--out", default="outputs/verdicts_comp.jsonl")
    args = ap.parse_args()
    n = 0
    with open(args.out, "w") as out:
        for fp in sorted(glob.glob(f"{args.dir}/*.verdicts.json")):
            d = json.load(open(fp))
            for v in d["verdicts"]:
                out.write(json.dumps({"id": d["id"], "top_p": float(v["top_p"]),
                                      "sample_idx": int(v["sample_idx"]), "correct": bool(v["correct"])}) + "\n")
                n += 1
    print(f"[merge] {n} verdicts -> {args.out}")

if __name__ == "__main__":
    main()
