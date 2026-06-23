#!/usr/bin/env python
"""Extract the ONE comprehensive premise from each generation: text after the
LAST 'Premise:' marker (single line). Truncated generations (finish_reason!=
'stop') dropped. One row per sample."""
import argparse, json, statistics

def extract(text):
    if "Premise:" in text:
        seg = text.split("Premise:")[-1].strip()
        return seg.split("\n")[0].strip().strip("*_ ").strip()
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", default="outputs/premises_comp.jsonl")
    ap.add_argument("--out", default="outputs/premises_comp_extracted.jsonl")
    ap.add_argument("--keep-truncated", action="store_true")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.premises)]
    n = 0; lens = []
    with open(args.out, "w") as f:
        for r in rows:
            if r["finish_reason"] != "stop" and not args.keep_truncated:
                continue
            prem = extract(r["text"])
            if not prem:
                continue
            n += 1; lens.append(len(prem.split()))
            f.write(json.dumps({"id": r["id"], "subject": r["subject"], "top_p": r["top_p"],
                                "sample_idx": r["sample_idx"], "gold": r["gold"], "premise": prem}) + "\n")
    print(f"[extract] {len(rows)} gens -> {n} clean premises | mean {statistics.mean(lens):.0f} words")

if __name__ == "__main__":
    main()
