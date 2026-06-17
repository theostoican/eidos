#!/usr/bin/env python
"""Extract the final premise sentence from each raw generation.

`premise_gen_v6.py` writes the model's FULL output (the <think>...</think> trace
plus prose plus the final `Premise:` line) to premises_v6.jsonl. This script
parses out just the premise sentence so downstream tooling (and humans) can read
the one-line visual fact without the reasoning trace.

Extraction rule (matches the committed judge_v6/<id>.compact.json files exactly):
  has_premise = the literal marker 'Premise:' appears in the text
  premise     = the text after the LAST 'Premise:' marker, stripped, FIRST LINE
                only (the model sometimes keeps reasoning on later lines); '' if
                the marker is absent (e.g. a generation truncated before it).

Outputs (one per question, keyed off questions_v6.json), mirroring the existing
compact packets:
  <out-dir>/<id>.compact.json  {id, subject, gold, question, options, image_path,
                                premises:[{top_p, sample_idx, has_premise, premise}]}
and a flat JSONL with every extracted premise:
  <out>  {id, top_p, sample_idx, has_premise, premise}
"""
import argparse, json, collections, os


def extract(text):
    """(has_premise, premise) — see module docstring."""
    has = "Premise:" in text
    if not has:
        return False, ""
    return True, text.split("Premise:")[-1].strip().split("\n")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--premises", default="outputs/premises_v6.jsonl",
                    help="raw generations from premise_gen_v6.py")
    ap.add_argument("--questions", default="outputs/questions_v6.json",
                    help="per-question metadata (subject/options/gold/image_paths)")
    ap.add_argument("--out", default="outputs/premises_v6_extracted.jsonl",
                    help="flat JSONL of extracted premises")
    ap.add_argument("--compact-dir", default=None,
                    help="if set, also write one <id>.compact.json per question here")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.premises)]
    qmeta = json.load(open(args.questions)) if os.path.exists(args.questions) else {}

    n_has = 0
    by_q = collections.defaultdict(list)
    with open(args.out, "w") as f:
        for r in rows:
            has, prem = extract(r["text"])
            n_has += has
            rec = {"id": r["id"], "top_p": r["top_p"], "sample_idx": r["sample_idx"],
                   "has_premise": has, "premise": prem}
            f.write(json.dumps(rec) + "\n")
            by_q[r["id"]].append(rec)

    print(f"[extract] {len(rows)} generations -> {args.out} | "
          f"with premise: {n_has} ({n_has/max(len(rows),1):.1%})")

    if args.compact_dir:
        os.makedirs(args.compact_dir, exist_ok=True)
        for qid, recs in by_q.items():
            q = qmeta.get(qid, {})
            recs.sort(key=lambda x: (x["top_p"], x["sample_idx"]))
            img = q.get("image_paths") or []
            packet = {
                "id": qid, "subject": q.get("subject"), "gold": q.get("gold"),
                "question": q.get("question"), "options": q.get("options"),
                "image_path": img[0] if img else None,
                "premises": [{"top_p": x["top_p"], "sample_idx": x["sample_idx"],
                              "has_premise": x["has_premise"], "premise": x["premise"]}
                             for x in recs],
            }
            json.dump(packet, open(os.path.join(args.compact_dir, f"{qid}.compact.json"), "w"),
                      indent=2, ensure_ascii=False)
        print(f"[extract] wrote {len(by_q)} compact packets -> {args.compact_dir}")


if __name__ == "__main__":
    main()
