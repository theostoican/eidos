#!/usr/bin/env python
"""ViRL39K -> data/pool.jsonl: the single-image multiple-choice questions with a single-letter answer.
  python data.py [--root data/virl39k]
The order is a fixed shuffle (Random(0)); training with seed 0 walks it in this order. Positions 7600-8399 are the
in-distribution held-out set (split = 'heldout'): training never draws them, evaluate.py --set virl scores them."""
import argparse, json, random, re
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
HELDOUT = range(7600, 8400)

ap = argparse.ArgumentParser(); ap.add_argument("--root", default=str(HERE / "data" / "virl39k")); a = ap.parse_args()
df = pd.read_parquet(Path(a.root) / "39Krelease.parquet")
out = []
for r in df.itertuples():
    if len(r.image) != 1: continue
    m = re.fullmatch(r"\\boxed\{([A-J])\}", r.answer.strip())
    if not m: continue
    letters = sorted(set(re.findall(r"(?:^|\s)\(?([A-J])(?:\)|\.)\s", r.question)))
    if len(letters) < 2 or letters != list("ABCDEFGHIJ"[:len(letters)]) or m.group(1) not in letters: continue
    q = r.question.replace("<image>", "").strip()
    # drop answer-format instructions that conflict with the chain-of-thought suffix appended at sampling time
    q = "\n".join(l for l in q.split("\n") if not re.search(r"(?i)directly answer|answer with the option|answer the question using a single|^hint:", l)).strip()
    out.append({"qid": r.qid, "question": q, "gold": m.group(1), "n_options": len(letters), "image": r.image[0],
                "category": r.category, "pass32": r.PassRate_32BTrained})
random.Random(0).shuffle(out)
for i, x in enumerate(out): x["split"] = "heldout" if i in HELDOUT else "train"
(HERE / "data").mkdir(exist_ok=True)
with open(HERE / "data" / "pool.jsonl", "w") as f:
    for x in out: f.write(json.dumps(x) + "\n")
print(len(out), "questions;", sum(x["split"] == "heldout" for x in out), "held out;", pd.Series([x["category"] for x in out]).value_counts().to_dict())
