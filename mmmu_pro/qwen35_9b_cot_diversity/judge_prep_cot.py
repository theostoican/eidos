#!/usr/bin/env python
"""Build judge packets for the Sonnet CoT-soundness judge.

For every extracted CoT sample we emit:
  - the question image(s), saved once per question id  (judge_cot/img/<id>_<k>.png)
  - the CoT text, one file per sample                  (judge_cot/cots/<id>__p<top_p>__s<idx>.txt)
  - a task manifest line                               (judge_cot/tasks.jsonl)

The judge (a vision LLM, here Claude Sonnet spawned as subagents) reads the image(s)
+ the CoT and returns one verdict line into outputs/verdicts_cot.jsonl:

    {"id": ..., "top_p": ..., "sample_idx": ..., "sound": true|false}

which is exactly the schema analyze_cot.py consumes. STRICT rubric (mirrors the premise
experiment's strict-binary rule): a CoT is "sound" only if BOTH the visual reads it makes
off the image AND the inferential steps that follow are correct. One decisive misread or
one invalid logical step -> unsound, regardless of whether the final letter happens to be
right (a lucky-guess / right-for-wrong-reasons trace is UNSOUND).

Truncated samples (finish_reason='length', no closing </think>, pred=null) are still
judged on the reasoning they DID produce -- pass --skip-truncated to drop them instead.
"""
import argparse, ast, io, json
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]
IMG_KEYS = [f"image_{i}" for i in range(1, 8)]

def load_options(row):
    return ast.literal_eval(row["options"]) if isinstance(row["options"], str) else row["options"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="outputs/questions.json")
    ap.add_argument("--extracted", default="outputs/cot_extracted.jsonl")
    ap.add_argument("--outdir", default="outputs/judge_cot")
    ap.add_argument("--skip-truncated", action="store_true",
                    help="drop samples with no parsed answer (finish_reason=length)")
    args = ap.parse_args()

    qmeta = json.load(open(args.questions))
    rows = [json.loads(l) for l in open(args.extracted)]

    outdir = Path(args.outdir)
    (outdir / "img").mkdir(parents=True, exist_ok=True)
    (outdir / "cots").mkdir(parents=True, exist_ok=True)

    # Save each question's image(s) once, keyed by ds_index from questions.json.
    from datasets import load_dataset
    print("[judge_prep] loading MMMU_Pro standard (10 options) test for images ...", flush=True)
    ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")

    img_paths, opt_text = {}, {}
    for qid, meta in qmeta.items():
        row = ds[meta["ds_index"]]
        paths = []
        for k, key in enumerate(IMG_KEYS, start=1):
            im = row.get(key)
            if im is None:
                continue
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            p = outdir / "img" / f"{qid}_{k}.png"
            im.save(p, format="PNG")
            paths.append(str(p))
        img_paths[qid] = paths
        opts = load_options(row)
        opt_text[qid] = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))

    n_tasks = n_skip = 0
    with open(outdir / "tasks.jsonl", "w") as f:
        for r in rows:
            qid = r["id"]
            if qid not in qmeta:
                continue
            if args.skip_truncated and r["pred"] is None:
                n_skip += 1
                continue
            cot_path = outdir / "cots" / f"{qid}__p{r['top_p']}__s{r['sample_idx']}.txt"
            cot_path.write_text(r["cot"] or "")
            f.write(json.dumps({
                "id": qid, "top_p": r["top_p"], "sample_idx": r["sample_idx"],
                "gold": r["gold"], "n_options": r["n_options"], "pred": r["pred"],
                "finish_reason": r["finish_reason"], "cot_chars": r["cot_chars"],
                "images": img_paths[qid], "options_text": opt_text[qid],
                "cot_path": str(cot_path),
            }) + "\n")
            n_tasks += 1

    print(f"[judge_prep] {len(qmeta)} questions | {n_tasks} judge tasks "
          f"({n_skip} truncated skipped) -> {outdir}/tasks.jsonl", flush=True)
    print(f"[judge_prep] images -> {outdir}/img/  cots -> {outdir}/cots/", flush=True)

if __name__ == "__main__":
    main()
