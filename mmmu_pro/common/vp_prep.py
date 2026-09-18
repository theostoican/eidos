#!/usr/bin/env python
"""Stage 1: turn one arm's committed traces into per-sample work layers, and map every
question id to its MMMU-Pro row. The arm (glob, temperature, grid) is given on the command
line -- normally by run_vp_arm.sh from the arm directory's vp_arm.sh.

The premise arm needs the <think> reasoning text, which analyze.py never touches (it only
parses the final letter). This decompresses the traces once rather than once per pass.

LAYERED OUTPUT: one file per sample_idx. A layer is a COMPLETE, BALANCED dataset over the
full question x top_p grid, so stopping after any layer leaves a curve computed on the same cells at
every top_p -- never a partial grid deeper at one end of the swept axis.

TRUNCATED TRACES ARE KEPT for premise extraction (they still made visual reads), but they are
SCORED WRONG for accuracy -- see the ballot rule below.
"""
import argparse, collections, glob, gzip, json, re
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]
ANS_RE = re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)


def parse_answer(text, n_options):
    """Identical to analyze.py's parser -- the premise arm must agree with the accuracy
    curve it is plotted against, question for question."""
    valid = set(LETTERS[:n_options])
    after = text.split("</think>")[-1] if "</think>" in text else text
    for c in reversed(ANS_RE.findall(after) or ANS_RE.findall(text)):
        if c.upper() in valid:
            return c.upper()
    for ch in reversed(re.findall(r"\b([A-J])\b", after)):
        if ch in valid:
            return ch
    return None


def split_cot(text):
    """Qwen3.5's template emits the opening <think> in the PROMPT, so generated text is
    [reasoning] </think> [answer]: the CLOSING tag is the delimiter."""
    if "</think>" in text:
        pre, post = text.split("</think>", 1)
        return pre.replace("<think>", "").strip(), post.strip(), True
    if "<think>" in text:
        return text.split("<think>", 1)[1].strip(), "", False
    return text.strip(), "", False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="the arm's trace files, e.g. 'cots/t16_sweep.shard*.jsonl.gz'")
    ap.add_argument("--temperature", type=float, required=True)
    ap.add_argument("--grid", required=True, help="comma-separated top_p levels of the arm")
    ap.add_argument("--outdir", default="outputs/vp_layers")
    ap.add_argument("--questions-out", default="outputs/vp_q.json")
    ap.add_argument("--no-dataset", action="store_true")
    a = ap.parse_args()
    grid = {float(x) for x in a.grid.split(",")}

    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    writers, counts, meta = {}, collections.Counter(), {}
    cfgs, n_read = set(), 0
    for f in sorted(glob.glob(a.glob)):
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            n_read += 1
            if r.get("temperature", 1.0) != a.temperature or r["top_p"] not in grid:
                continue
            cfgs.add(r.get("cfg_profile", "<unstamped>"))
            cot, ans_text, has_think = split_cot(r["text"])
            # THE BALLOT RULE (analyze.py): a ballot is valid only if the trace TERMINATED.
            # A truncated trace often still contains a stray letter that the bare-letter
            # fallback would read as an answer -- counting those lifts accuracy by up to
            # 2.6 pp (Qwen T=1.0), most of it at low top_p where truncation is commonest, i.e. exactly
            # the axis-dependent bias the ballot rule exists to prevent.
            stopped = r.get("finish_reason") == "stop"
            pred = parse_answer(r["text"], r.get("n_options", 10)) if stopped else None
            s = r["sample_idx"]
            if s not in writers:
                writers[s] = gzip.open(f"{a.outdir}/layer{s:02d}.jsonl.gz", "wt")
            writers[s].write(json.dumps({
                "id": r["id"], "subject": r.get("subject"), "top_p": r["top_p"],
                "sample_idx": s, "gold": r["gold"], "n_options": r.get("n_options", 10),
                "finish_reason": r.get("finish_reason"), "has_think": has_think,
                "pred": pred, "answer_correct": (pred == r["gold"]),
                "cot_chars": len(cot), "cot": cot,
            }) + "\n")
            counts[s] += 1
            meta.setdefault(r["id"], {"subject": r.get("subject"), "gold": r["gold"],
                                      "n_options": r.get("n_options", 10)})
    for w in writers.values():
        w.close()
    if len(cfgs) > 1:
        raise SystemExit(f"[prep] traces span config profiles {sorted(cfgs)}; refuse to mix.")
    n_q, n_p = len(meta), len(grid)
    print(f"[prep] {n_read} rows read | {sum(counts.values())} kept "
          f"(T={a.temperature}, {n_p} top_p) | cfg={sorted(cfgs)}")
    bad = {s: c for s, c in counts.items() if c != n_q * n_p}
    print(f"[prep] {len(counts)} layers x {n_q} questions x {n_p} top_p"
          + (f" | UNBALANCED: {bad}" if bad else " | every layer balanced"))

    if not a.no_dataset:
        from datasets import load_dataset
        ds = load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test")
        idx = {qid: i for i, qid in enumerate(ds["id"])}
        missing = [q for q in meta if q not in idx]
        if missing:
            raise SystemExit(f"[prep] {len(missing)} trace ids absent from the dataset")
        for q in meta:
            meta[q]["ds_index"] = idx[q]
        json.dump(meta, open(a.questions_out, "w"), indent=1)
        print(f"[prep] wrote {len(meta)} question meta -> {a.questions_out}")


if __name__ == "__main__":
    main()
