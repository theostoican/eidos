#!/usr/bin/env python
"""Extract the <think> CoT and the final parsed answer from each generation.

Input : outputs/cot_gen.jsonl  (raw generations: full text = <think>..</think> + answer)
Output: outputs/cot_extracted.jsonl  with, per sample:
  id, subject, top_p, sample_idx, gold, n_options, out_tokens, finish_reason,
  cot           -> the reasoning text inside <think>..</think> (or all text if no close tag),
  answer_text   -> text after </think> (the visible answer),
  pred          -> parsed option letter (A-J) or null,
  answer_correct-> pred == gold,
  has_think     -> whether a proper <think>..</think> block was present.

Truncated samples (finish_reason='length') usually never close </think>; their whole
text is treated as CoT and pred is null. Kept for the diversity computation, excluded
from answer accuracy by pred=null.
"""
import argparse, json, re
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]
ANS_RE = re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)

def parse_answer(text, n_options):
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
    """Return (cot, answer_text, has_think).

    Qwen3.5's chat template emits the opening <think> as part of the PROMPT, so the
    generated text is: [reasoning] </think> [answer] -- the opening tag is normally
    absent and the CLOSING </think> is the real delimiter. Truncated samples
    (finish_reason='length') never reach </think>: the whole text is reasoning.
    """
    if "</think>" in text:
        pre, post = text.split("</think>", 1)
        return pre.replace("<think>", "").strip(), post.strip(), True
    if "<think>" in text:  # opened but not closed
        return text.split("<think>", 1)[1].strip(), "", False
    # no closing tag (truncated mid-thought) -> all reasoning, no answer text
    return text.strip(), "", False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="outputs/cot_gen.jsonl")
    ap.add_argument("--out", default="outputs/cot_extracted.jsonl")
    args = ap.parse_args()
    n = n_think = n_pred = n_correct = 0
    with open(args.gen) as fin, open(args.out, "w") as fout:
        for line in fin:
            r = json.loads(line)
            cot, ans_text, has_think = split_cot(r["text"])
            pred = parse_answer(r["text"], r.get("n_options", 10))
            correct = (pred == r["gold"]) if pred is not None else None
            n += 1; n_think += int(has_think); n_pred += int(pred is not None)
            n_correct += int(correct is True)
            fout.write(json.dumps({
                "id": r["id"], "subject": r.get("subject"), "top_p": r["top_p"],
                "sample_idx": r["sample_idx"], "gold": r["gold"],
                "n_options": r.get("n_options", 10), "out_tokens": r.get("out_tokens"),
                "finish_reason": r.get("finish_reason"),
                "cot": cot, "cot_chars": len(cot), "answer_text": ans_text,
                "pred": pred, "answer_correct": correct, "has_think": has_think,
            }) + "\n")
    print(f"[extract] {n} samples | has_think={n_think} ({n_think/n:.0%}) | "
          f"parsed_answer={n_pred} ({n_pred/n:.0%}) | answer_correct={n_correct} "
          f"({n_correct/n:.1%} of all) -> {args.out}")

if __name__ == "__main__":
    main()
