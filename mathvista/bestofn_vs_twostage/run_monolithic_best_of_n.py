import os, re, json, time, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

# RQ1 control — ARM (a): plain monolithic best-of-N, NO focus modes.
#
# Purpose: the RQ1 "fixed by >=1 of 20 focus modes" number is a best-of-20 oracle
# metric whose only source of diversity is the prompt text (decoding is greedy).
# To know whether the focus-mode framing actually adds signal — or whether ANY 20
# swings at a hard input recover a few — we need a matched best-of-N control that
# uses the SAME vanilla question prompt (no focus modes) and gets its diversity from
# temperature sampling instead. Same model, same scoring, same N.
#
# We run it at scale (100 streamed MathVista testmini examples), and for each:
#   * greedy baseline (do_sample=False)            -> defines "failure" (its own baseline)
#   * N temperature samples of the same prompt     -> best-of-N (oracle) + majority vote
# Then we report, over the failures the greedy baseline makes, how many plain
# best-of-N recovers — directly comparable to RQ1's focus-mode recovery rate.

MODEL_ID    = "Qwen/Qwen3.5-9B"
OUT_DIR     = "monolithic_best_of_n"
N_EXAMPLES  = int(os.environ.get("N_EXAMPLES", "100"))   # stream the first N testmini examples
N_SAMPLES   = int(os.environ.get("N_SAMPLES", "20"))     # best-of-N (match RQ1's 20 modes)
GEN_BATCH   = int(os.environ.get("GEN_BATCH", "1"))      # samples per generate() call (1 = sequential, safest)
MAX_NEW     = int(os.environ.get("MAX_NEW", "768"))
TEMP        = float(os.environ.get("TEMP", "0.8"))
TOP_P       = float(os.environ.get("TOP_P", "0.95"))

torch.manual_seed(0)

# ---------------- answer extraction / scoring (identical to run_eval.py / RQ1) ----------------
def last_number(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m[-1]) if m else None

def extract_pred(resp, ex):
    tail = resp.strip()
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", tail)
    boxed = boxed[-1].strip() if boxed else None
    window = tail[-200:]
    if ex["question_type"] == "multi_choice":
        choices = ex["choices"]
        letters = {chr(65 + i): c for i, c in enumerate(choices)}
        if boxed and boxed.upper() in letters:
            return letters[boxed.upper()]
        if boxed:
            for c in choices:
                if str(c).strip().lower() == boxed.lower():
                    return c
        for L in reversed(re.findall(r"\b([A-D])\b", window)):
            if L in letters:
                return letters[L]
        hit = None
        for c in choices:
            if str(c).strip().lower() in tail.lower():
                hit = c
        return hit if hit is not None else window[-40:]
    if ex["answer_type"] in ("integer", "float"):
        if boxed and last_number(boxed) is not None:
            return last_number(boxed)
        return last_number(window)
    if boxed:
        return boxed
    return window.splitlines()[-1].strip() if window.strip() else ""

def is_correct(pred, ex):
    gt = ex["answer"]
    if ex["question_type"] == "multi_choice":
        return str(pred).strip().lower() == str(gt).strip().lower()
    if ex["answer_type"] in ("integer", "float"):
        p = pred if isinstance(pred, float) else last_number(pred)
        g = last_number(gt)
        if p is None or g is None:
            return False
        if ex["answer_type"] == "integer":
            return abs(p - g) <= 0.5
        prec = int(ex.get("precision") or 1)
        return abs(p - g) <= 0.5 * 10 ** (-prec) + 1e-9
    return str(pred).strip().lower() == str(gt).strip().lower()

def vote_key(pred):
    """Canonical key for plurality voting; None/empty are not votes."""
    if pred is None:
        return None
    s = str(pred).strip()
    if s == "" or s.lower() == "none":
        return None
    n = last_number(s)
    return f"num:{n}" if n is not None else f"txt:{s.lower()}"

# ---------------- load model + data ----------------
t0 = time.time()
print(f"Loading {MODEL_ID} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
).eval()
PAD_ID = processor.tokenizer.pad_token_id
if PAD_ID is None:
    PAD_ID = processor.tokenizer.eos_token_id
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

ds = load_dataset("AI4Math/MathVista", split="testmini")

def build_inputs(image, prompt):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)

@torch.no_grad()
def greedy(image, prompt):
    inputs = build_inputs(image, prompt)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    finished = gen.shape[0] < MAX_NEW
    return processor.decode(gen, skip_special_tokens=True).strip(), finished

@torch.no_grad()
def sample_n(image, prompt, n):
    """Return list of (text, finished) for n temperature samples (sub-batched)."""
    inputs = build_inputs(image, prompt)
    in_len = inputs["input_ids"].shape[1]
    outs = []
    done = 0
    while done < n:
        k = min(GEN_BATCH, n - done)
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=TEMP, top_p=TOP_P, num_return_sequences=k)
        rows = gen[:, in_len:]
        for r in rows:
            real_len = int((r != PAD_ID).sum())
            finished = real_len < MAX_NEW
            outs.append((processor.decode(r, skip_special_tokens=True).strip(), finished))
        done += k
    del inputs
    return outs

# ---------------- run ----------------
os.makedirs(OUT_DIR, exist_ok=True)
done_pids = {os.path.basename(p)[4:-5] for p in glob.glob(os.path.join(OUT_DIR, "pid_*.json"))}
print(f"Resuming: {len(done_pids)} cases already done", flush=True)

t_inf = time.time()
n_proc = 0
for i, ex in enumerate(ds):
    if i >= N_EXAMPLES:
        break
    pid = ex["pid"]
    if pid in done_pids:
        continue
    img = ex["decoded_image"].convert("RGB")

    try:
        g_resp, g_fin = greedy(img, ex["query"])
        g_pred = extract_pred(g_resp, ex)
        g_ok = bool(is_correct(g_pred, ex)) and g_fin

        samples = sample_n(img, ex["query"], N_SAMPLES)
        spreds, scorrect, votes, rep = [], [], Counter(), {}
        n_trunc = 0
        for txt, fin in samples:
            if not fin:
                n_trunc += 1
                continue
            p = extract_pred(txt, ex)
            ok = bool(is_correct(p, ex))
            spreds.append(str(p)); scorrect.append(ok)
            k = vote_key(p)
            if k is not None:
                votes[k] += 1
                rep.setdefault(k, str(p))   # keep an original pred string for this bucket
    except Exception as e:
        torch.cuda.empty_cache()
        print(f"[{i+1:3}] pid={pid:>4} SKIPPED ({type(e).__name__}: {str(e)[:80]})", flush=True)
        continue

    oracle_hit = any(scorrect)                       # best-of-N: at least one sample correct
    vote_key_winner = votes.most_common(1)[0][0] if votes else None
    vote_winner = rep.get(vote_key_winner)           # original pred text, checked as-is
    vote_ok = bool(is_correct(vote_winner, ex)) if vote_winner is not None else False
    # --- collapse measures over the finished samples' answer distribution ---
    n_valid = len(spreds)
    n_distinct = len(votes)
    top_count = votes.most_common(1)[0][1] if votes else 0
    top_share = (top_count / n_valid) if n_valid else 0.0          # 1.0 == fully collapsed

    rec = dict(pid=pid, question=ex["question"], qtype=ex["question_type"],
               answer_type=ex["answer_type"], category=ex["metadata"].get("category"),
               ground_truth=str(ex["answer"]),
               greedy_pred=str(g_pred), greedy_finished=g_fin, greedy_correct=g_ok,
               n_samples=N_SAMPLES, n_truncated=n_trunc, n_valid_samples=n_valid,
               sample_preds=spreds, n_sample_correct=int(sum(scorrect)),
               answer_counts=dict(votes), n_distinct_answers=n_distinct,
               top_answer=vote_winner, top_share=round(top_share, 3),
               oracle_hit=bool(oracle_hit),
               vote_winner=vote_winner, vote_correct=bool(vote_ok))
    with open(os.path.join(OUT_DIR, f"pid_{pid}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    n_proc += 1
    torch.cuda.empty_cache()
    tag = "OK " if g_ok else ("XX " if g_fin else "tr ")
    rec_tag = "RECOVER" if (not g_ok and oracle_hit) else ""
    print(f"[{i+1:3}] pid={pid:>4} greedy={tag} gt={str(ex['answer'])[:14]!r} "
          f"g_pred={str(g_pred)[:14]!r} | oracle={int(oracle_hit)} "
          f"vote={int(vote_ok)} ({sum(scorrect)}/{len(scorrect)} samples ok) {rec_tag}",
          flush=True)

# ---------------- aggregate over everything on disk ----------------
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT_DIR, "pid_*.json")))]
n = len(recs)
greedy_ok   = [r for r in recs if r["greedy_correct"]]
failures    = [r for r in recs if not r["greedy_correct"] and r["greedy_finished"]]
rec_oracle  = [r for r in failures if r["oracle_hit"]]
rec_vote    = [r for r in failures if r["vote_correct"]]

summary = dict(
    model=MODEL_ID, arm="monolithic_best_of_n_no_focus",
    n_examples=n, n_samples=N_SAMPLES, temperature=TEMP, top_p=TOP_P,
    greedy_accuracy=round(len(greedy_ok)/n, 4) if n else None,
    oracle_bestofN_accuracy=round(sum(r["oracle_hit"] for r in recs)/n, 4) if n else None,
    majority_vote_accuracy=round(sum(r["vote_correct"] for r in recs)/n, 4) if n else None,
    n_failures=len(failures),
    failures_recovered_oracle=len(rec_oracle),
    failures_recovered_vote=len(rec_vote),
    recovery_rate_oracle=round(len(rec_oracle)/len(failures), 4) if failures else None,
    recovery_rate_vote=round(len(rec_vote)/len(failures), 4) if failures else None,
    recovered_pids_oracle=[r["pid"] for r in rec_oracle],
    failure_pids=[r["pid"] for r in failures],
    # ---- collapse: how degenerate is the best-of-N answer distribution on failures? ----
    mean_top_share_failures=round(sum(r["top_share"] for r in failures)/len(failures), 3) if failures else None,
    mean_distinct_failures=round(sum(r["n_distinct_answers"] for r in failures)/len(failures), 2) if failures else None,
    n_failures_fully_collapsed=sum(1 for r in failures if r["top_share"] >= 0.9),  # >=90% on one answer
    n_collapsed_and_unrecovered=sum(1 for r in failures if r["top_share"] >= 0.9 and not r["oracle_hit"]),
)
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n==== ARM (a): monolithic best-of-{N_SAMPLES}, no focus modes ====")
print(f"examples: {n} | greedy acc: {summary['greedy_accuracy']} "
      f"| oracle best-of-N acc: {summary['oracle_bestofN_accuracy']} "
      f"| majority-vote acc: {summary['majority_vote_accuracy']}")
print(f"greedy failures: {len(failures)} | recovered by best-of-N (oracle): "
      f"{len(rec_oracle)} ({summary['recovery_rate_oracle']}) "
      f"| recovered by majority vote: {len(rec_vote)} ({summary['recovery_rate_vote']})")
print(f"Done in {time.time()-t_inf:.0f}s. Wrote {OUT_DIR}/summary.json", flush=True)
