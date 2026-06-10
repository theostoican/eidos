import os, re, json, time
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID        = "Qwen/Qwen3.5-9B"
TARGET_FAILURES = int(os.environ.get("TARGET_FAILURES", "10"))
MAX_SAMPLES     = int(os.environ.get("MAX_SAMPLES", "150"))
MAX_NEW         = int(os.environ.get("MAX_NEW", "1536"))
OUT_DIR         = "/root/mathvista_qwen/failures"

# ---------------- load model ----------------
t0 = time.time()
print(f"Loading {MODEL_ID} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
).eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

# ---------------- data ----------------
ds = load_dataset("AI4Math/MathVista", split="testmini")  # 1000 examples w/ answers

# ---------------- answer extraction / scoring ----------------
def last_number(s):
    m = re.findall(r"-?\d+(?:\.\d+)?", str(s).replace(",", ""))
    return float(m[-1]) if m else None

def extract_pred(resp, ex):
    """Find the model's final answer in its free-form reasoning text."""
    tail = resp.strip()
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", tail)
    boxed = boxed[-1].strip() if boxed else None
    window = tail[-200:]  # final answer lives near the end

    if ex["question_type"] == "multi_choice":
        choices = ex["choices"]
        letters = {chr(65 + i): c for i, c in enumerate(choices)}
        # (1) boxed letter e.g. \boxed{C}
        if boxed and boxed.upper() in letters:
            return letters[boxed.upper()]
        # (2) boxed text that equals a choice
        if boxed:
            for c in choices:
                if str(c).strip().lower() == boxed.lower():
                    return c
        # (3) a standalone option letter near the end, e.g. "(C)" / "option C"
        for L in reversed(re.findall(r"\b([A-D])\b", window)):
            if L in letters:
                return letters[L]
        # (4) choice text appearing anywhere (last occurrence wins)
        hit = None
        for c in choices:
            if str(c).strip().lower() in tail.lower():
                hit = c
        return hit if hit is not None else window[-40:]

    # free_form
    if ex["answer_type"] in ("integer", "float"):
        if boxed and last_number(boxed) is not None:
            return last_number(boxed)
        return last_number(window)
    # free-form text
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

# ---------------- inference loop ----------------
os.makedirs(OUT_DIR, exist_ok=True)
records, failures = [], []
n_truncated = 0
t_inf = time.time()
for i, ex in enumerate(ds):
    if len(failures) >= TARGET_FAILURES or i >= MAX_SAMPLES:
        break
    img = ex["decoded_image"].convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text",  "text": ex["query"]},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    finished = gen.shape[0] < MAX_NEW
    resp = processor.decode(gen, skip_special_tokens=True).strip()

    pred = extract_pred(resp, ex)
    ok = bool(is_correct(pred, ex))
    rec = dict(pid=ex["pid"], question=ex["question"], qtype=ex["question_type"],
               answer_type=ex["answer_type"], choices=ex["choices"],
               category=ex["metadata"].get("category"),
               ground_truth=str(ex["answer"]), prediction=str(pred),
               correct=ok, finished=finished, response=resp)
    records.append(rec)

    # Only COMPLETED + wrong answers count as genuine failures; a truncated
    # generation never emitted a final answer, so its 'wrong' prediction is an
    # extraction artifact, not a real model error -> skip it.
    if not finished:
        n_truncated += 1
        status = "trunc(skip)"
    elif ok:
        status = "OK"
    else:
        failures.append(rec)
        status = "XX(fail)"
    print(f"[{i+1:3}] pid={ex['pid']:>4} {status:<11} "
          f"gt={str(ex['answer'])[:18]!r} pred={str(pred)[:18]!r}   "
          f"(failures: {len(failures)})", flush=True)

# ---------------- report + save failures ----------------
n = len(records); ncorrect = sum(r["correct"] for r in records)
acc = ncorrect / n
print(f"\nScanned {n} samples in {time.time()-t_inf:.0f}s")
print(f"finished-accuracy: {acc:.1%} ({ncorrect}/{n}) | truncated(skipped): {n_truncated} "
      f"| clean failures: {len(failures)}")
if len(failures) < TARGET_FAILURES:
    print(f"WARNING: only {len(failures)} clean failures within MAX_SAMPLES={MAX_SAMPLES}; raise the cap.")

for r in failures:
    fdir = os.path.join(OUT_DIR, f"pid_{r['pid']}")
    os.makedirs(fdir, exist_ok=True)
    ex = next(e for e in ds if e["pid"] == r["pid"])
    ex["decoded_image"].convert("RGB").save(os.path.join(fdir, "image.png"))
    with open(os.path.join(fdir, "info.json"), "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(dict(model=MODEL_ID, n_samples=n, finished_accuracy=acc,
                   n_truncated_skipped=n_truncated, n_failures=len(failures),
                   failure_pids=[r["pid"] for r in failures]),
              f, indent=2)
print(f"Saved {len(failures)} failing cases under {OUT_DIR}/")
