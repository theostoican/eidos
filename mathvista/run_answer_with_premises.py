import os, re, json, time, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID    = "Qwen/Qwen3.5-9B"
LOSS_DIR    = "/root/mathvista_qwen/failures"           # original wrong-answer cases
PREMISE_DIR = "/root/mathvista_qwen/visual_premises"    # 8 premises per case (pid_*.json)
OUT_DIR     = "/root/mathvista_qwen/premise_answers"    # where re-answers get written
MAX_NEW     = int(os.environ.get("MAX_NEW", "1024"))    # enough to answer concisely w/o truncation

# ---------------- answer extraction / scoring (same logic as run_eval.py) ----------------
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

# ---------------- final-answer prompt: question + all premises ----------------
def split_query(query):
    """MathVista 'query' = 'Hint: <format>\nQuestion: <q> (Unit: ...)'. Split the two."""
    if "\nQuestion:" in query:
        hint, q = query.split("\nQuestion:", 1)
        return hint.strip(), q.strip()
    return "", query.strip()

def build_answer_prompt(query, premises):
    hint, question = split_query(query)
    lines = []
    for i, (mode, text) in enumerate(premises.items(), 1):
        text = text.replace("Visual premise:", "").strip()
        lines.append(f"{i}. {text}")
    premise_block = "\n".join(lines)
    parts = [f"Question:\n{question}", "", f"Visual premises:\n{premise_block}", ""]
    if hint:
        parts.append(hint)  # keep the dataset's answer-format hint so the answer is parseable
    parts.append("Now answer the question using the supported visual premise.")
    parts.append("Provide the final answer clearly.")
    return "\n".join(parts)

# ---------------- load model + data ----------------
t0 = time.time()
print(f"Loading {MODEL_ID} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
).eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

ds = load_dataset("AI4Math/MathVista", split="testmini")
by_pid = {e["pid"]: e for e in ds}

@torch.no_grad()
def generate(image, prompt):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    finished = gen.shape[0] < MAX_NEW
    return processor.decode(gen, skip_special_tokens=True).strip(), finished

# ---------------- run premise-augmented answering over the loss cases ----------------
os.makedirs(OUT_DIR, exist_ok=True)
prem_files = sorted(glob.glob(os.path.join(PREMISE_DIR, "pid_*.json")),
                    key=lambda p: int(os.path.basename(p)[4:-5]))
print(f"Found {len(prem_files)} premise files", flush=True)

records = []
n_fixed = n_truncated = 0
t_inf = time.time()
for ci, pf in enumerate(prem_files):
    pr = json.load(open(pf))
    pid = pr["pid"]
    ex = by_pid[pid]
    info = json.load(open(os.path.join(LOSS_DIR, f"pid_{pid}", "info.json")))
    img = ex["decoded_image"].convert("RGB")

    prompt = build_answer_prompt(ex["query"], pr["premises"])
    resp, finished = generate(img, prompt)
    pred = extract_pred(resp, ex)
    ok = bool(is_correct(pred, ex))

    orig_pred = info.get("prediction")            # the earlier (no-premise) wrong answer
    fixed = ok                                    # every case here was originally wrong
    n_fixed += int(fixed)
    n_truncated += int(not finished)

    rec = dict(pid=pid, question=pr["question"], qtype=ex["question_type"],
               answer_type=ex["answer_type"], choices=ex["choices"],
               ground_truth=str(ex["answer"]), original_prediction=str(orig_pred),
               premise_prediction=str(pred), now_correct=ok, fixed=fixed,
               finished=finished, prompt=prompt, response=resp)
    records.append(rec)
    with open(os.path.join(OUT_DIR, f"pid_{pid}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    print(f"[{ci+1:2}/{len(prem_files)}] pid={pid:>3} "
          f"{'FIXED' if fixed else 'still wrong':<11} "
          f"gt={str(ex['answer'])[:16]!r} orig={str(orig_pred)[:12]!r} -> new={str(pred)[:16]!r}"
          f"{'' if finished else ' (truncated)'}", flush=True)

# ---------------- summary + markdown report ----------------
n = len(records)
print(f"\nDone in {time.time()-t_inf:.0f}s | premise-augmented answering on {n} loss cases")
print(f"now-correct (fixed): {n_fixed}/{n} | truncated: {n_truncated}")

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(dict(model=MODEL_ID, n_cases=n, n_fixed=n_fixed, n_truncated=n_truncated,
                   fixed_pids=[r["pid"] for r in records if r["fixed"]],
                   still_wrong_pids=[r["pid"] for r in records if not r["fixed"]]),
              f, indent=2)

md = ["# Premise-augmented re-answering of MathVista loss cases", "",
      f"Model: `{MODEL_ID}` | {n} cases | fixed: **{n_fixed}/{n}**",
      "",
      "Each originally-wrong case is re-answered with its 8 visual premises fed back in.", ""]
for r in records:
    tag = "✅ FIXED" if r["fixed"] else "❌ still wrong"
    md.append(f"## pid {r['pid']} — {tag}")
    md.append(f"**Q:** {r['question']}")
    md.append(f"- ground truth: `{r['ground_truth']}`")
    md.append(f"- original (no-premise) answer: `{r['original_prediction']}`")
    md.append(f"- premise-augmented answer: `{r['premise_prediction']}`"
              + ("" if r["finished"] else " *(truncated)*"))
    md.append("")
with open(os.path.join(OUT_DIR, "premise_answers_report.md"), "w") as f:
    f.write("\n".join(md))
print(f"Wrote {n} JSON files + summary.json + premise_answers_report.md to {OUT_DIR}/")
