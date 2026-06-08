import os, re, json, time, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID    = "Qwen/Qwen3.5-9B"
LOSS_DIR    = "/root/mathvista_qwen/failures"            # original wrong-answer cases
PREMISE_DIR = "/root/mathvista_qwen/visual_premises"     # 8 premises per case (pid_*.json)
OUT_DIR     = "/root/mathvista_qwen/premise_answers_per_premise"
MAX_NEW     = int(os.environ.get("MAX_NEW", "1024"))

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

# ---------------- final-answer prompt: question + ONE premise ----------------
def split_query(query):
    """MathVista 'query' = 'Hint: <format>\nQuestion: <q> (Unit: ...)'. Split the two."""
    if "\nQuestion:" in query:
        hint, q = query.split("\nQuestion:", 1)
        return hint.strip(), q.strip()
    return "", query.strip()

def build_answer_prompt(query, premise_text):
    hint, question = split_query(query)
    premise_text = premise_text.replace("Visual premise:", "").strip()
    parts = [f"Question:\n{question}", "", f"Visual premise:\n{premise_text}", ""]
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

# ---------------- run: one prompt PER premise, per loss case ----------------
os.makedirs(OUT_DIR, exist_ok=True)
prem_files = sorted(glob.glob(os.path.join(PREMISE_DIR, "pid_*.json")),
                    key=lambda p: int(os.path.basename(p)[4:-5]))
print(f"Found {len(prem_files)} premise files", flush=True)

records = []
mode_fix_counts = {}                  # per-mode: how many cases it answers correctly
n_cases_fixed_by_any = 0
t_inf = time.time()
for ci, pf in enumerate(prem_files):
    pr = json.load(open(pf))
    pid = pr["pid"]
    ex = by_pid[pid]
    info = json.load(open(os.path.join(LOSS_DIR, f"pid_{pid}", "info.json")))
    img = ex["decoded_image"].convert("RGB")
    orig_pred = info.get("prediction")

    per_premise = {}
    correct_modes = []
    for mode, premise_text in pr["premises"].items():
        prompt = build_answer_prompt(ex["query"], premise_text)
        resp, finished = generate(img, prompt)
        pred = extract_pred(resp, ex)
        ok = bool(is_correct(pred, ex))
        per_premise[mode] = dict(premise=premise_text, prediction=str(pred),
                                 correct=ok, finished=finished, prompt=prompt, response=resp)
        if ok:
            correct_modes.append(mode)
            mode_fix_counts[mode] = mode_fix_counts.get(mode, 0) + 1

    case_fixed = len(correct_modes) > 0
    n_cases_fixed_by_any += int(case_fixed)
    rec = dict(pid=pid, question=pr["question"], qtype=ex["question_type"],
               answer_type=ex["answer_type"], choices=ex["choices"],
               ground_truth=str(ex["answer"]), original_prediction=str(orig_pred),
               correct_modes=correct_modes, fixed_by_any=case_fixed,
               per_premise=per_premise)
    records.append(rec)
    with open(os.path.join(OUT_DIR, f"pid_{pid}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    preds = {m: per_premise[m]["prediction"] for m in per_premise}
    print(f"[{ci+1:2}/{len(prem_files)}] pid={pid:>3} gt={str(ex['answer'])[:14]!r} "
          f"orig={str(orig_pred)[:10]!r} | fixed_by={correct_modes or '-'}", flush=True)
    for m in per_premise:
        mark = "OK" if per_premise[m]["correct"] else "  "
        print(f"        {mark} {m:<14} -> {per_premise[m]['prediction'][:40]!r}", flush=True)

# ---------------- summary + markdown report ----------------
n = len(records)
print(f"\nDone in {time.time()-t_inf:.0f}s | {n} cases x {len(records[0]['per_premise'])} premises "
      f"= {n*len(records[0]['per_premise'])} prompts")
print(f"cases fixed by >=1 premise: {n_cases_fixed_by_any}/{n}")
print("per-mode correct counts:", {k: mode_fix_counts.get(k, 0) for k in records[0]['per_premise']})

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(dict(model=MODEL_ID, n_cases=n,
                   n_premises_per_case=len(records[0]['per_premise']),
                   n_cases_fixed_by_any=n_cases_fixed_by_any,
                   per_mode_correct_counts={k: mode_fix_counts.get(k, 0)
                                            for k in records[0]['per_premise']},
                   fixed_pids=[r["pid"] for r in records if r["fixed_by_any"]]),
              f, indent=2)

md = ["# Per-premise re-answering of MathVista loss cases", "",
      f"Model: `{MODEL_ID}` | {n} cases | one prompt **per premise** "
      f"({len(records[0]['per_premise'])} per case)",
      f"Cases fixed by at least one premise: **{n_cases_fixed_by_any}/{n}**", "",
      "Per-mode correct counts: "
      + ", ".join(f"`{k}`={mode_fix_counts.get(k,0)}" for k in records[0]['per_premise']), ""]
for r in records:
    md.append(f"## pid {r['pid']}  (GT: `{r['ground_truth']}` | orig wrong: `{r['original_prediction']}`)")
    md.append(f"**Q:** {r['question']}")
    md.append(f"fixed by: {r['correct_modes'] or 'none'}\n")
    md.append("| mode | answer | correct |")
    md.append("|------|--------|---------|")
    for m, d in r["per_premise"].items():
        md.append(f"| {m} | `{d['prediction'][:40]}` | {'✅' if d['correct'] else ''} |")
    md.append("")
with open(os.path.join(OUT_DIR, "per_premise_report.md"), "w") as f:
    f.write("\n".join(md))
print(f"Wrote {n} JSON files + summary.json + per_premise_report.md to {OUT_DIR}/")
