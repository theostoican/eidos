"""Incrementally add a 9th premise mode ('9_digits') to the existing runs:
   1. generate the new premise for each loss case  -> append to visual_premises/pid_*.json
   2. re-answer each case with ONLY that new premise -> append to premise_answers_per_premise/pid_*.json
   3. rebuild summary.json + per_premise_report.md (and the premises_report.md)
The existing 8 modes are left untouched (not regenerated)."""
import os, re, json, time, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID    = "Qwen/Qwen3.5-9B"
LOSS_DIR    = "/root/mathvista_qwen/failures"
PREMISE_DIR = "/root/mathvista_qwen/visual_premises"
ANS_DIR     = "/root/mathvista_qwen/premise_answers_per_premise"
MAX_NEW_PREM = 320
MAX_NEW_ANS  = 1024

NEW_KEY  = "9_digits"
NEW_INSTR = ("Focus on digits/numbers visible in the image and their relationship to the "
             "surrounding objects: do the digits count or quantify something in the image "
             "(e.g. labels, counts, tallies, axis values), and what do they refer to? "
             "Perhaps axes on a chart? If so, count them.")

BASE = (
    "Given the image and the question, identify one possible visual premise "
    "that could be useful for solving the question.\n"
    "Do not answer the question yet.\n"
    "Focus only on visual evidence.\n"
    "State exactly one concise visual premise (1-2 sentences). Begin with 'Visual premise:'."
)
def build_premise_prompt(question):
    return f"{BASE}\nMode: {NEW_INSTR}\n\nQuestion: {question}"

# ---------------- answer extraction / scoring (identical to run_answer_per_premise.py) ----------------
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

def split_query(query):
    if "\nQuestion:" in query:
        hint, q = query.split("\nQuestion:", 1)
        return hint.strip(), q.strip()
    return "", query.strip()

def build_answer_prompt(query, premise_text):
    hint, question = split_query(query)
    premise_text = premise_text.replace("Visual premise:", "").strip()
    parts = [f"Question:\n{question}", "", f"Visual premise:\n{premise_text}", ""]
    if hint:
        parts.append(hint)
    parts += ["Now answer the question using the supported visual premise.",
              "Provide the final answer clearly."]
    return "\n".join(parts)

# ---------------- load model + data ----------------
t0 = time.time()
print(f"Loading {MODEL_ID} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

ds = load_dataset("AI4Math/MathVista", split="testmini")
by_pid = {e["pid"]: e for e in ds}

@torch.no_grad()
def generate(image, prompt, max_new):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    finished = gen.shape[0] < max_new
    return processor.decode(gen, skip_special_tokens=True).strip(), finished

# ---------------- run incremental mode over each case ----------------
prem_files = sorted(glob.glob(os.path.join(PREMISE_DIR, "pid_*.json")),
                    key=lambda p: int(os.path.basename(p)[4:-5]))
print(f"Found {len(prem_files)} cases; adding mode '{NEW_KEY}'", flush=True)

t_inf = time.time()
for ci, pf in enumerate(prem_files):
    pr  = json.load(open(pf))
    pid = pr["pid"]
    ex  = by_pid[pid]
    info = json.load(open(os.path.join(LOSS_DIR, f"pid_{pid}", "info.json")))

    # 1) generate premise (use the same failures image + info question as run_premises.py)
    img = Image.open(os.path.join(LOSS_DIR, f"pid_{pid}", "image.png")).convert("RGB")
    premise, _ = generate(img, build_premise_prompt(info["question"]), MAX_NEW_PREM)
    pr["premises"][NEW_KEY] = premise
    with open(pf, "w") as f:
        json.dump(pr, f, indent=2, ensure_ascii=False)

    # 2) answer with ONLY this new premise (use dataset image, same as answer script)
    img2 = ex["decoded_image"].convert("RGB")
    prompt = build_answer_prompt(ex["query"], premise)
    resp, finished = generate(img2, prompt, MAX_NEW_ANS)
    pred = extract_pred(resp, ex)
    ok = bool(is_correct(pred, ex))

    ans = json.load(open(os.path.join(ANS_DIR, f"pid_{pid}.json")))
    ans["per_premise"][NEW_KEY] = dict(premise=premise, prediction=str(pred),
                                       correct=ok, finished=finished, prompt=prompt, response=resp)
    if NEW_KEY in ans["correct_modes"]:        # idempotent on re-run: drop stale verdict first
        ans["correct_modes"].remove(NEW_KEY)
    if ok:
        ans["correct_modes"].append(NEW_KEY)
    ans["fixed_by_any"] = len(ans["correct_modes"]) > 0
    with open(os.path.join(ANS_DIR, f"pid_{pid}.json"), "w") as f:
        json.dump(ans, f, indent=2, ensure_ascii=False)

    print(f"[{ci+1:2}/{len(prem_files)}] pid={pid:>3} gt={str(ex['answer'])[:14]!r} "
          f"{NEW_KEY} -> {str(pred)[:30]!r} {'OK' if ok else ''}", flush=True)
    print(f"        premise: {premise.replace(chr(10),' ')[:110]}", flush=True)

print(f"\nGeneration done in {time.time()-t_inf:.0f}s", flush=True)

# ---------------- rebuild summary.json + per_premise_report.md from updated files ----------------
recs = [json.load(open(p)) for p in
        sorted(glob.glob(os.path.join(ANS_DIR, "pid_*.json")),
               key=lambda p: int(os.path.basename(p)[4:-5]))]
modes = list(recs[0]["per_premise"].keys())
mode_counts = {m: sum(r["per_premise"][m]["correct"] for r in recs) for m in modes}
n_fixed = sum(r["fixed_by_any"] for r in recs)

with open(os.path.join(ANS_DIR, "summary.json"), "w") as f:
    json.dump(dict(model=MODEL_ID, n_cases=len(recs), n_premises_per_case=len(modes),
                   n_cases_fixed_by_any=n_fixed, per_mode_correct_counts=mode_counts,
                   fixed_pids=[r["pid"] for r in recs if r["fixed_by_any"]]), f, indent=2)

md = ["# Per-premise re-answering of MathVista loss cases", "",
      f"Model: `{MODEL_ID}` | {len(recs)} cases | one prompt **per premise** ({len(modes)} per case)",
      f"Cases fixed by at least one premise: **{n_fixed}/{len(recs)}**", "",
      "Per-mode correct counts: " + ", ".join(f"`{m}`={mode_counts[m]}" for m in modes), ""]
for r in recs:
    md.append(f"## pid {r['pid']}  (GT: `{r['ground_truth']}` | orig wrong: `{r['original_prediction']}`)")
    md.append(f"**Q:** {r['question']}")
    md.append(f"fixed by: {r['correct_modes'] or 'none'}\n")
    md.append("| mode | answer | correct |")
    md.append("|------|--------|---------|")
    for m in modes:
        d = r["per_premise"][m]
        md.append(f"| {m} | `{d['prediction'][:40]}` | {'✅' if d['correct'] else ''} |")
    md.append("")
with open(os.path.join(ANS_DIR, "per_premise_report.md"), "w") as f:
    f.write("\n".join(md))

# rebuild premises_report.md too (so the new premise text shows up there)
precs = [json.load(open(p)) for p in prem_files]
pmd = ["# Visual premises for MathVista loss cases", "",
       f"Model: `{MODEL_ID}` | {len(precs)} cases x {len(precs[0]['premises'])} modes", ""]
for r in precs:
    pmd.append(f"## pid {r['pid']}  (GT: {r['ground_truth']} | wrong pred: {r['wrong_prediction']})")
    pmd.append(f"**Q:** {r['question']}")
    pmd.append("")
    for mk in r["premises"]:
        pmd.append(f"- **{mk}** — {r['premises'][mk]}")
    pmd.append("")
with open(os.path.join(PREMISE_DIR, "premises_report.md"), "w") as f:
    f.write("\n".join(pmd))

print(f"fixed_by_any: {n_fixed}/{len(recs)} | {NEW_KEY} correct count: {mode_counts[NEW_KEY]}")
print("Updated summary.json, per_premise_report.md, premises_report.md", flush=True)
