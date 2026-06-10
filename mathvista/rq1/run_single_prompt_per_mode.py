import os, re, json, time, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

# RQ1 — single prompt per mode (no separate premise-generation step).
#
# RQ2 is two-stage: for each (image, mode) it first GENERATES a visual premise
# (run_premises.py) and then RE-ANSWERS the question conditioned on that premise
# (run_answer_per_premise.py) — two model calls per (image, mode).
#
# RQ1 collapses that into ONE call: a single prompt per mode that both directs
# the model's attention (the mode's focus) AND asks for the final answer. We run
# it once per mode for each of the same 8 failure images and report performance.

MODEL_ID  = "Qwen/Qwen3.5-9B"
LOSS_DIR  = "../rq2/failures"               # the same 8 wrong-answer cases used by RQ2
OUT_DIR   = "answers_per_mode"
MAX_NEW   = int(os.environ.get("MAX_NEW", "1024"))

# ---- same 21 focus modes as RQ2 (run_premises.py), so the two RQs are comparable ----
MODES = {
    "1_salient":     "Focus on the most salient, immediately obvious visual evidence.",
    "2_subtle":      "Focus on small, fine-grained, or easily missed details.",
    "3_spatial":     "Focus on spatial relations: relative position, alignment, order, direction, overlap, or connections between elements.",
    "4_ocr":         "Focus only on text and symbols: OCR-readable labels, axis ticks, legends, units, symbols, and numbers.",
    "5_distractor":  "Focus on a possible distractor or misleading cue that could tempt a wrong answer.",
    "6_alternative": "Consider an alternative interpretation of the image that differs from the most obvious reading.",
    "7_uncertainty": "Account for a source of visual uncertainty or ambiguity (occlusion, low resolution, overlapping marks, unclear arrow direction, etc.) that could affect the answer.",
    "8_people":      "If there are people in the image, consider who they could be (likely identity, role, age, or relationship based on visual cues such as clothing, setting, and appearance).",
    "9_digits":      "Focus on digits/numbers visible in the image and their relationship to the surrounding objects: do the digits count or quantify something (labels, counts, tallies, axis values), and what do they refer to? Perhaps axes on a chart? If so, count them.",
    # ---- graph-reading modes (target curve/axis misreads, e.g. limit-from-graph questions) ----
    "10_point_on_curve":      "When a point is marked on a plotted curve or line, read its y-value by counting tick marks up from the x-axis and use that number.",
    "11_value_curve_reaches": "Read the y-value the plotted curve reaches at the x-position the question is about, counting ticks up from the x-axis.",
    "12_marker_on_vs_off":    "Distinguish markers that lie ON the plotted curve from markers drawn OFF the curve, and use the y-coordinate of the one that lies on the curve.",
    "13_trace_to_x":          "Trace along the plotted curve to the x-value the question asks about and read the y-coordinate there off the y-axis ticks.",
    "14_read_y_by_ticks":     "Count the y-axis tick marks one at a time from the x-axis (0) up to the exact height of the relevant marker on the chart. Do not stop early or undercount. Use that tick count as its y-value.",
    "15_curve_height":        "Treat the height (y-value) of the curve at the relevant location as a number read off the axis, not a qualitative description.",
    "16_two_points_pick_curve": "If two marked points share the same x-position, identify which one the curve actually passes through (or bends toward) and use that point's y-value.",
    "17_settle_value":        "If the curve flattens, levels off, or settles near the target x, read the y-value of that level off the axis.",
    "18_coordinate_on_curve": "Use the (x, y) of the point where the curve meets the question's x-value, reading the y by counting ticks up from the x-axis.",
    "19_approach_height":     "Read the y-height the curve approaches at the target x by counting ticks up from the x-axis.",
    "20_hole_on_curve":       "Count the y-axis tick marks one at a time starting from the x-axis (0), going up to the height of the open circle or marked point that lies on the curve. Do not stop early. Use that exact tick count as its y-value.",
}

# ---------------- answer extraction / scoring (same logic as RQ2) ----------------
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

# ---------------- single prompt: mode-focus + question, one call ----------------
def split_query(query):
    """MathVista 'query' = 'Hint: <format>\nQuestion: <q> (Unit: ...)'. Split the two."""
    if "\nQuestion:" in query:
        hint, q = query.split("\nQuestion:", 1)
        return hint.strip(), q.strip()
    return "", query.strip()

def build_prompt(query, mode_instruction):
    hint, question = split_query(query)
    parts = [
        "Look at the image carefully and answer the question.",
        f"Focus: {mode_instruction}",
        "",
        f"Question:\n{question}",
    ]
    if hint:
        parts += ["", hint]  # keep the dataset's answer-format hint so the answer is parseable
    parts += ["", "Provide the final answer clearly."]
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

# ---------------- run: ONE prompt per mode, per loss case ----------------
os.makedirs(OUT_DIR, exist_ok=True)
cases = sorted(glob.glob(os.path.join(LOSS_DIR, "pid_*")),
               key=lambda p: int(os.path.basename(p).split("_")[1]))
print(f"Found {len(cases)} loss cases", flush=True)

records = []
mode_fix_counts = {}                  # per-mode: how many cases it answers correctly
n_cases_fixed_by_any = 0
t_inf = time.time()
for ci, cdir in enumerate(cases):
    info = json.load(open(os.path.join(cdir, "info.json")))
    pid  = info["pid"]
    ex   = by_pid[pid]
    img  = ex["decoded_image"].convert("RGB")
    orig_pred = info.get("prediction")

    per_mode = {}
    correct_modes = []
    for mode, instr in MODES.items():
        prompt = build_prompt(ex["query"], instr)
        resp, finished = generate(img, prompt)
        pred = extract_pred(resp, ex)
        ok = bool(is_correct(pred, ex))
        per_mode[mode] = dict(prediction=str(pred), correct=ok, finished=finished,
                              prompt=prompt, response=resp)
        if ok:
            correct_modes.append(mode)
            mode_fix_counts[mode] = mode_fix_counts.get(mode, 0) + 1

    case_fixed = len(correct_modes) > 0
    n_cases_fixed_by_any += int(case_fixed)
    rec = dict(pid=pid, question=info["question"], qtype=ex["question_type"],
               answer_type=ex["answer_type"], choices=ex["choices"],
               ground_truth=str(ex["answer"]), original_prediction=str(orig_pred),
               correct_modes=correct_modes, fixed_by_any=case_fixed,
               per_mode=per_mode)
    records.append(rec)
    with open(os.path.join(OUT_DIR, f"pid_{pid}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

    print(f"[{ci+1:2}/{len(cases)}] pid={pid:>3} gt={str(ex['answer'])[:14]!r} "
          f"orig={str(orig_pred)[:10]!r} | fixed_by={correct_modes or '-'}", flush=True)
    for m in per_mode:
        mark = "OK" if per_mode[m]["correct"] else "  "
        print(f"        {mark} {m:<14} -> {per_mode[m]['prediction'][:40]!r}", flush=True)

# ---------------- summary + markdown report ----------------
n = len(records)
n_modes = len(MODES)
print(f"\nDone in {time.time()-t_inf:.0f}s | {n} cases x {n_modes} modes = {n*n_modes} prompts")
print(f"cases fixed by >=1 mode: {n_cases_fixed_by_any}/{n}")
print("per-mode correct counts:", {k: mode_fix_counts.get(k, 0) for k in MODES})

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(dict(model=MODEL_ID, approach="single_prompt_per_mode",
                   n_cases=n, n_modes=n_modes,
                   n_cases_fixed_by_any=n_cases_fixed_by_any,
                   per_mode_correct_counts={k: mode_fix_counts.get(k, 0) for k in MODES},
                   fixed_pids=[r["pid"] for r in records if r["fixed_by_any"]]),
              f, indent=2)

md = ["# RQ1 — single prompt per mode (MathVista loss cases)", "",
      f"Model: `{MODEL_ID}` | {n} cases | one prompt **per mode** ({n_modes} per case), "
      f"one model call each — no separate premise step",
      f"Cases fixed by at least one mode: **{n_cases_fixed_by_any}/{n}**", "",
      "Per-mode correct counts: "
      + ", ".join(f"`{k}`={mode_fix_counts.get(k,0)}" for k in MODES), ""]
for r in records:
    md.append(f"## pid {r['pid']}  (GT: `{r['ground_truth']}` | orig wrong: `{r['original_prediction']}`)")
    md.append(f"**Q:** {r['question']}")
    md.append(f"fixed by: {r['correct_modes'] or 'none'}\n")
    md.append("| mode | answer | correct |")
    md.append("|------|--------|---------|")
    for m, d in r["per_mode"].items():
        md.append(f"| {m} | `{d['prediction'][:40]}` | {'✅' if d['correct'] else ''} |")
    md.append("")
with open(os.path.join(OUT_DIR, "single_prompt_report.md"), "w") as f:
    f.write("\n".join(md))
print(f"Wrote {n} JSON files + summary.json + single_prompt_report.md to {OUT_DIR}/")
