import os, re, json, time, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText

# Two-stage inference with focus MODES, run on the SAME greedy failures the
# best-of-N monolithic control (run_monolithic_best_of_n.py) surfaces.
#
# This reproduces the RQ2 pipeline (premise-generation -> answer) exactly:
#   Stage 1: for each of 20 focus modes, GENERATE a visual premise (greedy).
#   Stage 2: RE-ANSWER the question conditioned on that premise (greedy).
# Diversity here comes from the 20 different mode PROMPTS (decoding is greedy),
# vs the control whose 20 answers come from temperature SAMPLING of one prompt.
# Same model, scoring, and token budget -> the only difference is the source of
# diversity, which is exactly the "does an explicit premise break the collapse?"
# question.

MODEL_ID    = "Qwen/Qwen3.5-9B"
CTRL_DIR    = "monolithic_best_of_n"            # control outputs (for failure pids)
OUT_DIR     = "two_stage_on_failures"
MAX_PREM    = int(os.environ.get("MAX_PREM", "320"))    # premise length (matches RQ2)
MAX_NEW     = int(os.environ.get("MAX_NEW", "1024"))    # answer length (matches control/RQ1)

BASE = (
    "Given the image and the question, identify one possible visual premise "
    "that could be useful for solving the question.\n"
    "Do not answer the question yet.\n"
    "Focus only on visual evidence.\n"
    "State exactly one concise visual premise (1-2 sentences). Begin with 'Visual premise:'."
)
MODES = {
    "1_salient":     "Focus on the most salient, immediately obvious visual evidence.",
    "2_subtle":      "Focus on small, fine-grained, or easily missed details.",
    "3_spatial":     "Focus on spatial relations: relative position, alignment, order, direction, overlap, or connections between elements.",
    "4_ocr":         "Focus only on text and symbols: OCR-readable labels, axis ticks, legends, units, symbols, and numbers.",
    "5_distractor":  "Focus on a possible distractor or misleading cue that could tempt a wrong answer.",
    "6_alternative": "Offer an alternative interpretation of the image that differs from the most obvious reading.",
    "7_uncertainty": "Identify a source of visual uncertainty or ambiguity (occlusion, low resolution, overlapping marks, unclear arrow direction, etc.) that could affect the answer.",
    "8_people":      "If there are people in the image, focus on who they could be (likely identity, role, age, or relationship based on visual cues such as clothing, setting, and appearance). If there are no people, say so.",
    "9_digits":      "Focus on digits/numbers visible in the image and their relationship to the surrounding objects: do the digits count or quantify something in the image (e.g. labels, counts, tallies, axis values), and what do they refer to? Perhaps axes on a chart? If so, count them.",
    "10_point_on_curve":      "When a point is marked on a plotted curve or line, read its y-value by counting tick marks up from the x-axis and report that number.",
    "11_value_curve_reaches": "Read the y-value the plotted curve reaches at the x-position the question is about, counting ticks up from the x-axis, and state the number.",
    "12_marker_on_vs_off":    "Distinguish markers that lie ON the plotted curve from markers drawn OFF the curve, and report the y-coordinate of the one that lies on the curve.",
    "13_trace_to_x":          "Trace along the plotted curve to the x-value the question asks about and read the y-coordinate there off the y-axis ticks.",
    "14_read_y_by_ticks":     "Count the y-axis tick marks one at a time from the x-axis (0) up to the exact height of the relevant marker on the chart. Do not stop early or undercount. Report that tick count as its y-value.",
    "15_curve_height":        "Report the height (y-value) of the curve at the relevant location as a number read off the axis, not as a qualitative description.",
    "16_two_points_pick_curve": "If two marked points share the same x-position, identify which one the curve actually passes through (or bends toward) and report that point's y-value.",
    "17_settle_value":        "If the curve flattens, levels off, or settles near the target x, read the y-value of that level off the axis and report the number.",
    "18_coordinate_on_curve": "Give the (x, y) of the point where the curve meets the question's x-value, reading the y by counting ticks up from the x-axis.",
    "19_approach_height":     "Read the y-height the curve approaches at the target x by counting ticks up from the x-axis, and report that number.",
    "20_hole_on_curve":       "Count the y-axis tick marks one at a time starting from the x-axis (0), going up to the height of the open circle or marked point that lies on the curve. Do not stop early. Report that exact tick count as its y-value.",
}

# ---------------- scoring (identical to the control) ----------------
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
    if pred is None:
        return None
    s = str(pred).strip()
    if s == "" or s.lower() == "none":
        return None
    nu = last_number(s)
    return f"num:{nu}" if nu is not None else f"txt:{s.lower()}"

def split_query(query):
    if "\nQuestion:" in query:
        hint, q = query.split("\nQuestion:", 1)
        return hint.strip(), q.strip()
    return "", query.strip()

def build_premise_prompt(question, mode_instruction):
    return f"{BASE}\nMode: {mode_instruction}\n\nQuestion: {question}"

def build_answer_prompt(query, premise_text):
    hint, question = split_query(query)
    premise_text = premise_text.replace("Visual premise:", "").strip()
    parts = [f"Question:\n{question}", "", f"Visual premise:\n{premise_text}", ""]
    if hint:
        parts.append(hint)
    parts.append("Now answer the question using the supported visual premise.")
    parts.append("Provide the final answer clearly.")
    return "\n".join(parts)

# ---------------- load ----------------
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
def generate(image, prompt, max_new):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    gen = out[0][inputs["input_ids"].shape[1]:]
    finished = gen.shape[0] < max_new
    txt = processor.decode(gen, skip_special_tokens=True).strip()
    del inputs
    return txt, finished

# ---------------- which failures? ----------------
env_pids = os.environ.get("FAILURE_PIDS", "").strip()
if env_pids:
    failure_pids = [p for p in env_pids.split(",") if p]
else:
    summ = json.load(open(os.path.join(CTRL_DIR, "summary.json")))
    failure_pids = summ["failure_pids"]
print(f"Two-stage on {len(failure_pids)} control failures: {failure_pids}", flush=True)

os.makedirs(OUT_DIR, exist_ok=True)
done_pids = {os.path.basename(p)[4:-5] for p in glob.glob(os.path.join(OUT_DIR, "pid_*.json"))}

t_inf = time.time()
for ci, pid in enumerate(failure_pids):
    if pid in done_pids:
        continue
    ex = by_pid[pid]
    img = ex["decoded_image"].convert("RGB")

    per_mode = {}
    correct_modes = []
    votes, rep = Counter(), {}
    for mode, instr in MODES.items():
        try:
            premise, _ = generate(img, build_premise_prompt(ex["question"], instr), MAX_PREM)
            ans, fin = generate(img, build_answer_prompt(ex["query"], premise), MAX_NEW)
        except Exception as e:
            torch.cuda.empty_cache()
            per_mode[mode] = dict(premise=None, prediction=None, correct=False,
                                  finished=False, error=f"{type(e).__name__}: {str(e)[:80]}")
            continue
        pred = extract_pred(ans, ex)
        ok = bool(is_correct(pred, ex)) and fin
        per_mode[mode] = dict(premise=premise, prediction=str(pred), correct=ok,
                              finished=fin, answer=ans)
        if ok:
            correct_modes.append(mode)
        k = vote_key(pred)
        if k is not None and fin:
            votes[k] += 1
            rep.setdefault(k, str(pred))

    fixed = len(correct_modes) > 0
    vk = votes.most_common(1)[0][0] if votes else None
    vote_winner = rep.get(vk)
    vote_ok = bool(is_correct(vote_winner, ex)) if vote_winner is not None else False
    n_valid = sum(votes.values())
    top_share = (votes.most_common(1)[0][1] / n_valid) if n_valid else 0.0

    rec = dict(pid=pid, question=ex["question"], qtype=ex["question_type"],
               answer_type=ex["answer_type"], ground_truth=str(ex["answer"]),
               fixed_by_any=fixed, correct_modes=correct_modes,
               n_distinct_answers=len(votes), top_answer=vote_winner,
               top_share=round(top_share, 3), n_valid=n_valid,
               vote_winner=vote_winner, vote_correct=bool(vote_ok),
               per_mode=per_mode)
    with open(os.path.join(OUT_DIR, f"pid_{pid}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)
    torch.cuda.empty_cache()
    print(f"[{ci+1:2}/{len(failure_pids)}] pid={pid:>4} gt={str(ex['answer'])[:14]!r} "
          f"| fixed={int(fixed)} ({len(correct_modes)} modes) vote={int(vote_ok)} "
          f"distinct={len(votes)} top_share={top_share:.2f} {correct_modes[:4]}", flush=True)

# ---------------- summary ----------------
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(OUT_DIR, "pid_*.json")))]
n = len(recs)
fixed = [r for r in recs if r["fixed_by_any"]]
voted = [r for r in recs if r["vote_correct"]]
summary = dict(model=MODEL_ID, arm="two_stage_modes_on_control_failures",
               n_failures=n, n_modes=len(MODES),
               failures_fixed_oracle=len(fixed),
               failures_fixed_vote=len(voted),
               recovery_rate_oracle=round(len(fixed)/n, 4) if n else None,
               recovery_rate_vote=round(len(voted)/n, 4) if n else None,
               mean_distinct=round(sum(r["n_distinct_answers"] for r in recs)/n, 2) if n else None,
               mean_top_share=round(sum(r["top_share"] for r in recs)/n, 3) if n else None,
               fixed_pids=[r["pid"] for r in fixed])
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n==== TWO-STAGE (modes) on {n} control failures ====")
print(f"fixed by >=1 mode (oracle): {len(fixed)}/{n} ({summary['recovery_rate_oracle']}) "
      f"| fixed by mode-vote: {len(voted)}/{n}")
print(f"mean distinct answers/case: {summary['mean_distinct']} | mean top_share: {summary['mean_top_share']}")
print(f"Done in {time.time()-t_inf:.0f}s -> {OUT_DIR}/summary.json", flush=True)
