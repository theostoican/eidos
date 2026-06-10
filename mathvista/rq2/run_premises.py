import os, json, time, glob
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

MODEL_ID  = "Qwen/Qwen3.5-9B"
LOSS_DIR  = "failures"          # the wrong-answer cases mined earlier (repo-relative)
OUT_DIR   = "visual_premises"
MAX_NEW   = int(os.environ.get("MAX_NEW", "320"))

# ---- base instruction (given) + 8 focus modes that diversify the premise ----
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
    # ---- graph-reading modes (target curve/axis misreads, e.g. limit-from-graph questions) ----
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
    # curve/hole-specific tick counter (recovers limit-from-graph misreads, e.g. pid_37)
    "20_hole_on_curve":       "Count the y-axis tick marks one at a time starting from the x-axis (0), going up to the height of the open circle or marked point that lies on the curve. Do not stop early. Report that exact tick count as its y-value.",
}

def build_prompt(question, mode_instruction):
    return f"{BASE}\nMode: {mode_instruction}\n\nQuestion: {question}"

# ---------------- load model ----------------
t0 = time.time()
print(f"Loading {MODEL_ID} ...", flush=True)
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
).eval()
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

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
    return processor.decode(gen, skip_special_tokens=True).strip()

# ---------------- run over the loss cases ----------------
os.makedirs(OUT_DIR, exist_ok=True)
cases = sorted(glob.glob(os.path.join(LOSS_DIR, "pid_*")),
               key=lambda p: int(os.path.basename(p).split("_")[1]))
print(f"Found {len(cases)} loss cases", flush=True)

all_records = []
t_inf = time.time()
for ci, cdir in enumerate(cases):
    info = json.load(open(os.path.join(cdir, "info.json")))
    img = Image.open(os.path.join(cdir, "image.png")).convert("RGB")
    pid, question = info["pid"], info["question"]
    premises = {}
    for mode_key, instr in MODES.items():
        premises[mode_key] = generate(img, build_prompt(question, instr))
        print(f"[case {ci+1}/{len(cases)} pid={pid}] {mode_key}: {premises[mode_key][:70]}", flush=True)
    rec = dict(pid=pid, question=question, ground_truth=info.get("ground_truth"),
               wrong_prediction=info.get("prediction"), category=info.get("category"),
               premises=premises)
    all_records.append(rec)
    with open(os.path.join(OUT_DIR, f"pid_{pid}.json"), "w") as f:
        json.dump(rec, f, indent=2, ensure_ascii=False)

# ---------------- combined markdown report ----------------
md = ["# Visual premises for MathVista loss cases", "",
      f"Model: `{MODEL_ID}` | {len(all_records)} cases x {len(MODES)} modes", ""]
for r in all_records:
    md.append(f"## pid {r['pid']}  (GT: {r['ground_truth']} | wrong pred: {r['wrong_prediction']})")
    md.append(f"**Q:** {r['question']}")
    md.append("")
    for mk, instr in MODES.items():
        md.append(f"- **{mk}** — {r['premises'][mk]}")
    md.append("")
with open(os.path.join(OUT_DIR, "premises_report.md"), "w") as f:
    f.write("\n".join(md))

print(f"\nDone in {time.time()-t_inf:.0f}s | wrote {len(all_records)} files + premises_report.md to {OUT_DIR}")
