#!/usr/bin/env python
"""Generate mmk12_vision_analysis.ipynb (mirrors the MMMU-Pro analysis, adapted to MMK12)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s):  cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# MMK12 headroom analysis with Qwen2.5-VL-7B-Instruct

**Goal:** Estimate the headroom remaining on the **MMK12** multimodal-reasoning benchmark for a
strong open VLM, and construct a curated set of **wrong predictions attributable to multimodal
issues** (perception / OCR / chart / diagram / table reading) rather than pure knowledge or
reasoning gaps. This mirrors the companion MMMU-Pro (vision) study, adapted to MMK12's format.

**Setup**
- Model: `Qwen/Qwen2.5-VL-7B-Instruct` (best open VLM that fits a 24 GB RTX 4090 in bf16).
- Benchmark: `FanqingM/MMK12` (the eval set released with *MM-Eureka*), `test` split — 2000 K-12
  science multiple-choice items, balanced across **math / physics / chemistry / biology**
  (500 each), 4-way (A–D).
- Scope: ~200 samples (configurable via `N_SAMPLES`).

**Key format difference vs MMMU-Pro vision.** In MMMU-Pro *vision* the entire question (stem +
options) is rendered *into* the screenshot, so OCR of the options is the perception task. In
**MMK12 the question stem and the options are plain text**, and the **image is a supporting
figure** (a table, plot, geometry diagram, circuit, structural formula, …). The model must *read
the figure* and combine it with the text. This changes the perception probe (see the diagnosis
section): there is no gold option-text to OCR-match against, so we replace the ground-truthed
fuzzy-match probe with an **image-anchored faithfulness judge**.

**Pipeline:** load data → CoT inference → parse `Answer: X` → score / headroom → categorize errors
→ dump multimodal-failure set.""")

code(r"""import os, re, json, time, random
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---- config ----
MODEL_ID    = "Qwen/Qwen2.5-VL-7B-Instruct"
N_SAMPLES   = 200          # MMK12 test samples to evaluate
MAX_NEW_TOK = 1024         # room for chain-of-thought + final "Answer: X"
SEED        = 0
OUT_DIR     = "/workspace/mmk12_out"
os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda"
print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))""")

code(r"""# ---- load MMK12 test split ----
ds = load_dataset("FanqingM/MMK12", split="test")
print("full test split size:", len(ds))
print("features:", {k: str(v) for k, v in ds.features.items()})

from collections import Counter
print("subjects:", dict(Counter(ds["subject"])))

ex = ds[0]
for k, v in ex.items():
    print(f"  {k:10s}:", (f"PIL {v.size} {v.mode}" if hasattr(v, "size") else repr(v)[:160]))

# Deterministic subsample of N_SAMPLES (same shuffle-then-take recipe as the MMMU-Pro study).
idx = list(range(len(ds)))
random.Random(SEED).shuffle(idx)
sel = sorted(idx[:N_SAMPLES])
subset = ds.select(sel)
print("\nEvaluating", len(subset), "samples | subject mix:", dict(Counter(subset["subject"])))""")

code(r"""# ---- load model + processor ----
t0 = time.time()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map=DEVICE,
).eval()
# Cap pixel budget so big figures don't blow up the visual token count / VRAM.
processor = AutoProcessor.from_pretrained(
    MODEL_ID, min_pixels=256*28*28, max_pixels=1280*28*28,
)
print(f"loaded in {time.time()-t0:.1f}s | VRAM {torch.cuda.memory_allocated()/1e9:.1f} GB")""")

code(r'''# ---- prompt + inference ----
# Unlike MMMU-Pro vision, the MMK12 question stem AND its A-D options are plain text, while the
# image carries the supporting figure. So the text prompt is the question itself plus an
# answer-format instruction; the image is attached alongside.
PROMPT_SUFFIX = (
    "\n\nLook carefully at the image, then reason step by step to solve the question. "
    "The last line of your response must be exactly of the form 'Answer: $LETTER' "
    "where $LETTER is the single letter (A, B, C, D) of the correct option."
)

def to_rgb(image):
    # MMK12 images are a mix of RGBA / L / RGB; flatten onto white so the VLM sees clean content.
    if image.mode == "RGB":
        return image
    from PIL import Image as _Img
    if image.mode in ("RGBA", "LA", "P"):
        im = image.convert("RGBA")
        bg = _Img.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        return bg
    return image.convert("RGB")

ANS_RE = re.compile(r"Answer:\s*\(?([A-E])\)?", re.IGNORECASE)

def parse_answer(text: str):
    """Extract the final answer letter; prefer the last 'Answer: X', else last bare letter."""
    m = list(ANS_RE.finditer(text))
    if m:
        return m[-1].group(1).upper()
    m2 = list(re.finditer(r"\b([A-E])\b", text))
    return m2[-1].group(1).upper() if m2 else None

@torch.inference_mode()
def generate(image, prompt, max_new_tokens):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": to_rgb(image)},
        {"type": "text",  "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(DEVICE)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = gen[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]

def run_one(ex):
    return generate(ex["image"], ex["question"] + PROMPT_SUFFIX, MAX_NEW_TOK)

# smoke test on one sample after the model loads
_t = run_one(subset[0])
print(_t[-400:])
print("--> parsed:", parse_answer(_t), "| gold:", subset[0]["answer"])''')

code(r"""# ---- run inference over the subset (resumable from checkpoint) ----
results, done_ids = [], set()
_ckpt = f"{OUT_DIR}/raw_results.json"
if os.path.exists(_ckpt):
    with open(_ckpt) as f:
        results = json.load(f)
    done_ids = {r["i"] for r in results}
    print(f"resuming inference: {len(results)} samples already done")

t0 = time.time()
for i, ex in enumerate(subset):
    if i in done_ids:
        continue
    out = run_one(ex)
    pred = parse_answer(out)
    gold = str(ex["answer"]).strip().upper()
    results.append({
        "i": i,
        "id": ex.get("id", i),
        "subject": ex.get("subject"),
        "gold": gold,
        "pred": pred,
        "correct": (pred == gold),
        "question": ex.get("question"),
        "raw": out,
    })
    if (i + 1) % 10 == 0 or i == 0:
        acc = sum(r["correct"] for r in results) / len(results)
        el = time.time() - t0
        msg = (f"[{i+1:3d}/{len(subset)}] running acc={acc:.3f} | "
               f"{el:.0f}s ({el/(i+1):.1f}s/ex)")
        print(msg, flush=True)
        with open(f"{OUT_DIR}/raw_results.json", "w") as f:
            json.dump(results, f, indent=2)
        with open(f"{OUT_DIR}/progress.log", "a") as f:
            f.write(msg + "\n")

with open(f"{OUT_DIR}/raw_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("saved", f"{OUT_DIR}/raw_results.json")""")

code(r"""# ---- accuracy + headroom ----
import math
n = len(results)
n_correct = sum(r["correct"] for r in results)
n_unparsed = sum(r["pred"] is None for r in results)
acc = n_correct / n

# Framing reference points for MMK12. These are APPROXIMATE anchors from the MM-Eureka report
# (which released MMK12) plus the 4-way-MC chance rate; our own measured number above is the
# ground truth, the rest are only to size the headroom.
REFS = {
    "Chance (4-way MC)":                    0.25,
    "Qwen2.5-VL-7B (MM-Eureka rpt, approx)": 0.50,
    "MM-Eureka-Qwen-7B / RL (approx)":       0.60,
    "Strong proprietary VLM (approx)":       0.70,
}

print(f"N evaluated         : {n}")
print(f"Correct             : {n_correct}")
print(f"Unparsed predictions: {n_unparsed}")
print(f"Accuracy            : {acc:.3f}")
se = math.sqrt(acc*(1-acc)/n)
print(f"95% CI              : [{acc-1.96*se:.3f}, {acc+1.96*se:.3f}]")

# Per-subject accuracy (MMK12 is balanced across 4 subjects).
from collections import Counter, defaultdict
by_subj = defaultdict(lambda: [0, 0])
for r in results:
    by_subj[r["subject"]][0] += int(r["correct"]); by_subj[r["subject"]][1] += 1
print("\nPer-subject accuracy:")
for s, (c, t) in sorted(by_subj.items()):
    print(f"  {s:10s} {c/t:.3f}  ({c}/{t})")

print("\nHeadroom vs reference points:")
for k, v in REFS.items():
    print(f"  {k:36s} {v:.2f}   gap = {v-acc:+.3f}")
print(f"\nWrong predictions to analyze: {n - n_correct}")""")

md(r"""## Diagnosing *why* each wrong prediction failed

Same intent as the MMMU-Pro study — separate **multimodal/perception** failures from pure
**reasoning/knowledge** failures — but the perception signal is adapted to MMK12's format.

Two complementary signals per wrong item:

1. **Image-faithfulness probe (perception-isolated).** In MMMU-Pro vision we could OCR the options
   straight off the screenshot and fuzzy-match them against the gold option strings. In MMK12 the
   options are already text, and the image is a *figure* with no gold transcription to match
   against. So we instead (a) ask the model to **describe exactly what the figure shows** (all
   numbers, labels, structures), then (b) ask a fresh **image-grounded judge** that actually looks
   at the figure to rate how *faithful* that description is (0–1) and name any missed/wrong detail.
   A low faithfulness score isolates a perception failure independent of the downstream reasoning.

2. **Image-grounded error judge.** Show the figure + gold + the model's own reasoning, classify the
   PRIMARY cause into a taxonomy, and flag whether it is multimodal-attributable.

An item is counted **multimodal-attributable** if EITHER the faithfulness probe shows the model
misread the figure, OR the error judge attributes the primary cause to perception.""")

code(r'''# ---- diagnostic functions ----
TAXONOMY = ["OCR_TEXT", "CHART_PLOT", "TABLE", "DIAGRAM_SPATIAL",
            "FINE_DETAIL", "COUNTING", "REASONING", "KNOWLEDGE", "FORMAT_PARSING"]
MULTIMODAL_CATS = {"OCR_TEXT", "CHART_PLOT", "TABLE", "DIAGRAM_SPATIAL",
                   "FINE_DETAIL", "COUNTING"}

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# (1) perception-isolated probe: describe-then-judge -------------------------------------------
DESCRIBE_PROMPT = (
    "Describe EXACTLY what this image shows for the purpose of answering a science question: "
    "list every number, axis label, table value, symbol, structure and spatial relationship you "
    "can see. Do NOT try to answer any question — only report what is visually present."
)

FAITH_TMPL = (
    "Below is a model's textual description of the image shown. Look at the image YOURSELF and "
    "judge how faithfully the description matches what is actually in the image.\n"
    "DESCRIPTION:\n\"\"\"\n{desc}\n\"\"\"\n\n"
    "Respond with ONLY a JSON object: "
    '{{"faithfulness": <0-1>, "missed_or_wrong": "<=25 words"}}. '
    "faithfulness=1 means every salient visual fact is correct and present; "
    "lower it for each missed or misread number/label/structure."
)

def perception_probe(image):
    desc = generate(image, DESCRIBE_PROMPT, 400)
    out  = generate(image, FAITH_TMPL.format(desc=desc[:1500]), 200)
    m = JSON_RE.search(out)
    score = None
    if m:
        try:
            j = json.loads(m.group(0)); score = float(j.get("faithfulness"))
        except Exception:
            score = None
    return desc, score

# (2) image-grounded error judge ---------------------------------------------------------------
JUDGE_TMPL = (
    "You are auditing why a vision-language model got a multiple-choice question WRONG. "
    "The supporting figure is the image shown; the question and its options were given as text.\n"
    "QUESTION (with options):\n\"\"\"\n{q}\n\"\"\"\n"
    "GOLD answer: {gold}\nMODEL answer: {pred}\n"
    "MODEL's reasoning was:\n\"\"\"\n{trace}\n\"\"\"\n\n"
    "First, look at the image yourself and verify what it actually shows. Then decide the "
    "PRIMARY cause of the model's error, choosing exactly one category:\n"
    "- OCR_TEXT: misread/garbled TEXT inside the figure\n"
    "- CHART_PLOT: misread values/axes/trends in a chart, graph or plot\n"
    "- TABLE: misread tabular data\n"
    "- DIAGRAM_SPATIAL: misread spatial/geometric/structural relations or a diagram\n"
    "- FINE_DETAIL: missed small / low-contrast / partially-occluded visual detail\n"
    "- COUNTING: miscounted objects/items\n"
    "- REASONING: read the image correctly but reasoned/calculated incorrectly\n"
    "- KNOWLEDGE: lacked the domain knowledge regardless of perception\n"
    "- FORMAT_PARSING: answer was correct in spirit but mis-formatted / unparseable\n\n"
    "Respond with ONLY a JSON object: "
    '{{"category": "<one>", "multimodal": <true|false>, '
    '"confidence": <0-1>, "evidence": "<=25 words"}}. '
    "Set multimodal=true iff the primary cause is a failure to correctly PERCEIVE the image."
)

def judge(image, q, gold, pred, trace):
    out = generate(image, JUDGE_TMPL.format(q=q[:1200], gold=gold, pred=pred, trace=trace[:1800]), 200)
    m = JSON_RE.search(out)
    if not m:
        return {"category": "UNKNOWN", "multimodal": None, "confidence": 0, "evidence": out[:120]}
    try:
        j = json.loads(m.group(0))
        if j.get("category") not in TAXONOMY:
            j["category"] = "UNKNOWN"
        return j
    except Exception:
        return {"category": "UNKNOWN", "multimodal": None, "confidence": 0, "evidence": out[:120]}

print("diagnostic functions ready")''')

code(r'''# ---- run diagnosis over wrong predictions ----
FAITH_FAIL_THR = 0.6   # image-description faithfulness below this => perception/figure-reading failure

wrong = [r for r in results if not r["correct"]]

# resume from checkpoint if present
diagnosed, _ddone = [], set()
_dpath = f"{OUT_DIR}/diagnosed_wrong.json"
if os.path.exists(_dpath):
    with open(_dpath) as f:
        diagnosed = json.load(f)
    _ddone = {d["i"] for d in diagnosed}
todo = [r for r in wrong if r["i"] not in _ddone]
print(f"{len(wrong)} wrong predictions; {len(diagnosed)} already diagnosed, "
      f"{len(todo)} to go...", flush=True)

for k, r in enumerate(todo):
    ex  = subset[r["i"]]
    img = ex["image"]
    desc, faith = perception_probe(img)
    jr = judge(img, r["question"], r["gold"], r["pred"], r["raw"])

    cat = jr.get("category", "UNKNOWN")
    judge_mm = bool(jr.get("multimodal"))
    faith_mm = (faith is not None and faith < FAITH_FAIL_THR)
    # multimodal-attributable if EITHER the image-faithfulness probe shows misreading,
    # OR the judge attributes it to perception (category in the multimodal set).
    is_multimodal = bool(faith_mm or judge_mm or (cat in MULTIMODAL_CATS))

    diagnosed.append({**r,
        "category": cat,
        "judge_multimodal": judge_mm,
        "img_faithfulness": faith,
        "img_misread": faith_mm,
        "is_multimodal": is_multimodal,
        "confidence": jr.get("confidence"),
        "evidence": jr.get("evidence"),
        "img_description": desc,
    })
    if (k + 1) % 5 == 0 or k == 0:
        msg = f"  [{k+1}/{len(todo)}] last: cat={cat} mm={is_multimodal} faith={faith}"
        print(msg, flush=True)
        with open(f"{OUT_DIR}/diagnosed_wrong.json", "w") as f:
            json.dump(diagnosed, f, indent=2)
        with open(f"{OUT_DIR}/progress.log", "a") as f:
            f.write(msg + "\n")

with open(f"{OUT_DIR}/diagnosed_wrong.json", "w") as f:
    json.dump(diagnosed, f, indent=2)
print("saved", f"{OUT_DIR}/diagnosed_wrong.json")''')

code(r'''# ---- summary + curated multimodal-failure set ----
from collections import Counter

cats = Counter(d["category"] for d in diagnosed)
mm_set = [d for d in diagnosed if d["is_multimodal"]]

print("="*64)
print("MMK12 — Qwen2.5-VL-7B-Instruct")
print("="*64)
print(f"Accuracy                : {acc:.3f}  ({n_correct}/{n})")
print(f"Total wrong             : {len(wrong)}")
print(f"Multimodal-attributable : {len(mm_set)}  "
      f"({len(mm_set)/max(len(wrong),1):.0%} of errors, "
      f"{len(mm_set)/n:.0%} of all items)")
print("\nError category breakdown (of wrong preds):")
for c, v in cats.most_common():
    tag = "  <-- multimodal" if c in MULTIMODAL_CATS else ""
    print(f"  {c:16s} {v:3d}{tag}")

# Headroom interpretation
recoverable = len(mm_set) / n
print(f"\nHeadroom attributable to multimodal/perception issues: ~{recoverable:.0%} "
      f"of items (i.e. fixing figure-reading alone could lift accuracy from "
      f"{acc:.2f} toward {acc+recoverable:.2f}).")

# Save the curated set (drop bulky raw fields for the headline export)
curated = [{
    "id": d["id"], "subject": d["subject"], "gold": d["gold"], "pred": d["pred"],
    "category": d["category"], "is_multimodal": d["is_multimodal"],
    "judge_multimodal": d["judge_multimodal"], "img_misread": d["img_misread"],
    "img_faithfulness": d["img_faithfulness"], "confidence": d["confidence"],
    "evidence": d["evidence"],
} for d in mm_set]
with open(f"{OUT_DIR}/multimodal_failures.json", "w") as f:
    json.dump({"summary": {"accuracy": acc, "n": n, "n_wrong": len(wrong),
                           "n_multimodal": len(mm_set),
                           "per_subject": {s: f"{c}/{t}" for s, (c, t) in sorted(by_subj.items())},
                           "categories": dict(cats)},
               "cases": curated}, f, indent=2)

# Export up to 12 example figures for visual inspection.
# MMK12 is ordered by subject, so naive mm_set[:12] would be all from one subject;
# pick a set that is DIVERSE across (subject, category) so the examples are representative.
def _pick_diverse(cases, k):
    chosen, cnt = [], Counter()
    pool = list(cases)
    while pool and len(chosen) < k:
        pool.sort(key=lambda c: (cnt[(c["subject"], c["category"])], cnt[c["subject"]], cnt[c["category"]]))
        c = pool.pop(0); chosen.append(c)
        cnt[(c["subject"], c["category"])] += 1; cnt[c["subject"]] += 1; cnt[c["category"]] += 1
    return chosen

ex_dir = f"{OUT_DIR}/multimodal_examples"; os.makedirs(ex_dir, exist_ok=True)
def _safe(s): return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))[:60]
examples = _pick_diverse(mm_set, 12)
for d in examples:
    to_rgb(subset[d["i"]]["image"]).save(
        f"{ex_dir}/{_safe(d['id'])}_{d['subject']}_{d['category']}_gold{d['gold']}_pred{d['pred']}.png")
print("example mix -> subjects:", dict(Counter(d["subject"] for d in examples)),
      "| categories:", dict(Counter(d["category"] for d in examples)))

print(f"\nSaved: {OUT_DIR}/multimodal_failures.json  (+ {len(examples)} example PNGs)")
print("\n--- sample multimodal failures ---")
for d in mm_set[:6]:
    print(f"[{d['id']}] {d['subject']}/{d['category']} gold={d['gold']} pred={d['pred']} "
          f"faith={d['img_faithfulness']} :: {d['evidence']}")''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
with open("/workspace/mmk12_vision_analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote /workspace/mmk12_vision_analysis.ipynb with", len(cells), "cells")
