import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []
def md(s):   cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md(r"""# Best-of-N control vs two-stage premise inference — MathVista failures

**Pure analysis notebook — no model, no GPU.** It reads saved outputs from two runs on the
same 50 MathVista `testmini` examples (`Qwen/Qwen3.5-9B`, `enable_thinking=False`,
`max_new_tokens=1024`):

- **Control** — a plain, mode-free **monolithic** prompt (the dataset's vanilla `query`),
  answered **20×** with temperature sampling (T=0.8, top_p=0.95). Its deployable output is the
  **majority vote** over the 20 samples.
- **Two-stage (modes)** — the RQ2 pipeline run on the control's failures: for each of 20 focus
  modes, **generate a visual premise** (greedy), then **answer conditioned on it** (greedy).

**The question.** On the genuine failures the monolithic model makes, does the two-stage premise
pipeline recover answers the control's majority vote cannot? We compare the control's **deployable
majority vote** against the two-stage **oracle** (best of its 20 modes — an upper bound that
assumes a perfect mode-picker).

A *failure* here = the greedy baseline is **wrong and finished** (truncated generations are
excluded — a cut-off answer is a length artifact, not a wrong reading).""")

code(r"""# Colab bootstrap: clone the repo and cd into rq1/ if the outputs aren't already here
import os
if not os.path.exists("monolithic_best_of_n"):
    !git clone https://github.com/theostoican/eidos.git
    %cd eidos/mathvista/bestofn_vs_twostage""")

md("## 1. Load the saved outputs")

code(r"""import json, glob
from collections import Counter

def load_dir(d):
    return {os.path.basename(p)[4:-5]: json.load(open(p))
            for p in glob.glob(os.path.join(d, "pid_*.json"))}

ctrl = load_dir("monolithic_best_of_n")          # control: 50 examples
two  = load_dir("two_stage_on_failures")         # two-stage: run on the control's failures
data = json.load(open("analysis_data.json"))     # dataset-derived: query, choices, image path
csum = json.load(open("monolithic_best_of_n/summary.json"))

# failure set = greedy wrong & finished
failures = sorted([p for p, r in ctrl.items()
                   if not r["greedy_correct"] and r["greedy_finished"]], key=int)
print(f"{len(ctrl)} control examples · {len(failures)} genuine failures: {failures}")

# the 20 focus modes used by the two-stage pipeline (instruction text, for showing prompts)
MODES = {
    "1_salient":"Focus on the most salient, immediately obvious visual evidence.",
    "2_subtle":"Focus on small, fine-grained, or easily missed details.",
    "3_spatial":"Focus on spatial relations: relative position, alignment, order, direction, overlap, or connections between elements.",
    "4_ocr":"Focus only on text and symbols: OCR-readable labels, axis ticks, legends, units, symbols, and numbers.",
    "5_distractor":"Focus on a possible distractor or misleading cue that could tempt a wrong answer.",
    "6_alternative":"Offer an alternative interpretation of the image that differs from the most obvious reading.",
    "7_uncertainty":"Identify a source of visual uncertainty or ambiguity (occlusion, low resolution, overlapping marks, unclear arrow direction, etc.) that could affect the answer.",
    "8_people":"If there are people in the image, focus on who they could be (likely identity, role, age, or relationship based on visual cues such as clothing, setting, and appearance). If there are no people, say so.",
    "9_digits":"Focus on digits/numbers visible in the image and their relationship to the surrounding objects: do the digits count or quantify something in the image (e.g. labels, counts, tallies, axis values), and what do they refer to? Perhaps axes on a chart? If so, count them.",
    "10_point_on_curve":"When a point is marked on a plotted curve or line, read its y-value by counting tick marks up from the x-axis and report that number.",
    "11_value_curve_reaches":"Read the y-value the plotted curve reaches at the x-position the question is about, counting ticks up from the x-axis, and state the number.",
    "12_marker_on_vs_off":"Distinguish markers that lie ON the plotted curve from markers drawn OFF the curve, and report the y-coordinate of the one that lies on the curve.",
    "13_trace_to_x":"Trace along the plotted curve to the x-value the question asks about and read the y-coordinate there off the y-axis ticks.",
    "14_read_y_by_ticks":"Count the y-axis tick marks one at a time from the x-axis (0) up to the exact height of the relevant marker on the chart. Do not stop early or undercount. Report that tick count as its y-value.",
    "15_curve_height":"Report the height (y-value) of the curve at the relevant location as a number read off the axis, not as a qualitative description.",
    "16_two_points_pick_curve":"If two marked points share the same x-position, identify which one the curve actually passes through (or bends toward) and report that point's y-value.",
    "17_settle_value":"If the curve flattens, levels off, or settles near the target x, read the y-value of that level off the axis and report the number.",
    "18_coordinate_on_curve":"Give the (x, y) of the point where the curve meets the question's x-value, reading the y by counting ticks up from the x-axis.",
    "19_approach_height":"Read the y-height the curve approaches at the target x by counting ticks up from the x-axis, and report that number.",
    "20_hole_on_curve":"Count the y-axis tick marks one at a time starting from the x-axis (0), going up to the height of the open circle or marked point that lies on the curve. Do not stop early. Report that exact tick count as its y-value.",
}""")

md(r"""## 2. The prompts

**Control** sends the dataset's own `query` verbatim (answer-format hint + question) with the
image — no focus instruction. **Two-stage** uses two prompts per mode: a premise prompt that asks
for one visual premise (no answer yet), then an answer prompt that re-asks the question
conditioned on that premise.""")

code(r'''# exact prompt builders used by the two-stage pipeline
BASE = ("Given the image and the question, identify one possible visual premise "
        "that could be useful for solving the question.\n"
        "Do not answer the question yet.\n"
        "Focus only on visual evidence.\n"
        "State exactly one concise visual premise (1-2 sentences). Begin with 'Visual premise:'.")

def split_query(query):
    if "\nQuestion:" in query:
        hint, q = query.split("\nQuestion:", 1)
        return hint.strip(), q.strip()
    return "", query.strip()

def premise_prompt(question, mode_instr):
    return f"{BASE}\nMode: {mode_instr}\n\nQuestion: {question}"

def answer_prompt(query, premise_text):
    hint, question = split_query(query)
    premise_text = premise_text.replace("Visual premise:", "").strip()
    parts = [f"Question:\n{question}", "", f"Visual premise:\n{premise_text}", ""]
    if hint: parts.append(hint)
    parts += ["Now answer the question using the supported visual premise.",
              "Provide the final answer clearly."]
    return "\n".join(parts)

# show the templates on a concrete failure (pid 19)
pid = "19"; q = data[pid]["question"]; query = data[pid]["query"]
print("=== CONTROL prompt (vanilla query, no modes) ===\n")
print(query)
print("\n=== TWO-STAGE prompt 1 — premise (mode 13_trace_to_x) ===\n")
print(premise_prompt(q, MODES["13_trace_to_x"]))
print("\n=== TWO-STAGE prompt 2 — answer (conditioned on a premise) ===\n")
print(answer_prompt(query, "Visual premise: The bar for the relevant class reaches the 400 gridline."))''')

md(r"""## 3. The control on its 8 failures

On each failure we look at the spread of the 20 sampled answers and the **majority vote** the
control would actually return.""")

code(r"""def pretty(k):  # vote-bucket key -> readable answer
    return k.split(":", 1)[1] if k and ":" in k else k

rows = []
for p in failures:
    c = ctrl[p]
    rows.append((p, c["ground_truth"], c["greedy_pred"], c["vote_winner"],
                 c["vote_correct"], c["n_distinct_answers"], c["top_share"]))

print(f"{'pid':>4} {'gt':<10} {'greedy':<10} {'vote':<12} {'vote_ok':<8} {'distinct':<9} top_share")
print("-"*70)
for p, gt, g, v, vok, nd, ts in rows:
    print(f"{p:>4} {gt[:9]:<10} {str(g)[:9]:<10} {str(v)[:11]:<12} "
          f"{('YES' if vok else 'no'):<8} {nd:<9} {ts:.2f}")

vote_rec = sum(ctrl[p]["vote_correct"] for p in failures)
print(f"\nControl MAJORITY VOTE recovers {vote_rec}/{len(failures)} of the genuine failures.")""")

md(r"""The control's vote recovers **none** of its genuine failures: even on the cases with real
answer spread, the plurality lands on the wrong reading. (The fully-collapsed cases — `distinct=1`
— are pids 7, 20, 50, which are benchmark label-noise: a phrasing technicality and two
gold-label errors where the model is arguably right.)""")

md(r"""## 4. Control majority vote vs two-stage oracle

The headline comparison: the control's **deployable** majority vote against the two-stage
**oracle** (fixed by ≥1 of its 20 modes).""")

code(r"""ctrl_vote_fixed  = [p for p in failures if ctrl[p]["vote_correct"]]
two_oracle_fixed = [p for p in failures if two.get(p, {}).get("fixed_by_any")]
two_vote_fixed   = [p for p in failures if two.get(p, {}).get("vote_correct")]

n = len(failures)
print(f"of {n} genuine failures:\n")
print(f"  control  · majority vote (deployable) : {len(ctrl_vote_fixed)}/{n}  {ctrl_vote_fixed}")
print(f"  two-stage· oracle  (best of 20 modes) : {len(two_oracle_fixed)}/{n}  {two_oracle_fixed}")
print(f"  two-stage· majority vote (deployable) : {len(two_vote_fixed)}/{n}  {two_vote_fixed}")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 3.6))
labels = ["control\nmajority vote", "two-stage\noracle", "two-stage\nmajority vote"]
vals   = [len(ctrl_vote_fixed), len(two_oracle_fixed), len(two_vote_fixed)]
bars = ax.bar(labels, vals, color=["#888", "#3b7", "#bd7"])
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.05, str(v), ha="center", fontweight="bold")
ax.set_ylim(0, n); ax.set_ylabel(f"failures recovered (of {n})")
ax.set_title("Recovery on the control's genuine failures")
plt.tight_layout(); plt.show()""")

md(r"""**Two-stage oracle recovers 2/8 (pids 19, 27); the control's vote recovers 0/8.** So the
premise pipeline, at its best case, fixes two failures the deployable control cannot. Put both on
equal (deployable) footing — vote vs vote — and the gap is a single case (pid 27).""")

md(r"""## 5. Worked examples — with the actual prompts

For the two failures two-stage fixes, we show the image, the control's vote (wrong), and the
two-stage mode that recovered it: its premise prompt, the premise the model produced, the answer
prompt, and the resulting answer.""")

code(r"""from IPython.display import display, Image, Markdown

def show_case(pid):
    c = ctrl[pid]; t = two[pid]; d = data[pid]
    display(Markdown(f"### pid {pid} — ground truth `{d['answer']}`"))
    display(Image(filename=f"failure_examples/pid_{pid}.png", width=420))
    display(Markdown(f"**Question:** {d['question']}"))

    # control side: the answer distribution over 20 samples + the (wrong) vote
    dist = ", ".join(f"`{pretty(k)}`×{v}" for k, v in
                     sorted(c["answer_counts"].items(), key=lambda kv: -kv[1]))
    display(Markdown(
        f"**Control (vanilla prompt, 20 samples):** answers = {dist}  \n"
        f"→ majority vote = `{c['vote_winner']}`  ·  correct? **{c['vote_correct']}**"))

    # two-stage side: the winning mode, its premise prompt, premise, answer prompt, answer
    mode = t["correct_modes"][0]
    pm = t["per_mode"][mode]
    display(Markdown(f"**Two-stage — recovering mode `{mode}`:** _{MODES[mode]}_"))
    display(Markdown("**Premise prompt →**\n```\n" + premise_prompt(d["question"], MODES[mode]) + "\n```"))
    display(Markdown(f"**Premise produced →** {pm['premise']}"))
    display(Markdown("**Answer prompt →**\n```\n" + answer_prompt(d["query"], pm["premise"]) + "\n```"))
    ans = pm["answer"].strip()
    ans = ans if len(ans) < 600 else ans[:600] + " …"
    display(Markdown(f"**Answer →** {ans}\n\n→ extracted `{pm['prediction']}` · correct? **{pm['correct']}**"))
    display(Markdown("---"))

for pid in two_oracle_fixed:
    show_case(pid)""")

md(r"""## 6. Takeaways

- On the monolithic model's **8 genuine failures**, the control's deployable **majority vote
  recovers 0**.
- **Two-stage oracle recovers 2** (pids 19, 27) — both cases where the control already had answer
  spread but its vote picked wrong. Deployable-to-deployable (vote vs vote), two-stage recovers
  **1** (pid 27).
- So two-stage's advantage shows up only at the **oracle** ceiling (a perfect mode-picker, 2×
  the model calls), and it is **2 cases out of 8** — direction and mechanism, not a measured
  effect size (n=8).
- The unrecoverable failures (pids 7, 20, 50) are **benchmark label noise**, where the model is
  confidently giving an answer that is arguably correct but disagrees with the gold label —
  nothing a premise can or should "fix".""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3"}

print("executing notebook ...")
client = NotebookClient(nb, timeout=600, kernel_name="python3",
                        resources={"metadata": {"path": "/root/eidos/mathvista/bestofn_vs_twostage"}})
client.execute()
with open("analysis.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote analysis.ipynb (executed, outputs embedded)")
