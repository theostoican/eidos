import os, json, glob
os.environ.setdefault("HF_HOME", "/dev/shm/hf")
os.environ.setdefault("HF_DATASETS_CACHE", "/root/hf_datasets")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
from datasets import load_dataset

# Pre-bake the dataset-derived bits the analysis notebook needs (the exact vanilla
# `query` used as the control prompt, the answer choices, and the failure images),
# so the Colab notebook is self-contained after a git clone — no dataset download.

CTRL_DIR = "monolithic_best_of_n"
IMG_DIR  = "failure_examples"
os.makedirs(IMG_DIR, exist_ok=True)

recs = [json.load(open(p)) for p in glob.glob(os.path.join(CTRL_DIR, "pid_*.json"))]
failures = sorted([r for r in recs if not r["greedy_correct"] and r["greedy_finished"]],
                  key=lambda r: int(r["pid"]))
fpids = [r["pid"] for r in failures]
print("failure pids:", fpids)

ds = load_dataset("AI4Math/MathVista", split="testmini")
by_pid = {e["pid"]: e for e in ds}

data = {}
for pid in fpids:
    ex = by_pid[pid]
    data[pid] = dict(question=ex["question"], query=ex["query"],
                     choices=ex["choices"], answer=str(ex["answer"]),
                     answer_type=ex["answer_type"], question_type=ex["question_type"])
    ex["decoded_image"].convert("RGB").save(os.path.join(IMG_DIR, f"pid_{pid}.png"))

with open("analysis_data.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print(f"wrote analysis_data.json ({len(data)} failures) + {IMG_DIR}/pid_*.png")
