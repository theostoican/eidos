#!/usr/bin/env python
"""Sample 16 answers per question from a model served by vLLM (serve.sh eval), then report voting metrics.
  python evaluate.py run --tag collision --set mmmu [--adapter runs/collision/final] [--model MODEL] [--limit N]
  python evaluate.py report
Sets: mmstar (MMStar val, 1,500), mmmu (MMMU-Pro standard-10 test, single image, 1,591), virl (ViRL39K held-out, 800).
Writes results/<set>/<tag>.jsonl (one row per question: id, gold, answers; None = no parsed answer); resumes by id."""
import argparse, asyncio, collections, json, os, re, zlib
from pathlib import Path
import numpy as np
from PIL import Image
from common import MODEL, SAMPLING, PROMPT_STANDARD, LETTERS, cap, content, parse_answer
HERE = Path(__file__).resolve().parent; RES = HERE / "results"
SETS = ("mmstar", "mmmu", "virl")

def items(which, data_root):
    if which == "virl":
        return [{"id": x["qid"], "text": x["question"] + "\n" + PROMPT_STANDARD, "gold": x["gold"], "n": x["n_options"],
                 "im": lambda x=x: Image.open(Path(data_root) / x["image"])}
                for x in map(json.loads, open(HERE / "data" / "pool.jsonl")) if x["split"] == "heldout"]
    from datasets import load_dataset
    if which == "mmstar":
        return [{"id": r["index"], "text": r["question"] + "\n" + PROMPT_STANDARD, "gold": r["answer"], "n": 4, "im": lambda r=r: r["image"]}
                for r in load_dataset("Lin-Chen/MMStar", split="val")]
    import ast
    out, img = [], re.compile(r"<image\s+(\d+)>")
    for r in load_dataset("MMMU/MMMU_Pro", "standard (10 options)", split="test"):
        if r.get("image_2") is not None or r.get("image_1") is None: continue          # single-image questions only
        opts = ast.literal_eval(r["options"]) if isinstance(r["options"], str) else r["options"]
        if len(opts) > 10: continue                                                     # one question has 12 options
        text = img.sub("[image]", f"{r['question']}\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)) + f"\n{PROMPT_STANDARD}")
        out.append({"id": r["id"], "text": text, "gold": r["answer"], "n": len(opts), "im": lambda r=r: r["image_1"]})
    return out

async def run(a):
    import aiohttp
    its = items(a.set, a.data_root)[: a.limit or None]; f = RES / a.set / f"{a.tag}.jsonl"; f.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["id"] for l in open(f)} if f.exists() else set(); its = [x for x in its if x["id"] not in done]
    print(len(its), "to run", flush=True); served = a.model; sem = asyncio.Semaphore(48)
    async with aiohttp.ClientSession() as sess:
        if a.adapter:                                       # the server resolves paths against its own directory: send an absolute one
            path = str(Path(a.adapter).resolve()); served = a.tag
            async with sess.post(a.url + "/v1/load_lora_adapter", json={"lora_name": served, "lora_path": path}) as r:
                txt = await r.text()
                if r.status != 200 and "already" not in txt.lower(): raise SystemExit(f"adapter {path} not loaded: {r.status} {txt[:300]}")
        with open(f, "a") as out:
            async def one(x):
                async with sem:                             # encode the image only when the request is about to go out
                    body = {"model": served, "messages": [{"role": "user", "content": content(x["text"], cap(x["im"]()))}],
                            "n": 16, "max_tokens": a.max_tokens, **SAMPLING}
                    for t in range(5):
                        try:
                            async with sess.post(a.url + "/v1/chat/completions", json=body, timeout=3600) as r:
                                j = await r.json()
                                if "choices" not in j: raise RuntimeError(str(j)[:300])
                                break
                        except Exception as e:
                            if t == 4: raise RuntimeError(f"question {x['id']} failed after 5 attempts: {e!r}")
                            await asyncio.sleep(5 * (t + 1))
                ans = [parse_answer(c["message"]["content"] or "", x["n"]) if c["finish_reason"] == "stop" else None for c in j["choices"]]
                out.write(json.dumps({"id": x["id"], "gold": x["gold"], "answers": ans}) + "\n"); out.flush()
            await asyncio.gather(*[one(x) for x in its])
    print("done", flush=True)

def metrics(b, g):
    """pass@1, maj@16 (ties split), error collision on all votes and on parsed votes only, unparsed share"""
    c = collections.Counter(x for x in b if x is not None); maj = 0.0
    if c:
        m = max(c.values()); top = [x for x, n in c.items() if n == m]; maj = (g in top) / len(top)
    w = collections.Counter(x for x in b if x is not None and x != g); n, k = len(b), sum(c.values()); pairs = sum(v * (v - 1) for v in w.values())
    return [np.mean([x == g for x in b]), maj, pairs / (n * (n - 1)), pairs / (k * (k - 1)) if k > 1 else np.nan, 1 - k / n]

def ci(d, key):
    """mean and 95% question-bootstrap CI; the resampling is seeded per contrast, so a CI never depends on other files"""
    d = d[~np.isnan(d)]; rng = np.random.default_rng(zlib.crc32(key.encode()))
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
    return f"{100 * d.mean():+.2f} [{100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}]"

def report(a):
    out = ["# Results (16 samples per question; 95% question-bootstrap CIs; arms pooled over the seeds present)\n"]
    for s in SETS:
        if not (RES / s).is_dir(): continue
        D = {p.stem: {r["id"]: r for r in map(json.loads, open(p))} for p in sorted((RES / s).glob("*.jsonl")) if p.stat().st_size}
        if not D: continue
        ids = sorted(set.intersection(*[set(d) for d in D.values()]), key=str)
        S = {t: np.array([metrics(D[t][i]["answers"], D[t][i]["gold"]) for i in ids]) for t in D}
        out += [f"## {s} (n = {len(ids)} questions)\n", "| run | pass@1 | maj@16 | voting gain | error collision | parsed-only collision | unparsed |",
                "| --- | --- | --- | --- | --- | --- | --- |"]
        out += [f"| {t} | " + " | ".join(f"{v:.4f}" for v in (np.nanmean(S[t][:, 0]), np.nanmean(S[t][:, 1]), np.nanmean(S[t][:, 1] - S[t][:, 0]),
                np.nanmean(S[t][:, 2]), np.nanmean(S[t][:, 3]), np.nanmean(S[t][:, 4]))) + " |" for t in S]
        arm = lambda name: [t for t in S if t == name or re.fullmatch(rf"{name}_s\d+", t)]
        pooled = lambda name: np.mean([S[t] for t in arm(name)], axis=0)
        out.append("")
        for x, y in (("collision", "grpo"), ("collision", "csr"), ("collision", "edas"), ("collision", "dfuser"), ("collision", "base"), ("grpo", "base")):
            if not arm(x) or not arm(y): continue
            X, Y = pooled(x), pooled(y); k = f"{s}:{x}-{y}"
            out.append(f"- {x} ({len(arm(x))} seeds) - {y} ({len(arm(y))}): maj@16 {ci(X[:, 1] - Y[:, 1], k + ':maj')}; pass@1 {ci(X[:, 0] - Y[:, 0], k + ':p1')}; "
                       f"voting gain {ci((X[:, 1] - X[:, 0]) - (Y[:, 1] - Y[:, 0]), k + ':vg')}; parsed-only collision {ci(X[:, 3] - Y[:, 3], k + ':pc')}")
        out.append("")
    txt = "\n".join(out); print(txt); (RES / "report.md").write_text(txt + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["run", "report"]); ap.add_argument("--tag"); ap.add_argument("--set", choices=SETS)
    ap.add_argument("--model", default=MODEL); ap.add_argument("--adapter", default=""); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024); ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--data-root", default=os.environ.get("VIRL39K_ROOT", str(HERE / "data" / "virl39k")))
    a = ap.parse_args()
    if a.cmd == "run" and not (a.tag and a.set): ap.error("run needs --tag and --set")
    asyncio.run(run(a)) if a.cmd == "run" else report(a)
