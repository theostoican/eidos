#!/usr/bin/env python
"""Single-GPU GRPO with LoRA for vision-language models, with the collision reward.
Rollouts come from a vLLM server (serve.sh train) that hot-loads the current LoRA adapter after every step; the update runs
in-process with Hugging Face Transformers + PEFT. Advantages are Dr. GRPO's (reward minus the group mean, no division by
the standard deviation; loss normalised by a constant number of tokens), one on-policy update per step, no KL term.
  python train.py --arm collision --seed 0 --out runs/collision
Arms: grpo (correctness only) | collision | csr | edas | dfuser. Earlier run names: iid = grpo, rahp_noref = collision."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import argparse, asyncio, json, random, shutil, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import aiohttp, torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from common import MODEL, SAMPLING, PROMPT_STANDARD, cap, content, parse_answer, correctness, collision, csr, dfuser, edas_adjust, floor_spoiled
HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["grpo", "collision", "csr", "edas", "dfuser"])
ap.add_argument("--coll-lam", type=float, default=1.0, help="collision penalty weight (1.0 in the reported runs)")
ap.add_argument("--csr-lam", type=float, default=0.5, help="CSR agreement weight")
ap.add_argument("--spoiled", choices=["zero", "min"], default="zero",
                help="reward of a spoiled rollout (no parsed answer) in collision and csr: 'zero' as in the reported runs; "
                     "'min' = -lambda, never above any wrong answer (and csr then ignores spoiled peers)")
ap.add_argument("--steps", type=int, default=60); ap.add_argument("--q", type=int, default=16, help="questions per step")
ap.add_argument("--G", type=int, default=8, help="rollouts per question"); ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--rank", type=int, default=16); ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--norm", type=float, default=1024.0, help="loss normaliser: tokens per group (Dr. GRPO)")
ap.add_argument("--mb", type=int, default=2, help="training micro-batch (rollouts per forward pass)")
ap.add_argument("--pool", default=str(HERE / "data" / "pool.jsonl")); ap.add_argument("--data-root", default=os.environ.get("VIRL39K_ROOT", str(HERE / "data" / "virl39k")))
ap.add_argument("--url", default="http://127.0.0.1:8100"); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
OUT = Path(a.out).resolve(); (OUT / "adapters").mkdir(parents=True, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed); RUNID = str(int(time.time()) % 1000000)
POOL = [x for x in map(json.loads, open(a.pool)) if x.get("split", "train") == "train"]     # the held-out questions are never drawn
if a.seed: random.Random(a.seed).shuffle(POOL)                                                # seed 0 keeps the pool order
(OUT / "config.json").write_text(json.dumps({**vars(a), "model": MODEL, "sampling": SAMPLING, "runid": RUNID}, indent=1))

def group_rewards(answers, gold):
    if a.arm == "collision": r = collision(answers, gold, a.coll_lam); lam = a.coll_lam
    elif a.arm == "csr": r = csr(answers, gold, a.csr_lam, parsed_only=a.spoiled == "min"); lam = a.csr_lam
    elif a.arm == "dfuser": return dfuser(answers, gold, random)
    else: return correctness(answers, gold)                                                  # grpo, edas
    return floor_spoiled(r, answers, -lam) if a.spoiled == "min" else r

proc = AutoProcessor.from_pretrained(MODEL, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0,
                       target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"))
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)

async def post(sess, path, body):
    """POST with retries. Adapter load/unload are not idempotent: if a retry finds the adapter already loaded (or already
    gone), the earlier attempt succeeded, so that counts as success."""
    for t in range(6):
        try:
            async with sess.post(a.url + path, json=body, timeout=3600) as r:
                txt = await r.text()
                if r.status != 200:
                    if t and "lora_adapter" in path and any(s in txt.lower() for s in ("already", "not found", "does not exist")): return txt
                    raise RuntimeError(f"{r.status} {txt[:300]}")
                return json.loads(txt) if txt.strip().startswith("{") else txt
        except Exception as e:
            if t == 5: raise
            print("[retry]", path, repr(e)[:200], flush=True); await asyncio.sleep(5 * (t + 1))

async def rollouts(sess, items, served):
    sem = asyncio.Semaphore(64)
    async def one(it):
        im = cap(Image.open(Path(a.data_root) / it["image"])); text = it["question"] + "\n" + PROMPT_STANDARD
        async with sem:
            j = await post(sess, "/v1/chat/completions", {"model": served, "messages": [{"role": "user", "content": content(text, im)}],
                                                         "n": a.G, "max_tokens": a.max_tokens, **SAMPLING})
        outs = [(ch["message"]["content"] or "", ch["finish_reason"]) for ch in j["choices"]]
        return {"item": it, "im": im, "text": text, "outs": outs, "answers": [parse_answer(t, it["n_options"]) if f == "stop" else None for t, f in outs]}
    return await asyncio.gather(*[one(it) for it in items])

_TL = threading.local(); POOLX = ThreadPoolExecutor(max_workers=12); NFALLBACK = 0
def tproc():
    """one processor per worker thread: HF fast tokenizers keep padding state and are not thread-safe"""
    if not hasattr(_TL, "p"): _TL.p = AutoProcessor.from_pretrained(MODEL, use_fast=True); _TL.p.tokenizer.padding_side = "left"
    return _TL.p

def encode(g, sub):
    """CPU side of one micro-batch: chat text with the image, and each response's token count L (scored tokens)."""
    global NFALLBACK
    p = tproc(); tok = p.tokenizer; texts, P = [], []
    prompt = p.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": g["text"]}, {"type": "image"}]}], add_generation_prompt=True, tokenize=False)
    for i in sub:
        resp, fin = g["outs"][i]; tail = resp + ("<|im_end|>" if fin == "stop" else ""); texts.append(prompt + tail)
        nl = len(tok(tail, add_special_tokens=False)["input_ids"])
        if len(tok(prompt, add_special_tokens=False)["input_ids"]) + nl == len(tok(prompt + tail, add_special_tokens=False)["input_ids"]): P.append(-nl)
        else: NFALLBACK += 1; P.append(p(text=[prompt], images=[g["im"]], return_tensors="pt")["input_ids"].shape[1])
    enc = p(text=texts, images=[g["im"]] * len(sub), return_tensors="pt", padding=True)
    T = enc["attention_mask"].sum(1).tolist(); return g, sub, enc, [(-q if q < 0 else t - q) for t, q in zip(T, P)]

def train_step(kept, denom):
    """forward/backward of every rollout with a non-zero advantage; CPU encoding runs ahead in threads"""
    jobs = []
    for g in kept:
        idx = [i for i in range(len(g["adv"])) if g["adv"][i] != 0]
        jobs += [POOLX.submit(encode, g, idx[k: k + a.mb]) for k in range(0, len(idx), a.mb)]
    ntok = 0
    for fut in jobs:
        g, sub, enc, L = fut.result(); enc = enc.to("cuda"); K = max(L) + 1
        out = model(**enc, logits_to_keep=K, use_cache=False); loss = 0.0
        for b, i in enumerate(sub):
            if L[b] <= 0: continue
            lp = torch.log_softmax(out.logits[b, K - 1 - L[b]: K - 1].float(), -1).gather(-1, enc["input_ids"][b, -L[b]:, None]).squeeze(-1)
            loss = loss - g["adv"][i] * lp.sum() / denom; ntok += L[b]
        if torch.is_tensor(loss): loss.backward()
        del out, enc, loss
    return ntok

async def main():
    served, qptr, log = MODEL, 0, open(OUT / "log.jsonl", "a")
    async with aiohttp.ClientSession() as sess:
        for step in range(a.steps):
            t0 = time.time(); kept, sampled, allr, acc, lens, nspoil, npen = [], 0, [], [], [], 0, 0
            while len(kept) < a.q and sampled < 4 * a.q:                # draw questions until q groups carry a learning signal
                if qptr + a.q > len(POOL): random.Random(1000 + qptr).shuffle(POOL); qptr = 0; print("[pool wrap]", flush=True)
                chunk = POOL[qptr: qptr + a.q]; qptr += a.q; sampled += len(chunk)
                for g in await rollouts(sess, chunk, served):
                    gold, ans = g["item"]["gold"], g["answers"]; r = group_rewards(ans, gold)
                    acc += [x == gold for x in ans]; allr += r; lens += [len(t) for t, _ in g["outs"]]
                    nspoil += sum(x is None for x in ans); npen += sum(1 for x, y in zip(r, ans) if y is not None and x < 0)
                    if max(r) > min(r) and len(kept) < a.q: g["r"] = r; kept.append(g)
            t1 = time.time(); model.train()
            for g in kept:
                mean = sum(g["r"]) / len(g["r"]); g["adv"] = [x - mean for x in g["r"]]
                if a.arm == "edas": g["adv"] = edas_adjust(g["adv"], g["answers"], g["item"]["gold"])
            ntok = train_step(kept, a.norm * len(kept))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(set_to_none=True)
            name = f"{a.arm}_{RUNID}_s{step + 1}"; path = OUT / "adapters" / name; model.save_pretrained(path)
            await post(sess, "/v1/load_lora_adapter", {"lora_name": name, "lora_path": str(path)})
            if served != MODEL: await post(sess, "/v1/unload_lora_adapter", {"lora_name": served})
            served = name
            rec = {"step": step + 1, "kept": len(kept), "sampled": sampled, "acc": sum(acc) / len(acc), "reward": sum(allr) / len(allr),
                   "penalised_frac": npen / len(allr), "spoiled_frac": nspoil / len(allr), "resp_chars": sum(lens) / len(lens),
                   "train_tokens": ntok, "t_gen": round(t1 - t0), "t_train": round(time.time() - t1), "n_fallback": NFALLBACK}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            for old in (OUT / "adapters").iterdir():                    # keep every 20th step and the latest
                if old.name != name and not any(old.name.endswith(f"_s{k}") for k in range(20, a.steps + 1, 20)): shutil.rmtree(old, ignore_errors=True)
    model.save_pretrained(OUT / "final"); print("final adapter:", OUT / "final", flush=True)

asyncio.run(main())
