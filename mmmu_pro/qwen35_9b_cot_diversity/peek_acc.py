#!/usr/bin/env python
"""Incremental peek for the temperature x top_p sweep. The RESULT of interest is an
INVERTED-U in per-sample answer accuracy WITH RESPECT TO top_p, checked at EACH temperature
(temperature is the second axis, to find the regime where the p-hump appears, if any).

Reads current cot_gen.shard*.jsonl, groups by (temperature, top_p), and for every temperature
whose FULL top_p sweep is present, prints accuracy vs top_p + an inverted-U-vs-p verdict, so
the run can be pruned early if no temperature produces a p-hump.
"""
import json, glob, re, collections, math, statistics

LETTERS = [chr(ord("A") + i) for i in range(26)]
ANS_RE = re.compile(r"Answer:\s*\(?\s*([A-J])\b", re.IGNORECASE)

def parse_answer(text, n):
    valid = set(LETTERS[:n])
    after = text.split("</think>")[-1] if "</think>" in text else text
    for c in reversed(ANS_RE.findall(after) or ANS_RE.findall(text)):
        if c.upper() in valid: return c.upper()
    for ch in reversed(re.findall(r"\b([A-J])\b", after)):
        if ch in valid: return ch
    return None

rows = []
for f in glob.glob("outputs/cot_gen.shard*.jsonl"):
    for l in open(f):
        try: rows.append(json.loads(l))
        except Exception: pass
rows = [r for r in rows if r.get("finish_reason") == "stop"]

# (temp, top_p) -> {qid: [correct...]}
grid = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    T = r.get("temperature"); pred = parse_answer(r["text"], r.get("n_options", 10))
    if T is None or pred is None: continue
    grid[(T, r["top_p"])][r["id"]].append(int(pred == r["gold"]))

def cellstat(T, p):
    pc = [sum(v)/len(v) for v in grid[(T, p)].values()]
    if not pc: return None
    return statistics.mean(pc), (statistics.pstdev(pc)/math.sqrt(len(pc)) if len(pc) > 1 else 0), len(pc)

temps = sorted({T for (T, p) in grid})
for T in temps:
    ps = sorted({p for (t2, p) in grid if t2 == T})
    print(f"=== temperature={T} : per-sample accuracy vs top_p  (INVERTED-U in p?) ===")
    curve = []
    for p in ps:
        c = cellstat(T, p)
        if c is None: continue
        m, se, nq = c
        tag = " PARTIAL" if nq < 75 else ""
        print(f"   top_p={p}: {m:.3f} +/- {se:.3f} (nQ={nq}{tag})")
        if nq >= 75: curve.append((p, m, se))
    if len(curve) >= 4:   # need most of the p-sweep to judge the p-shape
        P = [c[0] for c in curve]; acc = [c[1] for c in curve]; se = [c[2] for c in curve]
        imax = max(range(len(acc)), key=lambda i: acc[i])
        interior = 0 < imax < len(acc) - 1
        amp = acc[imax] - max(acc[0], acc[-1]); noise = 2 * max(se)
        print(f"   -> peak at top_p={P[imax]} interior={interior} amp_above_ends={amp:+.3f} vs 2SE={noise:.3f}")
        if interior and amp > noise:
            print("   VERDICT: INVERTED-U in p at this temperature (interior peak above noise) -> KEEP GOING")
        else:
            print("   VERDICT: no inverted-U in p here (peak at endpoint or within noise)")
    else:
        print(f"   (only {len(curve)} full top_p point(s) at this temp — need >=4 to judge p-shape)")
    print()
