#!/usr/bin/env python
"""Checks the rewards against the implementation behind the reported runs, and the spoiled-ballot floor.  python test_rewards.py"""
import collections, random
from common import collision, csr, dfuser, floor_spoiled, parse_answer

def collision_as_run(answers, gold, lam=1.0):            # arm rahp_noref of the reported runs: rahp with an empty reference
    G = len(answers); cnt = collections.Counter(answers); r = [1.0 if a == gold else 0.0 for a in answers]
    return [x - (lam * max(0.0, cnt[a] / G - 0.0) if (a is not None and a != gold) else 0.0) for x, a in zip(r, answers)]

def csr_as_run(answers, gold, lam=0.5):
    G = len(answers); c = [1.0 if x == gold else 0.0 for x in answers]; r = list(c)
    for j in range(G):
        if answers[j] is None: continue
        r[j] += -lam * sum(abs(float(answers[j] == answers[i]) - c[j]) for i in range(G) if i != j) / (G - 1)
    return r

def dfuser_as_run(answers, gold, rng):
    r = [1.0 if a == gold else 0.0 for a in answers]; wrong = collections.Counter(a for a in answers if a is not None and a != gold)
    if wrong:
        m = max(wrong.values()); zerr = rng.choice(sorted(a for a, n in wrong.items() if n == m))
        r = [x + (0.5 if (a is not None and a != gold and a != zerr) else 0.0) for x, a in zip(r, answers)]
    return r

def gap(r, g, gold):        # worst wrong answer minus best spoiled ballot (inf when the group lacks either)
    wrong = [x for x, a in zip(r, g) if a is not None and a != gold]; spoiled = [x for x, a in zip(r, g) if a is None]
    return min(wrong) - max(spoiled) if wrong and spoiled else float("inf")

rng = random.Random(0); gc = gs = float("inf")
for _ in range(20000):
    g = [rng.choice(["A", "B", "C", "D", None]) for _ in range(8)]; gold = rng.choice("ABCD"); s = rng.random()
    assert collision(g, gold) == collision_as_run(g, gold)
    assert csr(g, gold) == csr_as_run(g, gold)
    assert dfuser(g, gold, random.Random(s)) == dfuser_as_run(g, gold, random.Random(s))
    gc = min(gc, gap(floor_spoiled(collision(g, gold, 1.0), g, -1.0), g, gold))
    gs = min(gs, gap(floor_spoiled(csr(g, gold, 0.5, parsed_only=True), g, -0.5), g, gold))
# collision: a spoiled ballot is strictly below every wrong answer (a wrong answer shares the group with it, so its share
# is at most 7/8). csr: a wrong answer every parsed peer agrees with reaches -lambda too, a tie: not answering still never pays.
assert gc > 0 and gs >= 0, (gc, gs)
assert parse_answer("so the answer is B.\nAnswer: C", 4) == "C" and parse_answer("Answer: E", 4) is None and parse_answer("I pick (B)", 4) == "B"
print(f"ok: the rewards match the reported runs on 20,000 random groups; with the floor, the worst wrong answer beats a "
      f"spoiled ballot by >= {gc:.3f} (collision) and ties it at worst ({gs:.3f}, csr)")
