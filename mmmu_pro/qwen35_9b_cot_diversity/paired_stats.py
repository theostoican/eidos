#!/usr/bin/env python
"""Paired significance for the majority-vote shape.

The SAME questions are run at every top_p, so top_p comparisons are PAIRED: between-question
difficulty is common and cancels. The independent binomial SE (sqrt(p(1-p)/n)) is therefore
the WRONG error bar for shape detection -- it is dominated by between-question variance that
the pairing removes. At n=87 questions the independent SE is ~0.046 while the effect being
hunted is ~0.040, which would make a real effect look like noise.

Provides:
  paired_diff(a, b) -> mean per-question difference in majority-correctness, its paired SE,
                       t-stat, and McNemar discordant counts.
Only questions present in BOTH conditions are used.
"""
import math
import numpy as np


def paired_diff(maj_a, maj_b):
    """maj_a, maj_b: dict qid -> 0/1 majority-correct. Returns stats for (a - b)."""
    shared = sorted(set(maj_a) & set(maj_b))
    if len(shared) < 5:
        return None
    d = np.array([maj_a[q] - maj_b[q] for q in shared], float)
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    # McNemar discordant pairs
    b = int(sum(1 for x in d if x > 0))   # a correct, b wrong
    c = int(sum(1 for x in d if x < 0))   # b correct, a wrong
    t = mean / se if se > 0 else 0.0
    return {"n_shared": n, "mean_diff": mean, "paired_se": se, "t": t,
            "b": b, "c": c, "n_discordant": b + c}


def fmt(st, label_a, label_b):
    if st is None:
        return f"  {label_a} vs {label_b}: too few shared questions"
    sig = "significant" if abs(st["t"]) >= 2 else "NOT significant"
    return (f"  {label_a} vs {label_b}: diff={st['mean_diff']:+.4f} "
            f"paired_SE={st['paired_se']:.4f} t={st['t']:+.2f} ({sig}) "
            f"| discordant {st['b']}/{st['c']} of {st['n_shared']}")
