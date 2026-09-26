# FINAL: T=1.2, top_k=-1, presence 1.5, top_p=0.7 (and 0.8) vs model-card defaults, ALL 1,729 questions, paired
p=0.7: maj@8 .7415 vs .7351 = +0.64 pp; maj@1 .7068 vs .7053 = +0.15 pp.
p=0.8: maj@8 +0.46 pp; maj@1 -0.02 pp.
Selection set (1,407): p0.7 +0.92 / p0.8 +0.85 at maj@8; holdout (322): p0.7 -0.62 / p0.8 -1.24.

# All arms vs card baseline (T=1.0, top_p .95, top_k 20, presence 1.5), paired, HOLDOUT (outside frozen set)
| arm | n | maj@1 | card | d | p | maj@8 | card | d | p |
| T1.2 top_k=-1 presence1.5 p0.7 | 322 | 0.7038 | 0.7015 | +0.23 pp | 0.729 | 0.7391 | 0.7453 | -0.62 pp | 0.638 |
| T1.2 top_k=-1 presence1.5 p0.8 | 322 | 0.7112 | 0.7015 | +0.97 pp | 0.177 | 0.7329 | 0.7453 | -1.24 pp | 0.318 |
# All arms vs card baseline (T=1.0, top_p .95, top_k 20, presence 1.5), paired, ALL questions (frozen + holdout)
| arm | n | maj@1 | card | d | p | maj@8 | card | d | p |
| T1.2 top_k=-1 presence1.5 p0.7 | 1729 | 0.7068 | 0.7053 | +0.15 pp | 0.632 | 0.7415 | 0.7351 | +0.64 pp | 0.292 |
| T1.2 top_k=-1 presence1.5 p0.75 | 212 | 0.7748 | 0.7712 | +0.35 pp | 0.650 | 0.7877 | 0.7830 | +0.47 pp | 0.782 |
| T1.2 top_k=-1 presence1.5 p0.8 | 1729 | 0.7051 | 0.7053 | -0.02 pp | 0.946 | 0.7397 | 0.7351 | +0.46 pp | 0.424 |
| T1.2 top_k=-1 presence1.5 p0.85 | 1407 | 0.6933 | 0.7062 | -1.29 pp | 0.000 | 0.7349 | 0.7328 | +0.21 pp | 0.758 |
