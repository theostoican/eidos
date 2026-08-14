# top_p sweep -- pre-registered analysis

Sources: 9 file(s). Config profile(s) in data: `['neutral']`.
Temperature arm: 1.0 | grid: [0.9, 0.925, 0.95, 0.975, 0.99, 1.0]

## Spoil rates per top_p

| top_p | generations | truncated | unparseable | spoiled % |
|---|---|---|---|---|
| 0.9 | 5520 | 110 | 0 | 1.99% |
| 0.925 | 5520 | 81 | 0 | 1.47% |
| 0.95 | 5520 | 60 | 0 | 1.09% |
| 0.975 | 5520 | 40 | 0 | 0.72% |
| 0.99 | 5520 | 29 | 0 | 0.53% |
| 1.0 | 5520 | 23 | 0 | 0.42% |

## Results

### n = 345 questions | grid = [0.9, 0.925, 0.95, 0.975, 0.99, 1.0]

| k | p=0.9 | p=0.925 | p=0.95 | p=0.975 | p=0.99 | p=1.0 | argmax | F | p(omni) | P(shape) | P(joint) | quad a | best raw/Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.7266 | 0.7293 | 0.7293 | 0.7237 | 0.7366 | 0.7299 | 0.99 | 1.61 | 0.155 | 0.870 | 0.342 | +0.5176 | 0.0093 / 0.0464 |
| 16 | 0.7710 | 0.7652 | 0.7681 | 0.7507 | 0.7594 | 0.7710 | 0.9 | 1.04 | 0.394 | 0.325 | 0.142 | +3.4381 | 0.0194 / 0.0971 |

optimum vs k: k=1->0.99, k=16->0.9 -- pre-declared non-decreasing in k: **VIOLATED**
