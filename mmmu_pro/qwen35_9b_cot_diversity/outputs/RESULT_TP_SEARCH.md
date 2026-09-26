# Can temperature / top_p alone beat the Qwen3.5-9B model-card defaults at maj@8?

Target: >= +1 pp maj@8 over the card defaults (thinking mode: T=1.0, top_p=0.95, top_k=20,
presence_penalty=1.5) on MMMU-Pro standard (10 options), Qwen3.5-9B, 8 samples/question,
official CoT prompt, 40,960-token budget, ballot rule (spoiled ballots count as wrong).
All comparisons are PAIRED on the same questions (frozen set of 1,407, or the first n
completed by an arm, which the client fills in a seeded shuffled order).

## Answer: no. The card is at the optimum of the T x p space to within ~0.5 pp.

Best single point: T=1.2, top_k=-1, presence 1.5, top_p=0.8 -> +0.46 pp maj@8 on all 1,729 questions (p=0.42), -0.02 pp maj@1.

### With the card's sampler held fixed (top_k=20, presence 1.5), varying only T and p

| T | p | n | maj@1 vs card | maj@8 vs card | p (maj@8) |
|---|---|---|---|---|---|
| 1.0 | 0.95 | -- | card | card | -- |
| 1.1 | 0.95 | 414 | -0.24 pp | -0.24 pp | 0.81 |
| 1.2 | 0.95 | 1407 | -0.50 pp | **+0.28 pp** | 0.67 |
| 1.2 | 1.00 | 415 | -5.09 pp | -2.89 pp | 0.04 |
| 1.4 | 0.95 | 413 | -5.66 pp | -2.18 pp | 0.13 |

Widening the nucleus from 0.95 to 1.0 costs ~3 pp at T=1.2; raising T past 1.2 costs 2-3 pp.

### Without top_k (presence 1.5), sweeping p at T=1.2 -- and the T=1.6 neutral sweep

| arm | n | maj@8 vs card |
|---|---|---|
| T1.2 top_k=-1, p 0.8 | 1407 frozen / 322 holdout / 1729 all | **+0.85 pp (p=0.19) / -1.25 pp (p=0.32) / +0.46 pp (p=0.42)** -- the best point of the search; the frozen-set edge did not confirm on the holdout |
| T1.2 top_k=-1, p in 0.4..0.7 | ~280-350 each | -1.8 to +0.6 pp (plateau, all within noise) |
| T1.2 top_k=-1, p 0.9 | 280 | -2.9 pp |
| T1.2 top_k=-1, p 0.95 | 1407 | -17.2 pp |
| T1.6 top_k=-1 presence 0, p 0.2..0.6 | 1407 each | -1.1 to -1.9 pp |
| T1.6 top_k=-1 presence 0, p 0.7 | 1407 | -10.1 pp |

### Mixed-temperature votes (chains at different T/p in one 8-ballot vote)
Split-half on the frozen set: best mix selected on half A (+1.0 pp) gives +0.43 pp on half B
(p=0.59); the a-priori 4 card + 4 T1.2 mix gives +0.14 pp on all 1,407 (RESULT_MIXED_BALLOTS.md).

### Reading
The 8-way vote gains ~+5 pp over a single sample at EVERY setting, hot or cold: extra sampling
diversity does not translate into more independent votes -- it translates into more wrong
single-sample answers, which the vote then has to absorb. The card's two safety knobs
(top_k=20, presence 1.5) are what keep spoilage near zero; removing either is the largest
effect in the table, and it is negative.

Detail tables: RESULT_ALL_VS_CARD.md (every arm), RESULT_HOLDOUT_VS_CARD.md (322 untouched
questions, once arms exist there), RESULT_MIXED_BALLOTS.md.
