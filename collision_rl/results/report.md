# Results (16 samples per question; 95% question-bootstrap CIs; arms pooled over the seeds present)

## mmstar (n = 1500 questions)

| run | pass@1 | maj@16 | voting gain | error collision | parsed-only collision | unparsed |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.5104 | 0.5787 | 0.0683 | 0.2132 | 0.2461 | 0.1114 |
| collision | 0.5280 | 0.5627 | 0.0346 | 0.2527 | 0.2564 | 0.0086 |
| collision_s1 | 0.5154 | 0.5393 | 0.0239 | 0.2580 | 0.2597 | 0.0037 |
| csr | 0.5201 | 0.5524 | 0.0324 | 0.2552 | 0.2565 | 0.0030 |
| csr_s1 | 0.5168 | 0.5394 | 0.0227 | 0.2591 | 0.2603 | 0.0023 |
| edas | 0.4660 | 0.4930 | 0.0270 | 0.2634 | 0.2649 | 0.0033 |
| edas_s1 | 0.4818 | 0.5074 | 0.0256 | 0.2610 | 0.2621 | 0.0028 |
| grpo | 0.5102 | 0.5411 | 0.0309 | 0.2633 | 0.2647 | 0.0035 |
| grpo_s1 | 0.5076 | 0.5419 | 0.0343 | 0.2521 | 0.2527 | 0.0025 |

- collision (2 seeds) - grpo (2): maj@16 +0.95 [-0.23, +2.11]; pass@1 +1.28 [+0.74, +1.83]; voting gain -0.33 [-1.25, +0.61]; parsed-only collision -0.07 [-0.62, +0.45]
- collision (2 seeds) - csr (2): maj@16 +0.51 [-0.51, +1.49]; pass@1 +0.33 [-0.14, +0.79]; voting gain +0.18 [-0.63, +0.99]; parsed-only collision -0.03 [-0.51, +0.43]
- collision (2 seeds) - edas (2): maj@16 +5.08 [+3.57, +6.62]; pass@1 +4.78 [+3.96, +5.60]; voting gain +0.30 [-0.83, +1.36]; parsed-only collision -0.54 [-1.31, +0.24]
- collision (2 seeds) - base (1): maj@16 -2.77 [-4.43, -1.13]; pass@1 +1.14 [+0.10, +2.17]; voting gain -3.90 [-5.24, -2.64]; parsed-only collision +1.14 [+0.19, +2.08]
- grpo (2 seeds) - base (1): maj@16 -3.72 [-5.62, -1.85]; pass@1 -0.15 [-1.29, +0.96]; voting gain -3.57 [-4.90, -2.23]; parsed-only collision +1.23 [+0.21, +2.31]

## mmmu (n = 1591 questions)

| run | pass@1 | maj@16 | voting gain | error collision | parsed-only collision | unparsed |
| --- | --- | --- | --- | --- | --- | --- |
| base | 0.2512 | 0.3283 | 0.0771 | 0.1624 | 0.2940 | 0.3927 |
| collision | 0.3254 | 0.3837 | 0.0583 | 0.2528 | 0.2609 | 0.0293 |
| collision_s1 | 0.3163 | 0.3678 | 0.0515 | 0.2544 | 0.2571 | 0.0086 |
| csr | 0.3175 | 0.3730 | 0.0555 | 0.2571 | 0.2590 | 0.0081 |
| csr_s1 | 0.3137 | 0.3620 | 0.0484 | 0.2641 | 0.2653 | 0.0037 |
| edas | 0.2768 | 0.3124 | 0.0356 | 0.2406 | 0.2416 | 0.0030 |
| edas_s1 | 0.2835 | 0.3224 | 0.0388 | 0.2512 | 0.2521 | 0.0038 |
| grpo | 0.3093 | 0.3498 | 0.0405 | 0.2617 | 0.2624 | 0.0025 |
| grpo_s1 | 0.2974 | 0.3419 | 0.0445 | 0.2550 | 0.2556 | 0.0018 |

- collision (2 seeds) - grpo (2): maj@16 +2.99 [+1.85, +4.16]; pass@1 +1.75 [+1.22, +2.28]; voting gain +1.24 [+0.31, +2.21]; parsed-only collision -0.00 [-0.48, +0.50]
- collision (2 seeds) - csr (2): maj@16 +0.82 [-0.36, +1.96]; pass@1 +0.52 [+0.03, +1.00]; voting gain +0.30 [-0.62, +1.26]; parsed-only collision -0.31 [-0.78, +0.15]
- collision (2 seeds) - edas (2): maj@16 +5.83 [+4.30, +7.31]; pass@1 +4.07 [+3.35, +4.82]; voting gain +1.77 [+0.62, +2.88]; parsed-only collision +1.21 [+0.58, +1.88]
- collision (2 seeds) - base (1): maj@16 +4.75 [+2.92, +6.60]; pass@1 +6.96 [+5.89, +8.10]; voting gain -2.22 [-3.55, -0.92]; parsed-only collision -1.27 [-2.55, -0.01]
- grpo (2 seeds) - base (1): maj@16 +1.76 [-0.27, +3.75]; pass@1 +5.21 [+3.98, +6.37]; voting gain -3.46 [-4.85, -2.06]; parsed-only collision -1.18 [-2.49, +0.09]

