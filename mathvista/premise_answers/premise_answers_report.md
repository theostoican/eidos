# Premise-augmented re-answering of MathVista loss cases

Model: `Qwen/Qwen3.5-9B` | 10 cases | fixed: **0/10**

Each originally-wrong case is re-answered with its 8 visual premises fed back in.

## pid 7 — ❌ still wrong
**Q:** What time is shown? Answer by typing a time word, not a number. It is (_) after eight.
- ground truth: `quarter`
- original (no-premise) answer: `quarter past`
- premise-augmented answer: `quarter past`

## pid 19 — ❌ still wrong
**Q:** what is the highest amount this class measures?
- ground truth: `400`
- original (no-premise) answer: `600.0`
- premise-augmented answer: `600.0`

## pid 20 — ❌ still wrong
**Q:** Is the sum of smallest two bar is greater then the largest bar?
- ground truth: `No`
- original (no-premise) answer: `Yes`
- premise-augmented answer: `Yes`

## pid 27 — ❌ still wrong
**Q:** What is the age gap between these two people in image?
- ground truth: `11`
- original (no-premise) answer: `23.0`
- premise-augmented answer: `2.0`

## pid 37 — ❌ still wrong
**Q:** What is the limit as x approaches -1?
- ground truth: `3`
- original (no-premise) answer: `2.0`
- premise-augmented answer: `2.0`

## pid 41 — ❌ still wrong
**Q:** What is the age gap between these two people in image?
- ground truth: `7`
- original (no-premise) answer: `13.0`
- premise-augmented answer: `10.0`

## pid 42 — ❌ still wrong
**Q:** What is the age gap between these two people in image?
- ground truth: `8`
- original (no-premise) answer: `0.0`
- premise-augmented answer: `0.0`

## pid 50 — ❌ still wrong
**Q:** Based on the food web, what would likely happen if the number of large roach would decrease?
- ground truth: `The population of predatory insects would decrease.`
- original (no-premise) answer: `The population of predatory insects would increase.`
- premise-augmented answer: `The population of predatory insects would increase.`

## pid 53 — ❌ still wrong
**Q:** What is the age gap between these two people in image?
- ground truth: `1`
- original (no-premise) answer: `8.0`
- premise-augmented answer: `0.0`

## pid 60 — ❌ still wrong
**Q:** What is the age gap between these two people in image?
- ground truth: `22`
- original (no-premise) answer: `0.0`
- premise-augmented answer: `30.0`
