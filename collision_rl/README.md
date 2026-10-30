# Collision RL

RL fine-tuning for a vision-language model that penalises wrong answers in proportion to how many rollouts share them,
so the model's errors stop agreeing with each other and majority voting over samples has more to gain.
Single GPU (RTX 4090, 24 GB), Qwen3-VL-2B-Instruct, LoRA.

## Method

GRPO with a modified reward. For each question we sample a group of G = 8 rollouts with answers a_1..a_G, gold y:

```
r_i = 1                          if a_i = y
r_i = -λ · n(a_i) / G            if a_i is a parsed wrong answer, n(a_i) = rollouts in the group giving a_i (i included)
r_i = 0                          if a_i is unparsed (no answer, or truncated at the token limit)
```

with λ = 1. A wrong answer given by one rollout costs 1/8; a wrong answer the whole group agrees on costs 1.
Example (gold A): answers `A A B B B C – –` get rewards `1 1 -3/8 -3/8 -3/8 -1/8 0 0`.
The shared wrong answers are the ones that win the majority vote, so the penalty acts on the errors that hurt maj@k.

Everything else is plain GRPO:
- Dr. GRPO advantages: reward minus the group mean, without dividing by the standard deviation.
- The loss is normalised by a constant 1,024 tokens per group.
- One on-policy update per step, no KL term.
- Groups whose rewards are all equal are skipped. Questions are drawn until 16 groups carry signal, at most 64 per step.

The reward functions are in [common.py](common.py), the trainer in [train.py](train.py).

## Results

Qwen3-VL-2B-Instruct, 60 steps, 2 seeds per arm. Each evaluation draws 16 samples per question. Full table:
[results/report.md](results/report.md), generated from the per-question files in [results/](results/).

| run (seed 0 / seed 1) | MMMU-Pro pass@1 | MMMU-Pro maj@16 | MMStar pass@1 | MMStar maj@16 |
| --- | --- | --- | --- | --- |
| base model | 0.251 | 0.328 | 0.510 | 0.579 |
| GRPO | 0.309 / 0.297 | 0.350 / 0.342 | 0.510 / 0.508 | 0.541 / 0.542 |
| **collision RL** | **0.325 / 0.316** | **0.384 / 0.368** | **0.528 / 0.515** | **0.563 / 0.539** |
| CSR (2605.15588), reimplemented | 0.318 / 0.314 | 0.373 / 0.362 | 0.520 / 0.517 | 0.552 / 0.539 |
| EDAS (2605.17333), reimplemented | 0.277 / 0.284 | 0.312 / 0.322 | 0.466 / 0.482 | 0.493 / 0.507 |

Differences in percentage points, pooled over both seeds, with 95% question-bootstrap CIs.
Voting gain = maj@16 − pass@1.

| contrast | benchmark | maj@16 | pass@1 | voting gain |
| --- | --- | --- | --- | --- |
| collision − GRPO | MMMU-Pro | **+2.99 [+1.85, +4.16]** | +1.75 [+1.22, +2.28] | +1.24 [+0.31, +2.21] |
| collision − GRPO | MMStar | +0.95 [−0.23, +2.11] | +1.28 [+0.74, +1.83] | −0.33 [−1.25, +0.61] |
| collision − CSR | MMMU-Pro | +0.82 [−0.36, +1.96] | +0.52 [+0.03, +1.00] | +0.30 [−0.62, +1.26] |
| collision − CSR | MMStar | +0.51 [−0.51, +1.49] | +0.33 [−0.14, +0.79] | +0.18 [−0.63, +0.99] |
| collision − EDAS | MMMU-Pro | +5.83 [+4.30, +7.31] | +4.07 [+3.35, +4.82] | +1.77 [+0.62, +2.88] |
| collision − EDAS | MMStar | +5.08 [+3.57, +6.62] | +4.78 [+3.96, +5.60] | +0.30 [−0.83, +1.36] |
| collision − base | MMMU-Pro | +4.75 [+2.92, +6.60] | +6.96 [+5.89, +8.10] | −2.22 [−3.55, −0.92] |
| collision − base | MMStar | −2.77 [−4.43, −1.13] | +1.14 [+0.10, +2.17] | −3.90 [−5.24, −2.64] |

What this shows:
- **MMMU-Pro:** collision RL beats GRPO by 3.0 points on maj@16. About 1.75 points of that is pass@1; the other 1.24 is a
  larger voting gain.
- **MMStar:** collision RL improves pass@1 over GRPO, but maj@16 and the voting gain do not move significantly.
- **All RL arms lose voting gain against the base model**, and on MMStar their maj@16 ends below the base model.
  Collision RL recovers part of that loss, not all of it.
- **Against CSR:** collision RL is ahead on every number, but only the MMMU-Pro pass@1 difference clears zero, so on this
  evidence the two are tied.
- **Mechanism not confirmed:** error collision among parsed answers is unchanged against GRPO (−0.00 pp on MMMU-Pro).
  The voting gain therefore does not show up as less agreement among the wrong answers that were parsed. See
  [Limitations](#limitations) for the spoiled-ballot confound.

Earlier versions of this code used other names for these runs: `iid` is `grpo`; `rahp_noref` and `coll` are `collision`.

## Training data

**[ViRL39K](https://huggingface.co/datasets/TIGER-Lab/ViRL39K)** (TIGER-Lab, MIT licence): 38,870 verifiable image QA
pairs. [data.py](data.py) keeps the questions that meet all of these:
- exactly one image;
- an answer of the form `\boxed{X}` with X a single letter;
- options labelled consecutively from A (`A.`, `(A)`, …), at least two of them, with the gold letter among them.

It removes the `<image>` placeholder and answer-format lines that conflict with the chain-of-thought suffix ("answer
with the option's letter directly", "Hint: …"). This leaves **9,975 questions**, written to
[data/pool.jsonl](data/pool.jsonl) (5.4 MB). The file is committed; the images are not.

| category (ViRL39K labels) | questions |
| --- | --- |
| (GradeSchool) Science | 3,060 |
| (GradeSchool) Geometric | 2,621 |
| (GradeSchool) Non-Geo Math | 1,690 |
| Tables/Diagrams/Charts | 846 |
| Social Science | 585 |
| Spatial Reasoning | 545 |
| Broader STEM Topics | 513 |
| Commonsense | 115 |

Number of options: 4 → 7,860; 2 → 753; 3 → 748; 5 → 601; 6 → 11; 7 → 2.

How the pool is split and sampled:
- The pool is shuffled once with `random.Random(0)`.
- Positions 7600–8399 (800 questions) are an in-distribution held-out set (`split: heldout`). Training never draws them;
  `evaluate.py --set virl` scores them.
- The other 9,175 questions are the training pool. Seed 0 walks it in file order; other seeds shuffle it first.
- One 60-step run consumes about 2,000 questions (1,936–2,032 in the four reported runs), each seen once.

Each line holds `qid`, `question`, `gold`, `n_options`, `image` (a path relative to the dataset root), `category`, `pass32`
(ViRL39K's pass rate for a 32B model; unused) and `split`.

## Evaluation data

| set | source | questions | options |
| --- | --- | --- | --- |
| MMStar | [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), `val` | 1,500 | 4 |
| MMMU-Pro | [MMMU/MMMU_Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro), `standard (10 options)`, `test` | 1,591 | up to 10 |
| ViRL held-out | data/pool.jsonl, `split: heldout` | 800 | 2–7 |

For MMMU-Pro we keep single-image questions only, dropping 1 question with 12 options. Both evaluation datasets download
automatically from the Hugging Face Hub.

Prompting and sampling, identical in training and evaluation:
- The prompt is the question text, then the image, then the official MMMU-Pro chain-of-thought suffix ("… The last line
  of your response should be of the following format: 'Answer: $LETTER' …").
- Images are downscaled to at most 512×512 pixels.
- Qwen3-VL's recommended sampling: temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5.
- At most 1,024 new tokens.

Scoring:
- An answer is the last `Answer: X` line, else the last bare option letter.
- A response that is truncated, or has no valid letter, is unparsed and counts as wrong.
- maj@16 votes over the parsed answers and splits ties. A question with no parsed answer scores 0.
- **Error collision** is the probability that two different samples give the same wrong answer, computed over all votes
  and over parsed votes only.

## Training setup

| setting | value |
| --- | --- |
| model | Qwen/Qwen3-VL-2B-Instruct (override with `VLM_MODEL`) |
| adapter | LoRA r = 16, α = 32, no dropout, on the language model's q/k/v/o/gate/up/down projections; vision encoder frozen |
| optimiser | AdamW, lr 2e-5, no weight decay, gradient clip 1.0 |
| steps | 60, with 16 groups × 8 rollouts per step |
| rollouts | vLLM server on the same GPU (40% of memory); it hot-loads the new adapter after every step |
| update | Hugging Face Transformers + PEFT, bf16, gradient checkpointing, micro-batch 2 |
| time | about 50 minutes per run on one RTX 4090 |

## Setup

Python 3.12 and CUDA 13 (the torch build in [requirements.txt](requirements.txt)). Other CUDA versions should work with
the matching torch and vLLM wheels.

```bash
cd collision_rl
pip install -r requirements.txt
huggingface-cli download TIGER-Lab/ViRL39K --repo-type dataset --local-dir data/virl39k
(cd data/virl39k && unzip -q images.zip)          # -> data/virl39k/images/...
python data.py                                    # optional: rebuilds data/pool.jsonl (the committed file is identical)
python test_rewards.py                            # checks the rewards
```

If the dataset already sits elsewhere, set `VIRL39K_ROOT=/path/to/virl39k` (the directory with `39Krelease.parquet` and
`images/`) instead of downloading it again.

## Reproduce

The whole experiment runs in about 7 hours on one 24 GB GPU: four training runs, then evaluation of the base model and
four adapters on MMStar and MMMU-Pro, then the report. Finished runs and finished questions are skipped on a re-run.

```bash
./run.sh
```

Or step by step:

```bash
./serve.sh train                                            # vLLM on 127.0.0.1:8100, 40% of GPU memory
python train.py --arm collision --seed 0 --out runs/collision
python train.py --arm grpo      --seed 0 --out runs/grpo
./serve.sh eval                                             # restart with 90% of GPU memory
python evaluate.py run --tag base      --set mmmu
python evaluate.py run --tag collision --set mmmu --adapter runs/collision/final
python evaluate.py run --tag grpo      --set mmmu --adapter runs/grpo/final
./serve.sh stop
python evaluate.py report                                   # -> results/report.md
```

Each training run writes:
- `runs/<name>/config.json`: every argument;
- `runs/<name>/log.jsonl`, one line per step: accuracy, mean reward, share of penalised and of spoiled rollouts,
  response length, timings;
- adapters at steps 20, 40 and 60, and `runs/<name>/final`.

`evaluate.py report` pools seeds by name: `collision` and `collision_s1` both count as `collision`.

### Options

| flag | default | meaning |
| --- | --- | --- |
| `--arm` | – | `grpo`, `collision`, `csr`, `edas` or `dfuser` |
| `--coll-lam` | 1.0 | λ in the collision penalty |
| `--spoiled` | `zero` | reward of an unparsed rollout in `collision` and `csr`: `zero` as in the reported runs, or `min` = −λ (see Limitations) |
| `--csr-lam` | 0.5 | CSR's agreement weight |
| `--steps`, `--q`, `--G` | 60, 16, 8 | steps, groups per step, rollouts per group |
| `--lr`, `--rank`, `--max-tokens` | 2e-5, 16, 1024 | |
| `--mb` | 2 | training micro-batch; use 1 for larger models |

## Files

| file | purpose |
| --- | --- |
| [common.py](common.py) | model name, sampling, prompt, image handling, answer parsing, the reward functions |
| [data.py](data.py) | ViRL39K → data/pool.jsonl |
| [train.py](train.py) | GRPO trainer and the five arms |
| [evaluate.py](evaluate.py) | sampling (`run`) and metrics with bootstrap CIs (`report`) |
| [serve.sh](serve.sh) | start and stop the vLLM server (`train`, `eval`, `stop`) |
| [run.sh](run.sh) | the full experiment |
| [test_rewards.py](test_rewards.py) | checks the rewards against the code behind the reported runs, plus the `--spoiled min` floor |
| [results/](results/) | per-question answers (16 per question, `null` = unparsed) for every reported run, and report.md |

## Baselines

- **GRPO:** correctness reward only.
- **CSR** ([2605.15588](https://arxiv.org/abs/2605.15588)): `R_j = c_j − λ · mean_{i≠j} |1[a_j = a_i] − c_j|`.
  A correct rollout pays for each peer that disagrees with it, a wrong rollout for each peer that agrees with it. Our
  version judges agreement by exact match and uses a constant λ = 0.5.
- **EDAS** ([2605.17333](https://arxiv.org/abs/2605.17333)): reshapes the advantages of wrong rollouts by how rare their
  answer is among the group's wrong answers. We follow the rule in the official code (α 0.4, β 0.2, κ 2).
- **D-FUSEr** (Baek et al., ICML 2026): +1/2 for a wrong answer that differs from the most common wrong answer.
  - The `dfuser` arm estimates that answer from the group; the paper estimates it offline. It also runs with this
    trainer's settings, not the paper's (lr 1e-6, group size 16).
  - It was trained (2 seeds) but not evaluated on MMStar or MMMU-Pro, so it is not in the results.

The CSR and D-FUSEr rewards are our own reimplementations; for EDAS only the adjustment rule comes from the official code.

## Limitations

- **One model.** All results are for Qwen3-VL-2B-Instruct. A weaker variant (leave-one-out share, λ = 0.5, different
  prompts) did not help on InternVL; the variant here has not been run on a second model.
- **The two benchmarks disagree.** The voting gain appears on MMMU-Pro only; on MMStar collision RL improves pass@1
  alone.
- **Tied with CSR.** Collision RL is ahead of CSR, but within the CIs on both benchmarks.
- **Spoiled-ballot confound.**
  - An unparsed rollout gets 0, while a shared wrong answer gets a negative reward, so in the reported runs not
    answering beats answering wrong.
  - Collision RL leaves more answers unparsed than GRPO (seed 0): 2.9% vs 0.25% on MMMU-Pro, 0.9% vs 0.35% on MMStar.
  - pass@1 counts these as wrong, so the pass@1 gain is not inflated. maj@16 drops them from the vote, though, so part
    of the voting gain could come from not answering hard questions.
  - `--spoiled min` removes the incentive: an unparsed rollout then scores −λ, never above any wrong answer
    ([test_rewards.py](test_rewards.py) checks this). It has not been run at full scale yet.
- **Truncation.** At 1,024 new tokens the base model leaves 39% of MMMU-Pro answers unparsed, and 48% on the ViRL
  held-out set. Part of every arm's gain over the base model is learning to finish within the limit.
- **Seeds.** vLLM sampling is not seeded, so a re-run reproduces the numbers only up to sampling noise (compare the two
  seeds above).
- **ViRL held-out results are not reported.** In the earlier code, seed 1's training stream overlapped about 20% of the
  held-out questions. This code excludes the held-out set from training for every seed, but the held-out evaluation has
  not been re-run. MMStar and MMMU-Pro were never in training data, so those results are unaffected.
