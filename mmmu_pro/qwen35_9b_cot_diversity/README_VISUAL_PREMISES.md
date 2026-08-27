# Do the visual premises get better with `top_p`? (T=1.0)

**Question.** The `top_p` sweep at T=1.0 shows answer accuracy rising +3.75 pp from 0.5 to 0.99
(`RESULT_T10.md`). Is that because the model *sees* the image better at higher `top_p`?

**Answer. No.** Perception is flat across the axis while diversity rises. Whatever produces the
accuracy gain sits downstream of seeing.

| quantity | `top_p` 0.5 → 1.0 | test |
|---|---|---|
| **visual-premise soundness** | 0.9409 → 0.9292 (−1.2%) | RM-ANOVA F=1.63, **p=0.10** |
| **premise diversity** (Vendi, count-matched) | 3.382 → 3.730 (+10.3%) | F=42.5, **p=8e-71** |
| **answer accuracy** (maj@1, ballot rule) | 0.677 → 0.713 (+5.3%) | F=6.43, **p=4e-09** |

Across the grid the soundness and accuracy curves **anticorrelate** (r=−0.51); within a question
the correlation is −0.03 over 158 questions. Diversity rises; correctness of the reads does not.

`Qwen/Qwen3.5-9B` on MMMU-Pro `standard (10 options)`, T=1.0, 10 `top_p` levels, 8 of the 16
samples per cell: **21,743 traces, 187,784 visual premises judged**, 303 questions paired at
every `top_p`.

---

## 1. Method

Three stages, each resumable, one sample "layer" at a time so an interrupted run always leaves a
complete balanced grid:

1. **Extract** (`vp_extract.py`) — Qwen3.5-9B, non-thinking, temp 0, pulls *visual premises* from
   each `<think>` trace: atomic claims about what the model saw ("the far-left inductor is
   labelled j20"), never arithmetic, inference or option elimination. The generator does the
   extracting because this is text-to-text rewriting, not judgement — it never sees the image.
2. **Judge** (`vp_judge.py`) — InternVL3-8B-AWQ checks each premise against the real image,
   **with the gold answer withheld**. A different model family from the generator, so not a
   self-judge. `UNVERIFIABLE` (a derivation that slipped through extraction) is excluded from the
   denominator; its rate is reported per `top_p`.
3. **Analyse** (`vp_result.py`, `vp_diversity.py`) — same paired discipline as `analyze.py`:
   omnibus first, bootstrap over questions, no subsetting knob.

**Counting.** Per (question, `top_p`) cell, premises are pooled over the cell's samples and
soundness is `sound / (sound + unsound)`. The headline is the mean over questions, which weights
each question equally and keeps the design paired. Truncated traces are **kept** — they made
visual reads before running out of budget, and excluding them subsets the data along the very
axis under test (11.09% truncated at `top_p`=0.5 vs 0.42% at 1.0). The drop-truncated variant
moves soundness by ≤1 pp and is reported alongside.

## 2. Diversity has to be count-matched, or it measures nothing

Premise count *itself* rises with `top_p` (6.40 → 7.33 per trace, +15%), and Vendi is the
**effective number of distinct items**, so it inherits that gradient directly:

| | 0.5 | 0.7 | 0.9 | 0.99 | 1.0 | rise | curvature |
|---|---|---|---|---|---|---|---|
| cell Vendi, **unmatched** | 3.852 | 4.073 | 4.261 | 4.365 | 4.443 | +15.4% | a=**+0.30** |
| cell Vendi, **count-matched** | 3.386 | 3.524 | 3.642 | 3.692 | 3.690 | +9.0% | a=**−0.44** |
| premises per cell (the confound) | 55.6 | 55.3 | 57.4 | 58.2 | 61.5 | +10.7% | |

About 40% of the apparent rise is just counting more premises, and the two versions disagree even
on the sign of the curvature. Every cell is therefore subsampled to exactly K items, 24 times,
averaged (K=20 premises per cell; K=5 traces for the across-sample measure, because the
empty-trace rate also drifts, 23.3% → 19.6%).

A **model-free** companion agrees: extract the numerals each trace asserts and measure how much
the samples disagree about the *values* they read off the image — mean Jaccard distance rises
0.208 → 0.291 (+37.7%, p=2e-06). Two measures with opposite blind spots, same direction.

## 3. The encoder was chosen by measurement, not reputation

A visual premise is often a number, and "$40,000" vs "$45,000" are *different observations*. Five
encoders were tested on **misread sensitivity** — how far a single changed digit travels from
"same claim" toward "a genuinely different read of the same image":

| encoder | misread sensitivity | dynamic range |
|---|---|---|
| all-MiniLM-L6-v2 | 0.141 | 0.896 |
| bge-base-en-v1.5 | 0.201 | 0.489 |
| **bge-large-en-v1.5** (used) | **0.211** | 0.492 |
| Qwen3-Embedding-0.6B | 0.138 | 0.627 |
| Qwen3-Embedding-8B | 0.185 | 0.601 |

**Scale does not help**: the 8B model is *worse* than 335M bge-large. No general-purpose encoder
treats a misread number as more than ~21% of a different claim, which is why the model-free
numeric measure exists.

## 4. Grounding decays as the trace proceeds

Soundness falls steeply with a premise's position in the reasoning — and this is the model, not
the judge losing attention down a list. Re-judging the same premises in **shuffled** order:

| ordering | slope per position |
|---|---|
| shuffled, by **trace** position (the model's own order) | **−0.0227** |
| shuffled, by **list** position (where the judge saw it) | −0.0030 |

On traces the extractor read end to end, soundness runs 0.977 at the first premise to 0.885 at
the tenth. Judge position bias is ~13% the size of the trace effect. Because higher `top_p`
produces more premises per trace, this is also most of the small negative soundness trend:
count-matching weakens it from p=0.011 to p=0.19.

## 5. The judge is the binding constraint

Same premises, same image, same judge, temp 0, only the order changed:

- agreement 86.3% on the SOUND/UNSOUND decision, **Cohen's κ = 0.363**
- **43.2% of UNSOUND verdicts flip to SOUND** on re-judge; 10.6% flip the other way

So "soundness is flat" is properly stated as **"no trend is detectable with a judge this
noisy"** — non-differential label noise attenuates real effects toward the null. Aggregate means
survive (187k premises average the noise out); individual trace verdicts do not.

## 6. Averaged soundness is the wrong statistic

**3,426 traces (15.8%)** assert both true and false things about the same image, and they answer
correctly 67.9% of the time against 74.1% for all-sound traces — a 1.3 pp penalty for being
demonstrably wrong about the picture.

Chasing individual cases showed why the curve is flat. Of the questions wrong on **all 8 samples**
at `top_p`=0.95 and right on at least one at 0.99, **three of eight** are genuine perception
flips — and none can be found by soundness, because winners and losers both score near 100%:

| question | gold | what only the winning trace saw |
|---|---|---|
| `test_Basic_Medical_Science_269` | D. Bone | dark spots that look like **lacunae**, **faint lines connecting** them, fibrous matrix — losers said "liver", and three asserted "the cells do not look like osteocytes in lacunae" |
| `test_Clinical_Medicine_283` | G. Thromboembolism | a vessel **lumen completely filled by a mass** with **layers of fibrin/platelets** — losers described lung parenchyma and never saw the clot |
| `test_Design_135` | E. perspective | bridge lines **converging on a vanishing point**, distant mountains fainter (atmospheric perspective) |

One decisive premise decides the answer while twenty incidental true ones dilute it to
invisibility. `outputs/zero_to_some_review.ipynb` has all eight with images, the winner-vs-loser
claim diff (matched by bge-large at cosine 0.90, so paraphrases do not read as new), and a
verdict on each.

## 7. Caveats

- **The judge is small.** InternVL3-8B-AWQ is what fits in 24 GB; the absolute 94% level is
  judge-dependent. Only the *shape* across `top_p` is claimed, and the judge is identical at
  every level. κ=0.36 bounds everything at trace resolution.
- **The extractor sees the first 6,000 tokens** of each trace (40.3% are clipped), and the clip
  rate drifts 36.8% → 44.1% along the axis. Premises asserted later are invisible to every
  measurement here.
- **Diversity rises on questions that get worse too.** Among true majority-vote flips, ΔVendi is
  +0.040 for wrong→correct and +0.050 for correct→wrong (p=0.94). The mechanism is statistical
  over many questions, not visible per question.
- **8 of 16 samples** were judged. Doubling would cut soundness SEMs ~0.007 → ~0.005, but the
  binding constraint is judge noise, not sample count: 3-vote shuffled-order judging buys more.
- **One model, one benchmark, one temperature arm.**

## 8. Files

```
vp_prep.py         split traces into balanced per-sample layers (ballot rule applied here)
vp_extract.py      premises from each <think> trace (token-capped, resumable)
vp_judge.py        InternVL3-8B-AWQ verdicts against the image, gold withheld
vp_result.py       soundness + accuracy tables and chart
vp_diversity.py    count-matched Vendi, cosine, and the model-free numeric measure
vp_embed_probe.py  the five-encoder misread-sensitivity comparison
vp_judge_shuffle.py  re-judge with premises reordered -> the kappa above
vp_zero_review.py  the Colab notebook of the eight zero-to-some cases
run_vp.sh          layer-at-a-time driver; run_vp_watchdog.sh keeps the GPU busy
outputs/
  vp_premises.jsonl.gz   extracted premises, one row per trace
  vp_verdicts.jsonl.gz   judged verdicts, one row per trace   <- the 187,784 premises
  RESULT_VP.md/.json          soundness + accuracy
  RESULT_VP_DIVERSITY.md      diversity, matched and unmatched
  RESULT_VP_EMBED_PROBE.md    encoder selection
  zero_to_some_review.ipynb   the eight cases, self-contained
```

**The premise and verdict data are committed gzipped, like `cots/`.** An earlier run of this arm
was lost because they were left untracked in a gitignored directory; ~15 GPU-hours had to be
regenerated. They are derived data, but they are not cheap derived data.
