# Handoff — re-run the top_p sweep as a clean maj@k experiment

For whoever picks this up next. The generation data in `cots/` is **already correct and
sufficient for the primary result** — Phase 1 below needs **no GPU**. Read §1 before
changing anything; it explains why the current reported numbers can't stand.

---

## 1. Why this re-run exists (audit findings)

An audit of the committed data found the following. Each is reproducible from
`cots/*.jsonl.gz` + `outputs/*.jsonl.gz`.

| # | finding | evidence |
|---|---|---|
| 1 | **Truncation is wildly asymmetric across `top_p`** and every truncated trace is silently dropped before analysis | 9.8% at `top_p`=0.1, 10.8% at 0.3, → **0.1% at 1.0**. README says "~5%". |
| 2 | **The per-sample inverted-U is an artifact of (1)** | Counting truncated as failures moves the peak from 0.5 to the 1.0 **edge**; bootstrap P(inverted-U) drops 73.5% → 3.2%. Repeated-measures ANOVA over the 7 levels: **F=0.68, p=0.667**. |
| 3 | **The maj@16 arm collapses to n=47 of 86 questions** because of (1) — the surviving subset is the easy one | maj@16 spans 0.89–0.96, a 3-question range; "peak at 0.7" is an argmax tie with 1.0 (already caveated in the README). |
| 4 | **Soundness does not predict answer correctness** — the causal story is unsupported | sound 0.739 (n=449) vs unsound 0.751 (n=398), χ² **p=0.752**; within-question paired **p=0.147**; between-question r=−0.041, **p=0.716**. |
| 5 | **The two judgings of identical claim sets disagree systematically** | atomic-pass/holistic-fail = 219, reverse = 3. McNemar **p=5.4e-61**. The "restored dynamic range" is substantially a rubric artifact. |
| 6 | The soundness trend itself **is** significant, but 20–50% of it is attributable to (1) | paired low vs high **p=0.0051**; bounding the dropped traces at scores 0.30/0.15/0.00 gives slope −0.113/−0.094/−0.075 vs the reported −0.150. |

**Net:** the only defensible claim from the current data is that `top_p` has no measurable
effect on accuracy under this configuration. This handoff specifies the re-run that tests
the original hypothesis properly.

---

## 2. The specification

### 2.1 Sampling — defaults OFF, only `top_p` varies

```
--top-k -1  --presence-penalty 0  --min-p 0  --repetition-penalty 1.0  --temperature 1.0
```

`cot_gen.py`'s argparse defaults are `top_k=20, presence_penalty=1.5`. **Do not use them.**
`top_k=20` caps the candidate pool below what `top_p` would select, so the sweep does almost
nothing; `presence_penalty=1.5` suppresses the repetition dynamics being measured. These are
already baked into `run_vpU.sh` — use it, don't hand-roll the command.

Everything except `top_p` is held fixed. One variable.

> ⚠️ Flagged assumption, not a blocker: at temperature 1.0, `top_p`≤1.0 can only *remove*
> probability mass, so theory predicts a rise-then-plateau, not an inverted-U. The falling
> arm lives above T=1.0. Run this spec as written — it is the clean test of the stated
> hypothesis — but expect flat, and see §7 for the natural follow-up.

### 2.1.1 Make the wrong config impossible — three code changes before you generate anything

The project's own headline lesson is that these flags decide the result. The code does not
currently enforce, record, or verify them. **Fix all three before Phase 2.**

**(a) The argparse defaults are a loaded gun.** `cot_gen.py:88-89` defaults to `top_k=20,
presence_penalty=1.5` — the config that erases the effect. Anyone who runs `cot_gen.py`
without the full flag list silently generates unusable data. Either flip these defaults to
the neutral values (`top_k=-1, presence_penalty=0`) or add a required
`--sampling-profile {neutral,qwen-recommended}` with no default, so the script refuses to run
until the choice is explicit. Do not rely on remembering the flags.

**(b) `--resume` does not check the config, and this is the live hazard for Phase 2.**
`cot_gen.py:173` keys completed cells on `(id, top_p, temperature)` only. Top up 0.6/0.8 with
the argparse defaults and it will **append `top_k=20` traces into the existing shard files,
interleaved with the correct ones, with nothing in the data to tell them apart.** That
irreversibly contaminates `cots/`. Make `--resume` read the config stamp from (c) and **abort
on any mismatch**. Until that guard exists, top up into a *new* output file, never into
`cots/u_gen.shard*.jsonl`.

**(c) Nothing records the sampling config — not the rows, not the summary, not the repo.**
Verified against the committed data:

| artifact | records `top_p` | records `temperature` | records `top_k` / penalties |
|---|---|---|---|
| each row in `cots/*.jsonl.gz` | ✅ | ✅ | ❌ |
| `*_summary.json` from `cot_gen.py:226` | ✅ | ✅ | ❌ |
| committed `outputs/` | — | — | ❌ (no `*_summary.json` committed at all) |

The sampling config of all 11,472 committed traces is attested **only by prose** in
`README.md` / `FINAL_SUMMARY.md` and by the flags in `run_vpU.sh`. It cannot be verified from
the data. For a project whose central finding is that a prior full run was invalidated by
these exact flags, that is the most important missing field in the dataset.

Required: write the **complete** sampling config (`top_k`, `top_p`, `temperature`, `min_p`,
`presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`, `max_tokens`, model
revision) into **every output row** — or into a per-shard header row referenced by the rows —
and into `*_summary.json`, and commit the summaries alongside the shards. Re-add the
`env_versions.txt` that was dropped in the prune commit (vLLM, transformers, torch versions).

Every analysis script should then **assert** the config it expects and fail loudly on
mismatch, rather than trusting the filename.

### 2.2 Grid — `top_p` > 0.4 only

**Use only `top_p` above 0.4. Do not include 0.1, 0.2, 0.3 or 0.4 in any reported analysis.**

Primary grid = the fully-generated values above 0.4: **0.5, 0.7, 0.9, 0.95, 1.0**
(86 questions × 16 samples each, already in `cots/`).

*Why exclude the low end.* Below ~0.5 the sweep stops measuring sampling diversity and starts
measuring a decoding pathology: near-greedy decoding of long reasoning traces degenerates into
repetition loops, at 9.8% (`top_p`=0.1) and 10.8% (0.3) versus 0.1% at 1.0. Whatever you do
with those traces — drop them and you bias the low end upward, count them as failures and the
low end is dominated by an artifact unrelated to the hypothesis — they contaminate the
comparison. Cutting below 0.4 removes the region where that failure mode dominates. It is also
where theory says nothing should happen: `top_p` only starts doing real work near the top of
the range, where truncation is actually removing tail mass.

*Two consequences you must design around:*

1. **An interior peak now has to sit at 0.7, 0.9 or 0.95.** If the true optimum is at 0.5, it
   is an *edge* of this grid and the shape test cannot detect interiority — that is a correct
   negative, not a failure of the method, but state it explicitly in the write-up rather than
   reporting "peak at 0.5" as if it were interior.
2. **Five points is thin for a two-lines test** (segments of 2–4 points). Densify before
   drawing conclusions — see Phase 2.

*What the restriction does NOT fix.* Truncation is still asymmetric inside the restricted
grid — 9.2% at `top_p`=0.5 versus 0.1% at 1.0, a ~90× differential. The ballot rule in §2.4
remains essential; the grid restriction is not a substitute for it.

*Extension (Phase 2, cheap).* Complete **0.6** (29 of 86 questions generated, needs 57 more)
and **0.8** (57 of 86, needs 29 more) to reach a 7-point grid above 0.4, and add **0.98** and
**0.99**. If any right-arm degradation exists at T=1.0 it is compressed into the top few
percent of probability mass, and the current grid has only 3 points above 0.9.

### 2.3 Metric — maj@k

Primary outcome is **maj@k for k ∈ {1, 2, 4, 8, 16}**, not per-sample accuracy. Majority vote
is the metric where diversity has value, so it is the one where an interior optimum is
theoretically expected.

**Sharp secondary prediction:** if a coverage-vs-precision tradeoff is real, the optimal
`top_p` must move **right as k grows**. The current results go the other way (maj@6 peaks at
the 1.0 edge, maj@16 at 0.7) — which was itself a signal that it was noise. Report the
optimum-vs-k relationship explicitly; it is stronger evidence than any single peak.

### 2.4 Counting rule — truncated samples are FAILURES, not exclusions

This is the core change. A generation with `finish_reason != "stop"` produced no answer.
In deployment that is a failure that consumed budget, not an event that didn't happen.

**Primary rule — spoiled ballot, fixed sample budget:**

- Every (question, `top_p`) cell has exactly **16 ballots**, one per generated sample.
- A ballot is *valid* if the trace terminated (`finish_reason == "stop"`) **and**
  `parse_answer` returns a letter. Otherwise it is **spoiled**.
- maj@k: draw k of the 16 ballots without replacement; take the plurality **among the valid
  ballots drawn**; if zero are valid, the draw is a **failure**. Spoiled ballots consume
  budget but do not vote.
- Tie-break: `Counter.most_common` first-seen order, matching the existing convention.

**Robustness rule (report alongside) — shared sentinel:** spoiled ballots all vote for one
sentinel answer that can never equal gold, so a cell with many truncations can have the
sentinel win the plurality. Strictly harsher; brackets the primary.

**Also report, unchanged:** the old exclude-truncated curve, so the delta between the three
rules is visible. That delta *is* a result — it quantifies how much of any observed shape is
a filtering choice.

**Do not** resample to replace truncated generations for the primary metric. That silently
changes the budget definition per cell and reintroduces the same bias.

### 2.5 Balancing

Under the new rule, **every question has 16 ballots at every `top_p`**, so per-k balancing no
longer drops anyone. The balanced set for maj@16 goes from **n=47 → n=86**. This is the single
biggest power gain in the re-run and it is free.

Keep the requirement that a question contributes only if present at every compared `top_p`.

---

## 3. Code changes

### `majk.py` — the only file that must change

1. **`load_cells(pattern, include_truncated=False)`** — drop the flag; always load every row.
   Return ballots, not just valid preds:

   ```python
   # was: preds = [a for a in (parse_answer(...) for r in rs) if a]      # silently drops both
   # now: one ballot per sample; None == spoiled (truncated OR unparseable)
   ballots = [parse_answer(r["text"], r.get("n_options", 10))
              if r.get("finish_reason") == "stop" else None
              for r in rs]
   out[key] = (ballots, rs[0]["gold"])
   ```

   Note `parse_answer` already returns `None` on unparseable output, so terminated-but-
   unparseable traces become spoiled ballots automatically — that is intended.

2. **`majk_cell(ballots, gold, k, B, rng)`** — sample k ballots, filter `None`, plurality of
   the remainder, failure if empty:

   ```python
   s = rng.sample(ballots, k)
   valid = [a for a in s if a is not None]
   hits += bool(valid) and Counter(valid).most_common(1)[0][0] == gold
   ```

   The `k >= len(preds)` fast path must now compare against `len(ballots)` (always 16), and
   the k==1 path becomes `sum(a == gold for a in ballots) / len(ballots)` — spoiled ballots
   count as wrong, which is the point.

3. **`majk_table`** — delete the per-k `len(...) >= k` filter. Every cell now has 16 ballots,
   so that guard only fires if generation itself is incomplete; assert instead of filter, and
   `log` loudly if any cell has fewer than 16 raw samples.

4. **`shape_verdict`** — replace. The current `amp > 2*max(se)` heuristic uses the marginal SE
   across questions, which is ~2.6× larger than the paired SE that actually applies (questions
   are shared across `top_p`; per-question accuracy correlates 0.917 across cells). Use the
   pre-registered test in §4 instead.

5. Add `--counting {spoiled,sentinel,exclude}` so all three rules in §2.4 come from one script.

### The other three drop sites — fix or annotate ALL of them

`majk.py` is not the only place truncated traces are discarded. There are **four** filters in
this directory and **three of them drop by default**. Nothing in the pipeline currently
reaches an analysis stage with truncated traces intact.

| file:line | what it does | required change |
|---|---|---|
| `majk.py:39` | `include_truncated=False` param, drops on load | **Remove the filter.** Ballot model, §3.1–3.3. This is the primary metric. |
| `analyze_cot.py:105` | `--include-truncated` flag, **default drops** | **Flip the default to keep.** Any run that drops must print the per-`top_p` drop rate, not just the total. |
| `combined_chart.py:54` | hard-coded `finish_reason != "stop": continue`, no flag | **Add the ballot model.** A figure with no flag to disable the filter is the worst case — the reader cannot tell it happened. |
| `extract_premises.py:112` | `--skip-truncated`, **default True** | Soundness arm, see §7. Re-extract with `--keep-truncated` before the soundness slope is quoted again. |

Note the help text at `analyze_cot.py:96-99`: it already documents that truncation prevalence
is "strongly top_p-dependent (11% at 0.5 → 0.6% at 1.0)" and that this "CONFOUNDS every top_p
/ diversity correlation". The confound was correctly identified — and then addressed by
dropping, which is the thing that biases the low end. Keep the diagnosis, reverse the remedy.

**Rule for this re-run:** no analysis or figure may silently exclude a generated sample. Either
count it as a failed ballot, or print the per-`top_p` exclusion counts in the output header so
the differential is visible in the artifact itself.

---

## 4. Pre-registered analysis plan — write the numbers down before you look

**Primary hypothesis:** maj@k accuracy is a non-monotone function of `top_p` with an interior
maximum, under the spoiled-ballot rule, at k=16.

**Shape test (primary):** two-lines. Split at the argmax; fit OLS to each segment; the shape
holds iff left slope > 0 **and** right slope < 0. Significance by **bootstrap over questions**
(resample question indices, apply the same resample to all `top_p` cells to preserve pairing),
B=20,000. Report P(shape holds). Pre-declare **α = 0.05**, i.e. the claim stands only if
P(shape) ≥ 0.95.

**Secondary:** sign of the quadratic coefficient, same bootstrap. Optimal `top_p` as a
function of k — pre-declare the prediction that it is non-decreasing in k.

**Omnibus first:** repeated-measures ANOVA over the 7 levels. If it does not reject, say so
prominently — pairwise tests after a null omnibus are not evidence. (It currently gives
p=0.667.)

**Multiplicity:** if you report pairwise tests against the peak, Holm-correct across all 6 and
report both raw and adjusted p. Never report only the winning comparison.

**Power note:** paired SE is ~1.2 pp, so a 3 pp effect is ~2σ at best. If you want the shape
test to be conclusive rather than suggestive, restrict to **informative items**. On the
restricted grid (0.5–1.0, ballot rule) 34 of 86 questions are answered correctly by all 16
samples at every `top_p` and 3 are always wrong, leaving **49 informative**. Report both the
full-set and informative-subset curves; pre-declare which is primary.

**For calibration, here is what the restricted grid already shows at k=1** under the ballot
rule — compute this first, it takes a minute and tells you what you are working with:

| top_p | 0.5 | 0.7 | 0.9 | 0.95 | 1.0 |
|---|---|---|---|---|---|
| all 86 questions | 0.754 | 0.765 | 0.766 | 0.773 | **0.789** |
| informative 49 | 0.629 | 0.648 | 0.651 | 0.662 | **0.691** |

Monotone rising to the 1.0 edge on both. maj@k is a different metric and may behave
differently — that is the point of running it — but do not expect the k=1 arm to produce a
peak, and do not go looking for one.

---

## 5. Execution phases

**Phase 0 — make the repo runnable (30 min, no GPU).** Currently no documented command works:
`majk.py` and `make_summary.py` read `outputs/u_gen.shard*.jsonl` but the shards ship as
`cots/u_gen.shard*.jsonl.gz`, and the analysis scripts default to `outputs/*.jsonl` that exist
only as `.gz`. Add transparent gzip open (`gzip.open` if `.gz`) and fix the default globs.
Also: `holistic_trend.py` balances over every `top_p` with ≥5 questions, which pulls in the
partial 0.4/0.6/0.8 and yields a **15-question** table — not the 50-question one in the README.
Only `make_summary.py` has the `COMPLETE` filter. Make them consistent.

**Phase 1 — the primary result, NO GPU REQUIRED.** The existing `cots/` were generated with
the correct sampling flags (see `run_vpU.sh`) and contain all 16 raw samples per cell,
truncated ones included. Everything in §2.3–§2.5 is a **pure re-analysis**. Run it, apply §4,
write the result down. This decides the question.

**Phase 2 — generation (GPU), if Phase 1 is ambiguous or the grid is too thin.** Stay above
0.4. Three jobs, cheapest first:

| job | traces | why |
|---|---|---|
| complete `top_p`=0.8 (29 missing questions) | 464 | 7-point grid above 0.4 |
| complete `top_p`=0.6 (57 missing questions) | 912 | ditto |
| add `top_p`=0.98, 0.99 (86 questions each) | 2,752 | density where the right arm must live |

`cot_gen.py --resume` handles top-up; use `run_vpU.sh` with `TOPPS` edited so the sampling
flags stay identical. Do **not** generate 0.1/0.2/0.3/0.4 — excluded by §2.2.

---

## 6. Do not do these

Each manufactures an interior peak from flat data. Two of them already happened here.

1. **Filter on `finish_reason` differentially across cells.** This alone moved the peak from
   1.0 to 0.5 in the current results.
2. **Pick the argmax, then report pairwise tests against it.** Guarantees your best comparison
   looks significant. The current README reports t=2.37 and t=2.09 this way; Holm-adjusted
   they are p=0.122 and p=0.200.
3. Compare unbalanced question sets across cells.
4. Choose k in maj@k, or choose among per-sample / maj@k / soundness, after seeing the curves.
5. Report `pearson` over the 7 aggregated points as evidence of effect size. With n=7 it
   describes how smooth the mean curve is, nothing more. Report the paired test on traces.

---

## 7. What to do about the soundness arm

Out of scope for this handoff, but do not quietly reuse the existing numbers — finding 4 says
the metric predicts nothing downstream and finding 5 says the instrument contradicts itself.
Minimum before it is cited again:

- Hand-adjudicate ~100 of the 219 atomic-pass/holistic-fail disagreements. One afternoon;
  it tells you which judging mode is correct and is a prerequisite for everything else.
- Re-judge including truncated traces, so the soundness slope isn't inheriting finding 1.
- Split claims into decision-relevant vs incidental (the judge already records which claim it
  flagged first, in the `wrong` field) and re-run the mediation on the relevant subset.

## 8. If Phase 1 comes back flat

That is a real, publishable result and should be written up as one, not buried:

> Under the model's non-default sampling configuration (`top_k` off, no penalties, T=1.0),
> sweeping `top_p` over 0.5–1.0 produces no measurable interior optimum in maj@k accuracy on
> MMMU-Pro; accuracy is flat-to-rising toward `top_p`=1.0. Non-terminating generations are
> counted as failures rather than excluded, which matters: they occur in 9.2% of traces at
> `top_p`=0.5 versus 0.1% at 1.0, and excluding them is on its own sufficient to manufacture
> an apparent interior peak. `top_p` below 0.5 is excluded as a separate regime dominated by
> repetition collapse (~10% of traces). This is consistent with Renze & Guven (EMNLP 2024
> Findings, arXiv:2402.05201), who found no significant temperature effect over 0.0–1.0
> across 9 models.

The natural follow-up, and the one I'd bet on: **`top_p` is inert at T ≤ 1 for RL-trained
reasoning models and becomes a real knob only once temperature pushes past 1.0**, where the
tail is inflated and truncation does actual work. Sweep `top_p` × T ∈ {0.7, 1.0, 1.3, 1.6}
and predict that curvature is absent at 0.7 and sharp at 1.6. A pattern across a grid is
evidence; a single interior argmax is not.
