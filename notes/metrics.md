# The evaluation metrics

What each metric measures, why the challenge chose it, and what it catches that
the others miss. Reference layer — `data-dictionary.md` does this for the data
and `citi-dataflow.md` for the model.

Every number here was measured on this repo's own runs.

---

## The short version

| metric | measures | used by |
|---|---|---|
| **macro F1** | per-class F1, **unweighted** mean — every class counts equally | task 1 |
| **edit score** | normalised Levenshtein over the *segment sequence*, durations collapsed | task 1 |
| **weighted F1** | per-class F1 weighted by **support** — frequent classes dominate | task 2 |

```
task 1 (steps)        (macro F1 + edit score) / 2
task 2 (instruments)  weighted F1 only
task 3 (multitask)    (task 1 + task 2) / 2
```

All three are computed **per video, then mean-averaged** — never pooled across
videos. Pooling is not a harmless approximation; see §5.

---

## 1. Macro F1 — because the classes are wildly imbalanced

F1 per class, then a plain unweighted mean. A class holding 0.06% of frames
counts exactly as much as one holding 23.9%.

That is the whole point. Step 7 (*tumour excision*) is 23.9% of annotated time;
step 13 (*nasal packing*) is 0.06% — a single 72-second segment in one video.
Plain accuracy would hand you ~24% for the constant prediction "tumour
excision", and a model could ignore every rare step and still look respectable.
Macro averaging removes that option.

Das et al. §3.4.1: *"Taking a macro-mean across classes ensures each class is
treated equally so major classes do not dominate."*

**What it actually measures, in practice: how separable your features are.** It
is a per-frame quantity — no notion of order or duration — so it rises when the
representation improves and is largely indifferent to temporal modelling. That
turns out to be the single most useful diagnostic in this repo (§4).

---

## 2. Edit score — because per-frame accuracy is blind to structure

Collapse each label sequence into *segments* (`groupby`, so runs of the same
class become one element), then compute a normalised Levenshtein distance
between the two segment sequences:

```
1 - Lev(pred_segments, true_segments) / max(len(pred), len(true))
```

**Duration is entirely discarded.** Only which steps occur, and in what order.
`[1,1,1,1,2,2]` and `[1,2,2,2,2,2]` both collapse to `[1, 2]` and score a
perfect 1.0 against each other.

This is the only term that punishes **flickering**. A frame-wise classifier can
be right most seconds while alternating between classes constantly — shattering
a smooth operation into thousands of spurious segments. Per-frame metrics cannot
see that; the edit score sees almost nothing else.

Two behaviours that look like bugs and are not:

- **It runs *after* the rarity exclusions.** Removing classes `-1/11/13` splices
  the sequence, so `[1,1,bg,bg,1,1]` becomes one segment, not three.
- **`1/Lev` in the paper (Eq. 3) is wrong.** It divides by zero on a perfect
  prediction, and CITI's reported 64.7 would imply ~1.5 edits across an entire
  multi-hour operation. The code is authoritative. See `CLAUDE.md`.

---

## 3. Weighted F1 — task 2's choice, and its weakness

Per-class F1 weighted by support, so a class holding 36% of positives moves the
score 180× more than one holding 0.2%.

For instruments, ids 16 (suction, 36.0%), 0 (nothing visible, 31.5%), 8
(kerrisons, 13.3%) and 13 (ring curette, 10.4%) carry ~91% of positives between
them. Get those four right and the other fifteen barely register.

**Our own run demonstrates the problem.** SANO reproduction, val split:

| reading | score |
|---|---|
| weighted (aligned) | **0.6309** |
| macro | **0.2513** |

The same predictions. The model **never predicts 9 of the 19 classes at all** —
and still scores 0.63 weighted, because those nine hold under 9% of positives.
Macro tells you what weighted hides.

Note the irony: Das et al. §3.4.3 and Table 6's own column header both say
*macro*-F1 for task 2, while the shipped script computes *weighted*. Nothing
resolves which produced the published numbers, so `evaluation/instruments.py`
prints all three readings and claims no leaderboard comparability. Details and
the column-ordering defect in `instruments.md` §3.

---

## 4. Why task 1 combines two metrics — the evidence

Macro F1 and edit score catch **orthogonal** failures, and this repo's results
are an unusually clean demonstration.

| model | macro F1 | edit | metric |
|---|---|---|---|
| frame-wise linear probe | 0.3060 | **0.0138** | 0.1599 |
| ARST (CITI) | 0.3255 | **0.3548** | 0.3402 |

(Both rows from the runs whose artifacts are on disk. MPS kernels are not
bit-deterministic, so ARST's third decimal moves between runs — see
`citi-baseline.md` §6. Nothing here turns on it.)

Same cached features, same 5 validation videos. Edit improved **26×**; macro F1
moved 0.020. The probe was *adequate* per second and *catastrophic* structurally
— it produced 13–34× too many segments:

| video | true segments | linear probe | ARST |
|---|---|---|---|
| 1 | 78 | 2,679 (34×) | 57 |
| 21 | 182 | 2,391 (13×) | 44 |

(Segment counts from the original ARST run; the ratios, not the exact counts,
are the point.)

**That split is diagnostic, not cosmetic.** Macro F1 barely moving under 24M
parameters of temporal modelling is what tells us the per-frame ceiling is set
by the frozen ImageNet features, not by the classifier. Everything in the
roadmap about unfreezing the backbone follows from reading those two columns
separately — which you cannot do if you only report the combined number.

The challenge leaderboard says the same thing at larger scale. Task 1, Table 5:

| rank band | edit |
|---|---|
| 1–3 (temporal models) | 46.5 – 64.7 |
| 4–7 (spatial-only) | 0.5 – 1.6 |

Macro F1 ranged 5.8–61.1 across the same teams. **The entire separation between
good and bad submissions was the edit column.**

---

## 5. Aggregation: per video, then mean — never pooled

Scores are computed per video and mean-averaged, reported as mean±std. Das et
al. state it explicitly: *"mean-averaged across the 8-testing-videos"*, *"not
pooled frame-wise"*.

Pooling **inflates** the score rather than approximating it, for two reasons:
concatenation merges the last segment of one video with the first of the next
(inventing agreement in the edit score), and opposite per-video errors cancel in
the frame-wise F1. `test_pooling_videos_flatters_the_score` pins this on a
two-video toy case: **0.583 pooled against 0.417 honest**.

So `metric.evaluate` takes `[(vid, y_true, y_pred), ...]` and never
concatenates.

---

## 6. CCI — not a metric, but it exists because of one

**Consistency Constraint Inference** (ARST §2.3) is a decoding rule, not an
evaluation rule. On a predicted transition at `t`, the decoder keeps asserting
the *old* phase and looks ahead `n=10` frames; the transition is accepted only
if all 10 lookahead predictions agree.

It exists to buy edit score, and it does — measured on one fixed checkpoint so
the weights are held constant:

| inference config | metric | Δ |
|---|---|---|
| default (CCI on) | 0.3402 | — |
| `--no-cci` | 0.2937 | **−0.047** |

Almost all of that is the edit term. The cost is that frame `t` is finalised
after observing `t+10`, making the system fixed-**lag** rather than strictly
causal — which sits awkwardly against the challenge's online-only rule, though
the challenge evidently tolerated it (TSO-NCT's threshold smoothing has the same
property). `--no-cci` gives the strictly causal number and both are reported.

---

## 7. What the exclusion rule is worth

The three preserved official behaviours — exclusion by ground truth only,
`zero_division=1`, and the edit score running after exclusion — are enumerated
as rules in `CLAUDE.md` and traced to `file:line` with their pinning tests in
[`walkthrough.md` §12](walkthrough.md). Not repeated here.

What belongs to *this* layer is what the first one is worth. Because a
predicted-but-excluded class joins the macro average at F1 = 0, **masking
classes 0/11/13 out of the argmax can only raise the score** — measured at
**+0.100** on a fixed checkpoint, and +0.062 macro out of fold when it became
the `masked` step variant. TSO-NCT (2nd place) did the same thing.

That is a scoring-rule exploit, not a modelling improvement, and the distinction
is the point: it moves the number without the model having learnt anything. The
faithful reproduction number stays the headline in
[`citi-baseline.md`](citi-baseline.md); the masked number is reported as a
variant in [`step-variants.md`](step-variants.md), never as a like-for-like
comparison against the challenge table.

Task 2's metric has a fourth behaviour that is genuinely broken rather than
merely surprising — see [`instruments.md` §3](instruments.md).

---

## 8. The caveat that applies to every number in this repo

**Our validation scores are not comparable to the published leaderboard.** The
challenge scored 8 private test videos that were never released; we score the 5
suggested validation videos, which were part of every team's *training* data.

For task 2 the gap is enormous — Das et al. §6.5 measures a **−47%** val→test
drop for instruments (SDS-HD: 89 on validation, 41.7 on test) against −7% for
steps. Treat the published tables as a direction, not a target line.

The internally valid comparisons are the ones within this repo, on the same 5
videos: linear probe against ARST, and constant baselines against the instrument
model.
