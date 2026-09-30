# Reproducing CITI — the team that won PitVis-2023 task 1

Our reference model. This note records what the published method is, what we
built, and — importantly — where we knowingly did something different.

Code: `src/pitvis/models/arst.py` (the network), `src/pitvis/training/arst.py`
(three-stage training and inference).

For the same model traced as tensor shapes — mp4 to score, every dimension read
off the real files — see `citi-dataflow.md`. This note is the *why*; that one is
the *what shape*.

---

## 1. Who CITI are and what they submitted

CITI (Xiaoyang Zou, Guoyan Zheng — Institute of Medical Robotics, Shanghai Jiao
Tong University) won task 1 of PitVis-2023:

| Rank | Team | Metric | Macro-F1 | Edit |
|---|---|---|---|---|
| **1** | **CITI** | **62.9±9.7** | 61.1±10.6 | 64.7±10.1 |
| 2 | TSO-NCT | 53.7±11.2 | 58.2±10.9 | 49.2±13.0 |
| 3 | UNI-ANDES-23 | 48.3±7.3 | 50.1±9.3 | 46.5±8.2 |
| 4 | SANO | 20.5±3.2 | 39.6±6.5 | 1.4±0.4 |
| 5 | DOLPHINS | 15.2±4.0 | 28.9±8.2 | 1.6±0.7 |
| 6 | GMAI | 3.7±0.2 | 6.8±0.3 | 0.5±0.1 |
| 7 | CAIR-POLYU-HK | 3.5±0.8 | 5.8±1.5 | 1.1±0.3 |

(Das et al. 2024, Table 5 — the 8 private testing videos, which were never
released. Our numbers are on the 5 validation videos and are *not* directly
comparable; see §6.)

The drop between rank 3 and rank 4 is entirely in the edit column: 46–65 for
the models that look at time, 0.5–1.6 for the ones that judge each frame on its
own. That single fact is why this model is worth reproducing.

Their submission applies their own earlier work:

> **ARST: auto-regressive surgical transformer for phase recognition from
> laparoscopic videos.** Xiaoyang Zou, Wenyong Liu, Junchen Wang, Rong Tao,
> Guoyan Zheng. *Computer Methods in Biomechanics and Biomedical Engineering:
> Imaging & Visualization* 11(4), 2023. [arXiv:2209.01148](https://arxiv.org/abs/2209.01148)

The challenge paper describes CITI's PitVis setup slightly differently from the
ARST paper (§4). The ARST paper is the one that specifies the network fully, so
it is what we built.

---

## 2. The architecture

Three stages. Each is trained, then held fixed while the next one trains
(ARST §2.1–2.2, §3.3):

```
frames  --> ResNet-50 --> 2048 numbers --> Linear --> 512  Z_t     one frame, summarised
Z_1:T   --> TeCNO (two stacked time-aware networks) --> 512  F_t   one frame, in context
F_1:T + the previous step labels --> ARST --> a score for each of 15 steps
```

**TeCNO** (Czempiel et al. 2020) is two stacked convolutional networks that
only ever look backwards in time, 8 layers each, 512 channels. Each layer skips
further back than the last (gaps of 1, 2, 4, … 128 seconds), so one stage can
see `1 + 2*(1+2+...+128) = 511` frames of history and two stages roughly double
that. At one frame per second that is **about 17 minutes of history**, which
matters for §3.

**ARST** is a small transformer — one encoder layer and one decoder layer, 512
features wide, with attention split across 8 heads of 64 features each. The
encoder reads `F_1:T`. The decoder reads the step labels shifted back by one
and looks across at the encoder's output: it forms its queries from the labels
and matches them against keys and values taken from the video features.

---

## 3. The two ideas that make it work

### It only attends to the recent past

Not the usual "everything up to now" mask. Position `t` may attend only to
`[t-W, t]` — a fixed window. The paper's own comparison (ARST Table 2, on the
Cholec80 dataset):

| W | 0 | 2 | **5** | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| Accuracy | 84.83 | 87.00 | **87.62** | 87.13 | 86.10 | 83.57 |

Accuracy falls off on *both* sides of W=5, so there is a genuine best window
here rather than a "more context is always better" curve. Their reasoning: once
you can see what was just predicted, the distant past becomes a distraction for
deciding the current frame.

This works only because TeCNO has already packed about 17 minutes of history
into `F_t`. The transformer is not the part doing long-range work — it decides
*how steps follow one another over a few seconds*, on top of features that are
already time-aware. Worth remembering before anyone tries raising W.

### It feeds its own predictions back in

The decoder reads the steps it has already predicted, so the model learns

```
p(y_1:T | F_1:T) = prod_t p(y_t | y_0:t-1, F_1:t)         (ARST eq. 2)
```

rather than judging each frame on its own. The way steps follow one another is
therefore part of the model, instead of being smoothed on afterwards. This is
the mechanism behind the edit score, and it is the thing our simple per-frame
baseline cannot do at all — it labels every second independently and produces
13–34 times too many segments.

During training the model is shown the whole correct label sequence at once
(which lets it train in parallel). At prediction time it runs one second at a
time, feeding each answer back in.

### How a step is written down for the decoder (ARST eq. 3)

Not a single 1 in a row of zeros. The 512 features are cut into equal blocks,
one per step, and the step being represented sets its whole block to 1. With 15
steps each block is `512 // 15 = 34` wide and the last 2 features are always
zero. Any two steps now differ in 68 places rather than 2 — a much stronger
signal into the decoder. The encoding is fixed, not learned.

### It waits before believing a change of step (ARST §2.3, Algorithm 1)

This is the paper's Consistency Constraint Inference, or CCI. When the model
first predicts a new step at time `t`, it keeps feeding in the *old* step and
predicts the next `n=10` seconds. The change is accepted only if all 10 agree
with the new step; otherwise it is thrown away. This targets exactly the
one-second flickering that ruins the edit score.

**One caveat about timing, worth naming.** CCI decides second `t` only after
looking at `t+1` to `t+n`, so the system runs about ten seconds behind rather
than deciding strictly in the moment. The challenge rule says "only information
from frames up to and including the current frame can be used to classify the
current frame" (Das et al. 2024 §3.2). The organisers evidently accepted this:
the second-place team's smoothing behaves the same way. `--no-cci` gives the
strictly-in-the-moment version, and the two are worth reporting side by side.

---

## 4. How close this is to the published method

| Component | Published | Ours | Same? |
|---|---|---|---|
| Frame encoder | ResNet-50, ImageNet start, **trained for 50 more epochs** | ResNet-50, ImageNet, **not trained further** | ✗ see below |
| 2048 -> 512 projection | learned | learned | ✓ |
| TeCNO | 2 stages x 8 backward-looking layers, 512 ch | same | ✓ |
| ARST | 1 encoder + 1 decoder layer, 512 wide, 8 heads of 64 | same | ✓ |
| Attention window | W=5 | W=5 (`--width`) | ✓ |
| Step encoding | fixed blocks | same | ✓ |
| Position encoding | sine and cosine | same | ✓ |
| Shown the true labels while training | yes | yes | ✓ |
| CCI | n=10 | n=10 (`--no-cci` turns it off) | ✓ |
| Optimisers | SGD 1e-4 / Adam 1e-4 / Adam 1e-5 | same | ✓ |
| ARST batches | one whole video at a time | 1024-second pieces | ~ see below |
| Frame rate | 1 per second | 1 per second | ✓ |

### The difference that matters: we do not train the frame encoder

ARST trains ResNet-50 on the surgical images for 50 epochs, with random crops,
flips, rotation and colour changes. **This reproduction cannot** —
`extract_features.py` saves the summarised frames and throws the pixels away,
so there is no route back to the encoder to train it. Our first stage trains
only the `2048 -> 512` projection, on top of cached ImageNet features that never
change.

We expected this to be the single biggest reason our score falls short of the
published one, and that turned out to be right. Roadmap 1.7 created the missing
route (a cache of one JPEG per second) and 3.6/3.6b then trained both encoders;
training DINOv2 is the largest single gain in the project. This reproduction
deliberately leaves its encoder untrained — it is what CITI's *published* setup
scores on our data, and changing it would mean the baseline no longer measures
the baseline. The improvements live in [`step-variants.md`](step-variants.md)
§6–7.

Note also that CITI's *PitVis* entry used a Swin transformer to encode frames
rather than ResNet-50 (Das et al. 2024 §5.2). We follow the ARST paper, partly
because it is the fully specified one and partly because its encoder is exactly
what our cache already holds.

### The smaller difference: we train ARST on pieces of a video

ARST uses one whole video per step of training. Attention cost grows with the
square of the length, and our longest video is 8,645 seconds — an
8645 x 8645 x 8-head attention table is about 2.4 GB. We cut videos into
1,024-second pieces (`--chunk`). Because attention only reaches 5 seconds back,
only the first 5 positions of each piece lose any context: about 0.5% of
positions. At prediction time the pieces overlap by 5 seconds, so nothing is
lost there.

---

## 5. Running it

```bash
uv run pitvis-train arst
```

Each of these turns one component off, so its contribution can be measured:

```bash
uv run pitvis-train arst --no-cci          # decide in the moment, no ten-second wait
uv run pitvis-train arst --width 0         # remove the attention window
uv run pitvis-train arst --mask-excluded   # never predict steps 0/11/13
```

Output lands in `data/arst/`: `citi.pt` (all three stages), `result.json`, and
`standardize.npz` — the mean and standard deviation of the training features,
which closes roadmap 1.3 (these used to be computed inside
`training/baseline.py` and thrown away).

---

> **Partly superseded.** [`step-variants.md`](step-variants.md) applies the
> instrument-side method to task 1: never predicting steps 0/11/13, plus
> weighting the rare steps more heavily in the loss, on DINOv2 features, lifts
> the challenge metric from 0.3402 to 0.4610 on the validation videos, and a
> trained DINOv2 encoder takes it to **0.5608**. Everything below is unchanged
> and is still what `pitvis-train arst` produces.

## 6. Results

**These numbers cannot be compared to Table 5.** The challenge scored 8 private
testing videos that were never released; we score the 5 suggested validation
videos (01, 12, 21, 24, 25), which were part of every team's *training* data.
Different videos, different difficulty, 5 cases against 8. Treat Table 5 as a
direction, not a target.

The number we can honestly compare against is our own per-frame baseline on the
same 5 videos. Validation split, seed 0, defaults unless stated:

| Config | Metric | Macro-F1 | Edit |
|---|---|---|---|
| per-frame baseline (one linear layer) | 0.1599 | 0.3060 | 0.0138 |
| **ARST, as published (W=5, CCI on)** | **0.3349 ± 0.0473** | 0.3226 | **0.3472** |
| ARST `--no-cci` | 0.2875 ± 0.0454 | 0.3185 | 0.2565 |
| ARST `--width 0` | 0.3439 ± 0.0506 | 0.3288 | 0.3591 |
| ARST `--mask-excluded` | 0.4111 ± 0.0531 | 0.3804 | 0.4417 |

Training is cheap: **112 s** for all three stages on this Mac's GPU, plus about
50 s to predict all 5 validation videos. The cached features are what buy that.

### The same run does not give the same number twice

Re-running the published setup after the package was restructured gave **0.3402
± 0.0484** (macro-F1 0.3255, edit 0.3548, 1,249 predictions of excluded steps
on scored rows) against the 0.3349 ± 0.0473 in the table — same seed, same data,
same code path. A third run on 2026-08-09, when `arst_v2`'s control variant was
added, gave 0.3425.

> **0.3402 is the published baseline**, and this note is where that is decided.
> Three real runs of the identical setup exist — 0.3349, 0.3402, 0.3425 —
> spanning 0.008, comfortably inside the ±0.048 spread between videos. They were
> being quoted interchangeably across the notes, so the headline improvement
> depended on which one a given file happened to pick.
>
> 0.3402 wins because it is the only one a saved model reproduces: it is what
> `data/arst/result.json` holds and what `pitvis-eval --ckpt data/arst/citi.pt`
> prints today. Every improvement quoted elsewhere is measured from it. The
> other two stay here as the evidence that runs vary, and are quoted nowhere
> else.

The spread across those runs is 0.005, far below the ±0.048 spread between
videos. This Mac's GPU adds up numbers in an order that varies between runs, so
`torch.manual_seed` fixes the starting weights and the shuffling but not the
arithmetic. Treat the third decimal of the table as noise, and do not read a
difference below 0.01 between any two rows as real — the `--width 0` row in
particular sits well inside that band.

### What changed, and what did not

**The edit score went 0.0138 -> 0.3472, about 25 times better. Macro-F1 barely
moved: 0.3060 -> 0.3226.** The whole gain is in keeping the predicted steps
stable over time, which is what the architecture is for, and which a per-frame
model cannot reach.

That split is also the clearest evidence for the untrained-encoder diagnosis in
§4. Macro-F1 is a per-frame measure — roughly, how well the features separate
the steps. Putting a strong time-aware model on top left it almost unchanged,
so the per-frame ceiling is set by the features, not by what reads them.
Training the encoder (roadmap 3.6, which needed 1.7 first) is where the
remaining F1 is.

### Measuring the prediction-time switches on one fixed model

The table above retrains for every row, so each number carries fresh run-to-run
noise on top of the effect being measured. Two of those rows change only how a
trained model is used — `--no-cci` and `--mask-excluded` alter no weights at
all. Scoring one saved model four ways (`uv run pitvis-eval`) separates them
properly:

| prediction setting | metric | Δ vs default | excluded steps predicted |
|---|---|---|---|
| default (W=5, CCI on) | 0.3402 ± 0.0484 | — | 1,249 |
| `--no-cci` | 0.2937 ± 0.0473 | **−0.047** | 1,149 |
| `--mask-excluded` | 0.4405 ± 0.0457 | **+0.100** | 0 |
| `--no-cci --mask-excluded` | 0.3969 ± 0.0358 | +0.057 | 0 |

Same weights throughout, so these differences are the effect and nothing else.
Both are larger here than the retraining table suggests — never predicting the
excluded steps is worth +0.100 rather than +0.076 — and the two overlap rather
than adding up: CCI is worth +0.047 on its own but only +0.044 alongside the
masking, which makes sense, because both are cleaning up the same unstable
predictions.

Use this table when reasoning about prediction-time choices, and the retraining
table only for changes that actually alter weights (`--width`).

### Waiting before believing a step change is worth it

Turning CCI off costs 0.047 (0.3349 -> 0.2875), almost all of it in the edit
score (0.3472 -> 0.2565). Remember that CCI is the part that decides ten seconds
late, so 0.2875 is the strictly-in-the-moment number.

### The attention window did not reproduce

`--width 0` scores 0.3439 against 0.3349 for the paper's best W=5 — slightly
*better*, and well inside the ±0.05 spread between videos. We do not see the
rise-then-fall pattern of ARST Table 2 on Cholec80.

Two possible reasons, neither confirmed: (a) TeCNO's roughly 17 minutes of
backward context already supplies all the history that is needed, leaving the
transformer's reach nearly irrelevant here, and (b) 5 validation videos with a
spread of 0.05 cannot resolve a 0.009 difference. Do not tune W on this split —
it is not measuring anything.

### The failure turned into its opposite

Predicted numbers of segments against the truth:

| video | true segments | per-frame baseline | ARST |
|---|---|---|---|
| 1 | 78 | 2,679 (34x) | 57 (0.7x) |
| 12 | 84 | 1,612 (19x) | 39 (0.5x) |
| 21 | 182 | 2,391 (13x) | 44 (0.2x) |
| 24 | 78 | 1,145 (15x) | 31 (0.4x) |
| 25 | 59 | 1,477 (25x) | 27 (0.5x) |

The baseline chopped the video into 13–34 times too many pieces. ARST now
produces too **few**, at 0.2–0.7 times the true count. Feeding predictions back
in, plus waiting before believing a change, made the model reluctant to switch
steps — the right trade for the edit score, but it creates a new problem: whole
short or rare steps get swallowed and are never predicted at all. Per-step
recall shows it — step 3 (septum displacement) and step 9 (synthetic graft
placement) are both at **0.000**, and step 6 (durotomy) at 0.036.

This is the next thing to work on, and it is a genuinely different problem from
the one we started with.

### Never predicting the excluded steps really is nearly free

`--mask-excluded` removes steps 0, 11 and 13 from the set the model may choose,
and gains **0.076** (0.3349 -> 0.4111) — the largest single improvement
measured, from a three-line change. It works because the official metric drops
excluded steps from the *truth* only, and calls `f1_score` without naming the
label set, so any prediction of an excluded step joins the average at F1 = 0.
The as-published run of that table predicts 1,252 of them on scored rows; the
later 0.3402 run predicts 1,249, and the two counts belong to those two runs
rather than to different measurements.

The second-place team did the same thing — "any steps not considered for
evaluation were replaced with the most recent permitted step". It exploits how
the score is computed rather than making the model better, so both numbers are
reported and the headline stays the as-published one.
