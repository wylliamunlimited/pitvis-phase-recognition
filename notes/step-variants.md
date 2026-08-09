# Step variants — the iteration on top of ARST

The task-1 counterpart to [`instrument-variants.md`](instrument-variants.md),
and the same protocol. [`citi-baseline.md`](citi-baseline.md) stays the CITI
reproduction; this is what we tried to beat it with.

Code: `src/pitvis/training/arst_v2.py`, `src/pitvis/training/crossval.py`.

---

## 1. What was wrong

ARST reproduces at **0.3425** on the five validation videos. Table 8 benchmarks
CITI at **70** on those same five. That is the same ~50% shortfall instrument
recognition had, and the two reproductions share exactly one deviation from
their published counterparts: a frozen ImageNet backbone.

The per-class failure has the same shape too — from `citi-baseline.md`, step 3
(septum displacement) and step 9 (synthetic graft) at **0.000 recall**, step 6
(durotomy) at 0.036 — against an unweighted cross-entropy trained over a
distribution running 23.9% (tumour excision) to 0.06% (nasal packing).

And one lever had been sitting unclaimed. `CLAUDE.md` records that masking
classes 0/11/13 out of the argmax **"can only raise the official metric"** —
the metric filters by ground truth only and calls `f1_score` with no `labels=`,
so predicting an excluded class joins the macro average at F1 = 0. It existed
only as an ablation flag, off by default.

---

## 2. Protocol

Identical to task 2, and shared code: 5-fold cross-validation over the same
frozen folds (`data/folds.py`), each of the 19 training videos held out exactly
once, aggregated per-video-then-mean by one `evaluation.metric.evaluate` call.
VAL scored **once**, for the winner. `macro_f1` ranks; the official `metric`
is guarded against regression beyond one std of the 19-video spread.

The harness is task-agnostic — `crossval.Task` holds the loader, the scorer and
which metric ranks, so the fold logic and the leaderboard are shared rather
than copied.

---

## 3. The variants

```mermaid
flowchart TD
    F["cached features<br/>(T, 2048) float32"]
    STD["<b>standardize</b><br/>fold-train mean/std"]
    SP["<b>SpatialEmbedding</b><br/>Linear 2048 -> 512"]
    TC["<b>TeCNO</b><br/>2 x 8 dilated causal layers<br/>~17 min receptive field"]
    AR["<b>ARST</b><br/>1-layer enc-dec, banded causal mask W=5<br/>auto-regressive over phase labels"]
    CE["<b>cross-entropy</b> unweighted<br/>at all three stages"]
    D["<b>argmax + CCI</b><br/>n=10, all 15 classes eligible"]
    P["(T,) one step per second"]
    F --> STD --> SP --> TC --> AR --> CE
    AR --> D --> P
```

**control** — ARST unchanged, the anchor.

**masked** — the decision node only: `0/11/13` removed from the argmax, so an
excluded class can never be emitted onto a scored row.

```mermaid
flowchart TD
    AR["ARST logits (T, 15)"]
    D["<b>argmax + CCI</b><br/>classes 0, 11, 13 masked to -inf"]
    P["(T,) steps"]
    AR --> D --> P
    style D fill:#eef7ff,stroke:#69c
```

**weighted** — the loss node only: capped inverse-frequency class weights at
all three stages, computed on the fold's own training videos. Capped at 10 for
the same reason task 2's `pos_weight` is capped at 50 — nasal packing is 0.06%
of frames, so an uncapped weight lands in the thousands.

```mermaid
flowchart TD
    SP["SpatialEmbedding"] --> TC["TeCNO"] --> AR["ARST"]
    CE["<b>class-weighted cross-entropy</b><br/>capped inverse frequency<br/>at all three stages"]
    AR --> CE
    style CE fill:#eef7ff,stroke:#69c
```

**dinov2** — the input node only: `(T, 768)` DINOv2 ViT-B/14 features.

```mermaid
flowchart TD
    F["<b>DINOv2 features</b><br/>(T, 768) float32<br/>ViT-B/14 @ 224"]
    STD["standardize"] --> SP["SpatialEmbedding<br/>Linear 768 -> 512"]
    SP --> TC["TeCNO"] --> AR["ARST"] --> D["argmax + CCI"] --> P["(T,) steps"]
    F --> STD
    style F fill:#eef7ff,stroke:#69c
```

**best** — masking + class weights on DINOv2.

---

## 4. Results

### Leaderboard — 19 out-of-fold training videos

| variant | space | macro_f1 | edit_score | metric |
|---|---|---|---|---|
| **best @ dinov2** | dinov2_vitb14 | **0.5044**±0.103 | **0.5789**±0.114 | **0.5417**±0.092 |
| best | resnet50 | 0.4909±0.121 | 0.5218±0.126 | 0.5063±0.109 |
| masked | resnet50 | 0.4667±0.103 | 0.5127±0.123 | 0.4897±0.098 |
| weighted | resnet50 | 0.4393±0.113 | 0.4404±0.080 | 0.4399±0.076 |
| dinov2 | dinov2_vitb14 | 0.4337±0.095 | 0.4322±0.096 | 0.4329±0.082 |
| control (ARST) | resnet50 | 0.4047±0.093 | 0.4282±0.081 | 0.4164±0.076 |

### The pattern reproduced

**DINOv2 alone gains +0.029 macro — inside control's ±0.093 spread and inside
the ±0.076 guard tolerance.** On its own it is indistinguishable from noise.
Composed with masking and class weights it is the best variant on all three
metrics, and beats the same composition on ResNet-50.

That is exactly what happened on task 2 (+0.021 alone, +0.055 composed). **Two
tasks, same shape**: the representation only pays once the loss and the
decision rule stop masking it. It is a strong argument for never running a
backbone swap first — the null result is real and it is misleading.

**Masking was the largest single lever** at +0.062 macro / +0.073 metric, and
it is not a model change at all. It had been available as an ablation flag,
off by default, since before any of this work.

### Per class — where it moved

| step | name | support | control | best |
|---|---|---|---|---|
| 3 | septum displacement | 986 | 0.007 | **0.167** |
| 14 | debris clearance | 659 | 0.228 | **0.421** |
| 10 | fat graft placement | 1,905 | 0.466 | **0.582** |
| 1 | nasal corridor creation | 2,325 | 0.650 | 0.752 |
| 7 | tumour excision | 17,875 | 0.664 | **0.588** |
| 8 | haemostasis | 9,451 | 0.549 | **0.512** |
| 9 | synthetic graft placement | 2,696 | 0.278 | **0.224** |

**Three classes regress**, and that is the expected cost rather than a
surprise: reweighting moves capacity away from the dominant classes, and tumour
excision alone is 23.9% of frames. The macro average says the trade is worth
taking; a support-weighted reading would not.

### The single VAL scoring

| | ARST (control) | winner | delta |
|---|---|---|---|
| challenge `metric` | 0.3425 | **0.4610**±0.043 | **+0.119** |
| macro F1 | 0.3083 | **0.4420**±0.079 | +0.134 |
| edit score | 0.3767 | **0.4801**±0.041 | +0.103 |

Against Table 8's **70** for CITI on these same five videos, 46.1 is still well
short — the frozen backbone remains the untested lever (roadmap 1.7/3.6).

---

## 5. What this did not test

- **Backbone fine-tuning.** Still the largest untested lever, and the DINOv2
  interaction above is now evidence *for* it rather than against.
- **Per-class decision thresholds**, which paid on task 2. Steps are
  multi-class with a single argmax rather than 15 independent sigmoids, so the
  equivalent is a prior-corrected argmax — untried.
- **Generalisation past the training split.** CV ranks on training videos; the
  paper reports −7 points val→test for steps, milder than instruments' −47 but
  not nothing.
