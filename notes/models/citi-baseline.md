# Reproducing CITI — the PitVis-2023 task-1 winner

Our reference model. This note records what the published method actually is,
what we implemented, and — importantly — where we knowingly diverge.

Code: `src/pitvis/models/arst.py` (architecture), `src/pitvis/training/arst.py` (three-stage
training + inference).

For the same model traced as tensor shapes — mp4 to score, every dimension read
off the real artifacts — see `citi-dataflow.md`. This note is the *why*; that
one is the *what shape*.

---

## 1. Who CITI are and what they submitted

CITI (Xiaoyang Zou, Guoyan Zheng — Institute of Medical Robotics, Shanghai Jiao
Tong University) won task-1 of PitVis-2023:

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

The cliff between rank 3 and rank 4 is entirely in the edit column: 46-65 for
the temporal models, 0.5-1.6 for the spatial-only ones. That single fact is why
this model is worth reproducing.

Their submission is an application of their own prior work:

> **ARST: auto-regressive surgical transformer for phase recognition from
> laparoscopic videos.** Xiaoyang Zou, Wenyong Liu, Junchen Wang, Rong Tao,
> Guoyan Zheng. *Computer Methods in Biomechanics and Biomedical Engineering:
> Imaging & Visualization* 11(4), 2023. [arXiv:2209.01148](https://arxiv.org/abs/2209.01148)

The challenge paper describes CITI's PitVis configuration slightly differently
from the ARST paper (§4). The ARST paper is the one with a full architectural
specification, so it is what we implemented.

---

## 2. The architecture

Three stages, each trained then frozen before the next (ARST §2.1-2.2, §3.3):

```
frames  --> ResNet-50 --> 2048-d --> Linear --> 512-d  Z_t     spatial embedding
Z_1:T   --> TeCNO (two cascaded causal TCNs)  --> 512-d  F_t   temporal feature
F_1:T + shifted phase labels --> ARST         --> 15-way logits
```

**TeCNO** (Czempiel et al. 2020) is two cascaded causal TCNs, 8 dilated layers
each (dilations 1, 2, ..., 128), 512 channels. Causal receptive field per stage
is `1 + 2*(1+2+...+128) = 511` frames; two stages roughly double it. At 1 fps
that is **~17 minutes of history**, which matters for §3.

**ARST** is a one-layer encoder-decoder transformer, `d_model=512`, 8 heads,
64-d per head. The encoder consumes `F_1:T`; the decoder consumes the *shifted
phase labels* and cross-attends to the encoder (Q from the decoder, K/V from
the encoder).

---

## 3. The two ideas that make it work

### Banded causal mask

Not the usual upper-triangular mask. Position `t` attends only to `[t-W, t]` —
a *window*, not the whole past. The paper's ablation (ARST Table 2, Cholec80):

| W | 0 | 2 | **5** | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| Accuracy | 84.83 | 87.00 | **87.62** | 87.13 | 86.10 | 83.57 |

Performance falls off on *both* sides, so this is a real optimum rather than a
"more context is better" curve. The stated rationale: once recent predictions
are available, long-range past becomes noise for the current decision.

This works only because TeCNO already carries ~17 minutes of context in `F_t`.
The transformer is not the thing doing long-range modelling — it is doing
*short-range transition* modelling on top of an already-temporal feature. Worth
keeping in mind before "improving" W upward.

### Auto-regression over phase labels

The decoder consumes its own past predictions, so the model learns

```
p(y_1:T | F_1:T) = prod_t p(y_t | y_0:t-1, F_1:t)         (ARST eq. 2)
```

rather than a per-frame posterior. Phase-transition structure is modelled
*explicitly* instead of being smoothed on afterwards. This is the mechanism
behind the edit score, and it is what our frame-wise linear probe structurally
cannot do — it predicts every second independently, producing 13-34x too many
segments.

Training uses teacher forcing (the whole shifted ground truth at once, so it
parallelises). Inference is a genuine sequential rollout.

### Phase embedding (ARST eq. 3)

Not one-hot. A 512-d vector is split into `c` equal segments; phase `i` sets its
entire segment to 1. With 15 classes the segment width is `512 // 15 = 34` and
the final 2 dimensions are always zero. Any two phases now differ in 68
coordinates instead of 2 — a much stronger signal into the decoder. It is a
fixed encoding, not learned.

### Consistency Constraint Inference (ARST §2.3, Algorithm 1)

On a predicted transition at `t`, keep feeding the *old* phase and predict the
next `n=10` frames. Accept the transition only if all `n` agree with the new
phase; otherwise revert. Attacks exactly the flickering that destroys the edit
score.

**Causality caveat worth naming.** CCI decides frame `t` after observing
`t+1..t+n`, so the system is fixed-*lag*, not strictly causal — despite the
challenge rule that "only information from frames up to and including the
current frame can be used to classify the current frame" (Das et al. 2024
§3.2). The challenge evidently tolerated this: TSO-NCT's threshold smoothing
(2nd place) has the same property. `--no-cci` gives the strictly causal
version, and the two are worth reporting side by side.

---

## 4. Faithfulness: what we implemented vs what is published

| Component | Published | Ours | Faithful? |
|---|---|---|---|
| Spatial backbone | ResNet-50, ImageNet init, **fine-tuned 50 epochs** | ResNet-50, ImageNet, **frozen** | ✗ see below |
| 2048 -> 512 projection | learned | learned | ✓ |
| TeCNO | 2 stages x 8 dilated causal layers, 512 ch | same | ✓ |
| ARST | 1-layer enc-dec, d=512, 8 heads, d_k=64 | same | ✓ |
| Banded causal mask | W=5 | W=5 (`--width`) | ✓ |
| Phase embedding | segmented, fixed | same | ✓ |
| Positional encoding | sinusoidal | same | ✓ |
| Teacher forcing | yes | yes | ✓ |
| CCI | n=10 | n=10 (`--no-cci` to ablate) | ✓ |
| Optimisers | SGD 1e-4 / Adam 1e-4 / Adam 1e-5 | same | ✓ |
| ARST batching | one whole video per iteration | 1024-frame chunks | ~ see below |
| Frame rate | 1 fps | 1 fps | ✓ |

### The deviation that matters: the backbone is frozen

ARST fine-tunes ResNet-50 on the surgical data for 50 epochs with SGD, random
crops, flips, rotation and colour jitter. **We cannot.** `extract_features.py`
saves embeddings and discards the pixels, so there is no data path to
backpropagate into the backbone (roadmap 1.7). Our spatial stage trains only
the `2048 -> 512` projection on cached frozen ImageNet features.

This is expected to be the single largest source of gap to the published
number. The cached features come from a network that has never seen an
endoscope — the cross-cutting risk already flagged in `roadmap.md`.

Note also that CITI's *PitVis* submission used a Swin transformer spatial
encoder rather than ResNet-50 (Das et al. 2024 §5.2). We follow the ARST paper,
partly because it is the fully specified one and partly because its backbone is
exactly what our cache already holds.

### The minor deviation: chunked ARST training

ARST uses one whole video per iteration. Full-video self-attention is O(T²) and
our longest video is 8,645 frames — an 8645x8645x8-head attention matrix is
~2.4 GB. We chunk to 1,024 frames (`--chunk`). Because the mask is banded at
W=5, only the first 5 queries of each chunk lose any context: ~0.5% of
positions. The encoder at *inference* is chunked with W-overlap and is exact.

---

## 5. Running it

```bash
uv run pitvis-train arst
```

Ablations that isolate each claim:

```bash
uv run pitvis-train arst --no-cci          # strictly causal, no lag
uv run pitvis-train arst --width 0         # kill the banded attention
uv run pitvis-train arst --mask-excluded   # drop 0/11/13 from the argmax
```

Artifacts land in `data/arst/`: `citi.pt` (all three stages), `result.json`,
and `standardize.npz` — the train-split feature mean/std, which closes roadmap
1.3 (previously computed inline in `train_baseline.py` and thrown away).

---

> **Superseded in part.** [`step-variants.md`](step-variants.md) applies the
> instrument-variant protocol to task 1: argmax masking plus class weights on
> DINOv2 features lifts the challenge metric from 0.3425 to 0.4610 on val, and
> a fine-tuned DINOv2 encoder takes it to **0.5608**. The reproduction below is
> unchanged and still what `pitvis-train arst` produces.

## 6. Results

**These numbers are not comparable to Table 5.** The challenge scores 8 private
testing videos that were never released; we score the 5 suggested validation
videos (01, 12, 21, 24, 25), which were part of every team's *training* data.
Different videos, different difficulty, 5 vs 8 cases. Treat Table 5 as a
direction, not a target line.

The internally comparable number is our own frame-wise linear probe on the same
5 videos. Val split, seed 0, defaults unless stated:

| Config | Metric | Macro-F1 | Edit |
|---|---|---|---|
| frame-wise linear probe | 0.1599 | 0.3060 | 0.0138 |
| **ARST, faithful (W=5, CCI on)** | **0.3349 ± 0.0473** | 0.3226 | **0.3472** |
| ARST `--no-cci` | 0.2875 ± 0.0454 | 0.3185 | 0.2565 |
| ARST `--width 0` | 0.3439 ± 0.0506 | 0.3288 | 0.3591 |
| ARST `--mask-excluded` | 0.4111 ± 0.0531 | 0.3804 | 0.4417 |

Training is cheap: **112 s** for all three stages on MPS, plus ~50 s inference
across the 5 val videos. The frozen feature cache is what buys that.

### These numbers are not bit-reproducible

Re-running the faithful config after the package restructure gave **0.3402 ±
0.0484** (macro-F1 0.3255, edit 0.3548, 1,249 leaked) against the 0.3349 ±
0.0473 in the table — same seed, same data, same code path.

The gap is 0.005, an order of magnitude below the ±0.048 per-video spread. MPS
reduction kernels are not bit-deterministic across runs, so `torch.manual_seed`
fixes the initialisation and the shuffle order but not the arithmetic. Treat the
table's third decimal as noise, and do not read a difference of <0.01 between
any two configs here as signal — the `--width 0` result in particular sits well
inside this band.

### What moved, and what did not

**The edit score went 0.0138 -> 0.3472, a 25x improvement. Macro-F1 barely
moved: 0.3060 -> 0.3226.** The entire gain is temporal consistency, which is
exactly what the architecture targets and exactly what a frame-wise model
cannot reach.

That split is also the clearest available evidence for the frozen-backbone
diagnosis in §4. Macro-F1 is a per-frame quantity, essentially a measure of how
separable the features are. Adding a strong temporal model on top left it
almost unchanged — so the per-frame ceiling is set by the features, not by the
classifier. Fine-tuning the backbone (roadmap 3.6, gated by 1.7) is where the
remaining F1 lives.

### Inference-time ablations, measured at fixed weights

The table above retrains for every row, so each number carries fresh MPS noise
on top of the effect being measured. Two of those rows ablate *inference*, not
training — `--no-cci` and `--mask-excluded` change no weight at all. Scoring one
checkpoint four ways (`uv run pitvis-eval`) isolates them properly:

| inference config | metric | Δ vs default | leaked |
|---|---|---|---|
| default (W=5, CCI on) | 0.3402 ± 0.0484 | — | 1,249 |
| `--no-cci` | 0.2937 ± 0.0473 | **−0.047** | 1,149 |
| `--mask-excluded` | 0.4405 ± 0.0457 | **+0.100** | 0 |
| `--no-cci --mask-excluded` | 0.3969 ± 0.0358 | +0.057 | 0 |

Same weights throughout, so these deltas are the ablation and nothing else.
Both effects are larger here than the retrain-based table suggests — masking is
worth +0.100 rather than +0.076 — and the two compose sub-additively: CCI is
worth +0.047 without masking but only +0.044 with it, which makes sense given
that both are attacking prediction noise.

Prefer this table when reasoning about inference choices, and the retraining
table only for things that actually change weights (`--width`).

### CCI earns its keep

Removing it costs 0.047 (0.3349 -> 0.2875), almost all of it edit score
(0.3472 -> 0.2565). Worth remembering that this is the fixed-lag component, so
the strictly-causal number is 0.2875.

### The banded mask did NOT reproduce

`--width 0` scores 0.3439 vs 0.3349 for the paper's W=5 optimum — slightly
*better*, and well inside the ±0.05 per-video spread. We do not reproduce
ARST Table 2's inverted-U on Cholec80.

Two plausible reasons, neither verified: (a) TeCNO's ~17-minute causal
receptive field already supplies all the temporal context, leaving the
transformer's attention span nearly irrelevant here, and (b) 5 validation
videos with a std of 0.05 cannot resolve a 0.009 difference. Do not tune W on
this split — it is not measuring anything.

### The failure mode inverted

Predicted segment counts against ground truth:

| video | true segs | linear probe | ARST |
|---|---|---|---|
| 1 | 78 | 2,679 (34x) | 57 (0.7x) |
| 12 | 84 | 1,612 (19x) | 39 (0.5x) |
| 21 | 182 | 2,391 (13x) | 44 (0.2x) |
| 24 | 78 | 1,145 (15x) | 31 (0.4x) |
| 25 | 59 | 1,477 (25x) | 27 (0.5x) |

The probe over-segmented by 13-34x. ARST now **under**-segments at 0.2-0.7x.
Auto-regression plus CCI made the model conservative about transitions, which
is the right trade for the edit score but introduces a new problem: whole short
or rare steps get absorbed and never predicted at all. Per-class recall shows
it — step 3 (septum displacement) and step 9 (synthetic graft placement) are
both at **0.000**, and step 6 (durotomy) at 0.036.

This is the next thing to attack, and it is a genuinely different problem from
the one we started with.

### The free win is real

`--mask-excluded` removes classes 0/11/13 from the argmax and gains **0.076**
(0.3349 -> 0.4111) — the single largest improvement measured, from a
three-line change. It works because the official metric filters by ground truth
only and calls `f1_score` with no `labels=`, so any prediction of an excluded
class joins the macro average at F1 = 0. The faithful run leaks 1,252 such
predictions onto scored rows.

TSO-NCT (2nd place) did the same thing — "any steps not considered for
evaluation were replaced with the most recent permitted step". It is a
scoring-rule exploit, not a modelling improvement, so both numbers are reported
and the headline stays the faithful one.
