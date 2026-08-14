# The CITI cascade, traced with dimensions

Every tensor shape between an `.mp4` on disk and a scored prediction. Companion
to `citi-baseline.md`, which covers *why* the architecture is what it is and
where we diverge from the paper; this one covers *what shape the data is* at
each hop.

All numbers here were read off the real artifacts, not copied from the code.
Video 01 is the running example (it is the first validation video).

---

## 0. The cascade at a glance

```
26531686/video_01.mp4          172,812 frames @ 24 fps, 1280x720 H.264
        │
        │  ffmpeg  select=not(mod(n,24))          1-fps decimation
        ▼
  7,201 RGB frames             (720, 1280, 3) uint8   each
        │
        │  timm transform                          resize/crop/normalise
        ▼
  7,201 tensors                (3, 224, 224) float32  each
        │
        │  frozen resnet50, num_classes=0          global-pooled
        ▼
  features.npy                 (7201, 2048) float32   ← THE CACHE BOUNDARY
  labels.npy                   (7201,)      int64
        │
        │  standardise (train-split mean/std)
        ▼   ══════════════ stages below are trained ══════════════
  STAGE 1  SpatialEmbedding    (7201, 2048) → (7201, 512)      Z_t
        ▼
  STAGE 2  TeCNO               (1, 7201, 512) → (1, 7201, 512)  F_t
        ▼
  STAGE 3  ARST                (1, T, 512) + (1, T) → (1, T, 15)
        ▼
  argmax → preds (7201,) int64
        ▼
  official metric, per video, then mean±std
```

The cache boundary is the important line. Everything above it runs once, ever.
Everything below runs in 112 seconds. That asymmetry is the entire reason the
pipeline is shaped this way.

---

## 1. Raw video → frames

`data/extract_features.py` shells out to `ffprobe` for two numbers, then to `ffmpeg`
to decode.

```
ffprobe -select_streams v:0 -count_packets \
        -show_entries stream=r_frame_rate,nb_read_packets
```

Real output for four videos, with the frame-count arithmetic carried through:

| video | packets | fps | `T = ceil(packets/fps)` | annotation rows | `rows == T+1` |
|---|---|---|---|---|---|
| 01 | 172,812 | 24 | 7,201 | 7,202 | ✓ |
| 20 | 207,475 | 24 | 8,645 | 8,646 | ✓ |
| 24 | 191,217 | **25** | 7,649 | 7,650 | ✓ |
| 25 | 104,088 | 24 | 4,337 | 4,338 | ✓ |

Two things this table pins down:

**fps is read per video, never assumed.** Video 24 is 25 fps; every other video
is 24. `select=not(mod(n,25))` for that one, `mod(n,24)` for the rest. Hardcoding
24 would silently produce 7,969 frames for video 24 against 7,650 labels — a
4% temporal stretch, and the labels would drift further out of alignment the
deeper into the operation you go.

**Resolution is probed too, and the raw pipe is sized from it.** All 25 challenge
videos are 1280x720 — `1280*720*3 = 2,764,800` bytes per frame — but the size is
read per video rather than hardcoded, because `embed_video` also serves
`pitvis-predict` on arbitrary files. A wrong frame size desynchronises the pipe
and corrupts every frame after the first, so this is not a constant worth
assuming.

### The off-by-one

Annotation rows are always exactly `T+1`. The extra row is the last second, for
which no frame exists (a 172,812-frame video at 24 fps is 7,200.5 seconds long;
you get 7,201 sampled frames indexed 0..7200, and 7,202 annotation rows).

`data/extract_features.py:207-209` handles it by assertion, not by trust:

```python
assert len(steps) == expected + 1
assert steps[-1] == -1           # dropped row must be background
labels = steps[:expected].copy()
```

Every video ends in a background run of 6-147 seconds, so the dropped row is
verifiably background in all 24 cases. This is why the labeled corpus is
**115,562** frames rather than the 115,586 annotation rows quoted in
`CLAUDE.md` — exactly 24 rows, one per video.

---

## 2. Frame → 2,048-d embedding

The transform is not hand-written; it is whatever `timm` resolves for the
checkpoint, so the features match the weights' training distribution. Freshly
resolved:

```
input_size:    (3, 224, 224)
interpolation: bicubic
crop_mode:     center
crop_pct:      0.95
mean:          (0.485, 0.456, 0.406)
std:           (0.229, 0.224, 0.225)
```

So a `(720, 1280, 3)` uint8 frame becomes a `(3, 224, 224)` float32 tensor.
Note the aspect ratio is **not** preserved — a 16:9 endoscopic frame is squared
to 1:1. That is what the ImageNet checkpoint expects, and matching the
checkpoint matters more than preserving geometry when the backbone is frozen.

`num_classes=0` removes the classifier head, so `model(x)` returns the pooled
final-block activations:

```
(64, 3, 224, 224)  →  (64, 2048)      batch of 64
```

**2,048 is not a chosen number.** It is ResNet-50's last-block channel count,
fixed by the architecture. Global average pooling collapses the final `(2048, 7,
7)` feature map over its spatial dimensions, leaving one number per channel.
Every frame becomes 2,048 floats regardless of what is in it.

### What lands on disk

```
data/features/video_01/features.npy   (7201, 2048) float32   59.0 MB
data/features/video_01/labels.npy     (7201,)      int64
```

Labels use the 15-way encoding: raw `-1` → `0`, raw `k` → `k`. Verified range
on video 01 is `[0, 14]`.

Whole cache: **115,562 frames × 2,048 float32 = 0.95 GB** (939 MB on disk
across 25 video directories — video 19 has features but no labels).

> **Cache provenance.** `data/features/manifest.json` records the feature-space
> content hash (`67912d3efc6852e7`) and per-video provenance. `extract_features.py`
> refuses to write into a cache whose manifest describes a different feature
> space, which is what makes a backbone swap safe rather than silently corrupting.
> `uv run pitvis-verify` checks the whole cache against it.

---

## 3. Cache → split

`data/dataset.py` loads per video and never concatenates across the split boundary.

```python
VAL   = [1, 12, 21, 24, 25]
TRAIN = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,20,22,23]
```

| split | videos | frames | as fraction |
|---|---|---|---|
| TRAIN | 19 | 84,666 | 73.3% |
| VAL | 5 | 30,896 | 26.7% |

Ratio is **2.74:1**, not the ~4:1 the paper's split intends — partly because
video 19's annotations are missing from the download, partly because the five
validation videos happen to be long ones (7,201 / 4,942 / 6,767 / 7,649 /
4,337).

Standardisation statistics come from the train split only and are computed once
in `training/arst.py:275`:

```
X = concat over TRAIN            (84666, 2048)
mean, std                        (2048,), (2048,)     per-channel
```

Saved to `data/arst/standardize.npz`. Per-channel, not global — each of the
2,048 ResNet channels has its own activation scale.

---

## 4. Stage 1 — SpatialEmbedding

```
in       (7201, 2048)     standardised cached features
project  Linear(2048, 512)
out z    (7201, 512)                                    ← Z_t, kept
classify Linear(512, 15)   on relu(z)
out      (7201, 15)                                     ← training head, discarded
```

**1,056,783 parameters.** Trained frame-wise on shuffled frames (batch 1,024),
SGD at 1e-4, 20 epochs. The classifier head exists only to give the projection a
gradient; after training, `embed()` keeps `z` and throws the logits away.

This is the stage where our reproduction departs from the paper. ARST fine-tunes
the whole ResNet-50 here for 50 epochs. We train a single `2048 → 512` linear map
on frozen features, because extraction discarded the pixels. See
`citi-baseline.md` §4.

---

## 5. Stage 2 — TeCNO

Two cascaded causal TCNs. The shape story here is mostly about layout:

```
in  z            (1, 7201, 512)     (B, T, C)
    .transpose   (1, 512, 7201)     (B, C, T)   ← Conv1d wants channels-first
stage1 → logits  (1, 7201, 15)
       → softmax (1, 15, 7201)      feeds stage 2
stage2 → logits  (1, 7201, 15)      supervised too
       → hidden  (1, 7201, 512)     ← F_t, this is what ARST consumes
```

**17,079,838 parameters** — by far the largest stage, two thirds of the model.

Note stage 2's input channel count is **15, not 512**: it consumes stage 1's
class posteriors, not its features. That is the TeCNO idea — the second stage
refines a distribution over phases rather than re-processing appearance. Both
stages are supervised (`loss = sum over stages`), which is why `forward` returns
a list.

### Receptive field

Each stage has 8 dilated layers, dilations 1, 2, 4, …, 128, kernel 3, causal
(left-padded only):

```
per stage:   1 + 2*(1 + 2 + ... + 128) = 511 frames  =  8.5 min @ 1 fps
two stages:  511 + 511 - 1             = 1,021 frames = 17.0 min @ 1 fps
```

Seventeen minutes of surgical history is baked into every `F_t`. Hold onto that
for the next section — it is the reason the transformer on top can afford to be
almost myopic.

---

## 6. Stage 3 — ARST

```
                    feats (1, T, 512)              prev (1, T) int64
                          │                              │
                    PositionalEncoding            SegmentedPhaseEmbedding
                          │                              │  (16, 512) table
                          │                        (1, T, 512)
                          │                              │
                          │                        PositionalEncoding
                          ▼                              ▼
                  EncoderLayer  ──── K,V ────→  DecoderLayer
                  (banded mask)                 self-attn (banded)
                          │                     cross-attn (Q from decoder)
                    mem (1, T, 512)                      │
                                                   (1, T, 512)
                                                         │
                                                  Linear(512, 15)
                                                         ▼
                                                   (1, T, 15)
```

**7,364,111 parameters.** One encoder layer, one decoder layer, `d_model=512`,
8 heads (64-d each), feed-forward 2,048.

Measured on a 1,024-frame chunk: `encode` → `(1, 1024, 512)`, `decode` →
`(1, 1024, 15)`.

### The two label tensors

This trips people up, so both are printed here from video 01:

```
labels  y        (1, 7201)    ground truth
shifted prev     (1, 7201)    first 5 = [15, 0, 0, 0, 1]
```

`shift()` prepends a start-of-sequence symbol and drops the last label, so the
decoder at position `t` sees the label for `t-1`. SOS is index **15** — one past
the 14 real classes — which is why the embedding table is `(16, 512)` and not
`(15, 512)`.

### Segmented phase embedding

Not one-hot. The 512-d vector is cut into 15 segments of `512 // 15 = 34`; phase
`i` sets its whole segment to 1. The final **2** dimensions are always zero, and
SOS (row 15) is all zeros.

The point is distance. Two one-hot phases differ in 2 coordinates; two segmented
phases differ in **68**. The decoder gets a far stronger signal about which
phase it was just in. It is a fixed buffer, never learned.

### The banded mask — the load-bearing shape

Real output of `banded_causal_mask(8, width=5)`, 1 = attention permitted:

```
        key: 0  1  2  3  4  5  6  7
query 0      1  0  0  0  0  0  0  0
      1      1  1  0  0  0  0  0  0
      2      1  1  1  0  0  0  0  0
      3      1  1  1  1  0  0  0  0
      4      1  1  1  1  1  0  0  0
      5      1  1  1  1  1  1  0  0
      6      0  1  1  1  1  1  1  0
      7      0  0  1  1  1  1  1  1
```

Upper triangle is zero — that is ordinary causality, required by the challenge
rule. But look at the **lower-left corner**: query 7 cannot see key 0 either.
The band is 6 wide (`t-5 … t`) and slides forward. Position 7 has forgotten
position 0 entirely.

That looks like a bug until you connect it to §5. `F_t` already carries 17
minutes of causal context from TeCNO. The transformer is not the component doing
long-range modelling — it is doing **short-range transition modelling on top of
an already-temporal feature**. Widening `W` does not add context; it adds noise
from a past that `F_t` has already summarised.

(Our own ablation did not reproduce the paper's inverted-U here — `--width 0`
scored marginally *better*. See `citi-baseline.md` §6; five videos cannot
resolve that difference.)

---

## 7. Where T stays whole and where it gets cut

This is the part most likely to hide a bug, and the shapes differ between
training and inference. Worth its own table.

| stage | training | inference |
|---|---|---|
| Spatial | frames shuffled, batch 1,024 — T meaningless | whole video, `(T, 2048)` |
| TeCNO | **whole video**, `(1, T, 512)` | whole video |
| ARST encoder | 1,024-frame chunks | 1,024-frame chunks, **W-overlapped** |
| ARST decoder | teacher-forced, whole chunk at once | **one frame at a time** |

Three separate reasons:

**TeCNO is never chunked.** It is convolutional, so an 8,645-frame video costs
O(T) memory. Chunking would truncate the 1,021-frame receptive field at every
boundary — precisely the thing that makes the stage worth having.

**ARST is chunked because attention is O(T²).** The longest video is 8,645
frames; an `8645 × 8645 × 8`-head attention matrix is ~2.4 GB. At chunk 1,024,
only the first `W=5` queries of each chunk lose context — about 0.5% of
positions.

**Inference chunking is exact, unlike training.** `encode_memory` overlaps
consecutive chunks by `W` frames and keeps only the non-overlapped output:

```python
lo = max(0, s - W)                 # overlap carries the band
out = model.encode(f[:, lo:hi], offset=lo)
mem[:, s:hi] = out[:, s - lo:]
```

Because the mask is banded at width `W`, a query needs only the `W` keys behind
it — so this is bit-identical to encoding the full sequence. The training-time
approximation is deliberate and bounded; the inference-time one is not an
approximation at all.

The `offset` argument threaded through `encode`/`decode`/`PositionalEncoding`
exists for exactly this reason: a chunk starting at absolute frame 3,072 must
receive positional encodings 3,072…, not 0…. Dropping `offset` would be a silent
correctness bug that only shows up as slightly degraded long-video accuracy.

### The decoder rollout

Training runs the decoder over a whole chunk in one pass, because teacher forcing
supplies all the shifted ground-truth labels up front. Inference cannot — the
input at `t` is the model's own prediction at `t-1`. So `cci_decode` walks
forward one frame at a time:

```
for t in 0..T-1:
    logits = decode(mem[:, t-W : t+1], prev[:, t-W : t+1])[0, -1]   # (15,)
    p = argmax(logits)
    prev[0, t+1] = p
```

The decoder input window is `(1, ≤6)` — never the full prefix, because the band
makes anything older irrelevant. That is what keeps a sequential rollout over
8,645 frames tractable.

CCI adds a lookahead on top: at a predicted transition, it probes up to `n=10`
frames forward while continuing to assert the *old* phase, and accepts the
transition only if all 10 agree. This makes the system fixed-**lag** rather than
strictly causal — frame `t` is finalised after observing `t+10`. `--no-cci`
gives the strict version.

---

## 8. Prediction → score

```
preds (7201,) int64 ──┐
truth (7201,) int64 ──┴─→ [(vid, truth, pred), ...]  5 tuples
                              │
                              │  per video, never concatenated
                              ▼
                     evaluation/official.py  (vendored verbatim)
                              │
                     (macro-F1 + normalised edit) / 2
                              │
                              ▼
                       mean ± std over 5 videos
```

Rows whose **ground truth** is in `{0, 11, 13}` are dropped before scoring. Note
the filter is on ground truth only — a *prediction* of an excluded class on a
retained row survives into the macro average at F1 = 0. That asymmetry is what
`--mask-excluded` exploits, worth +0.076. See `citi-baseline.md` §6.

Classes 11 and 13 are absent from the validation split entirely, so the scored
label set is 12 classes wide.

---

## 9. Dimension reference

| symbol | shape | dtype | what |
|---|---|---|---|
| raw frame | `(720, 1280, 3)` | uint8 | decoded at 1 fps |
| transformed | `(3, 224, 224)` | float32 | timm resize/crop/normalise |
| cached feature | `(T, 2048)` | float32 | frozen ResNet-50, pooled |
| labels | `(T,)` | int64 | 15-way, 0 = background |
| `Z_t` | `(B, T, 512)` | float32 | spatial embedding |
| `F_t` | `(B, T, 512)` | float32 | TeCNO stage-2 hidden |
| phase table | `(16, 512)` | float32 | 15 classes + SOS, fixed |
| `prev` | `(B, T)` | int64 | shifted labels/predictions |
| logits | `(B, T, 15)` | float32 | final |
| mask | `(L, L)` | float32 | 0 / -inf, band width 5 |

Constants: `D_MODEL=512`, `N_HEADS=8` (64-d each), `BAND_WIDTH=5`,
`TCN_LAYERS=8`, `CCI_N=10`, feed-forward 2,048, `NUM_CLASSES=15`.

Parameters:

| stage | params | share |
|---|---|---|
| SpatialEmbedding | 1,056,783 | 4.1% |
| TeCNO | 17,079,838 | 67.0% |
| ARST | 7,364,111 | 28.9% |
| **total** | **25,500,732** | |

Plus the frozen ResNet-50 (~23.5 M) which is never trained and never loaded at
training time — it exists only in the cache it produced.

---

## 10. The open thread

The trace makes one thing conspicuous. Between the raw frame and `features.npy`
there is a `(720, 1280, 3)` → `(2048,)` reduction — a factor of ~1,350 — and it
is performed by a network that has never seen an endoscope. Everything below the
cache boundary is 25.5 M trainable parameters operating on the residue of that
one frozen decision.

That is consistent with what the results show: adding TeCNO + ARST moved the edit
score 25x (0.0138 → 0.3472) but left macro-F1 nearly flat (0.3060 → 0.3226).
Temporal structure was recoverable from the cache; per-frame discriminability was
not, because it was already discarded upstream.

Two directions follow, and they are different kinds of work:

1. **Re-open the cache boundary** (roadmap 1.7 → 3.6). Fine-tuning the backbone
   needs a pixel path, which means extraction has to stop being a one-way door.
   This is where the remaining macro-F1 lives.
2. **Fix the under-segmentation.** ARST now predicts 0.2-0.7x too *few* segments,
   with steps 3 and 9 at 0.000 recall — the failure mode inverted from the linear
   probe's 13-34x over-segmentation. This is a decoder/CCI problem and needs no
   new features.

Which of those is worth attacking first depends on whether you care more about
the headline metric or about the model actually finding the rare steps.
