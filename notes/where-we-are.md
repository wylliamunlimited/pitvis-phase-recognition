# Where we are — orientation snapshot

*Written 2026-08-09. Read this when you have lost the thread. It is a status
document, not a reference one: the layered notes (`walkthrough.md`,
`citi-baseline.md`, `instruments.md`, and the two `*-variants.md`) stay the
source of truth for detail. Numbers here were current on the date above.*

---

## 1. The vocabulary, because most of the confusion lives here

There are **24 usable videos** (video 19 has no annotations), split once and
permanently:

```
TRAIN = 19 videos   [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,20,22,23]
VAL   =  5 videos   [1, 12, 21, 24, 25]
```

**VAL is the exam.** Nothing is ever trained on it. Das et al. benchmark these
exact five videos in Table 8 (CITI 70 on steps, SANO 81 on instruments), so it
is also the only number comparable to the paper.

**Why we cannot just use VAL to pick models.** Five videos give a per-video
spread of about ±0.05 — larger than most differences we are trying to detect.
And choosing a winner by looking at VAL turns VAL into training data: you would
be reporting the best of six noise draws, not the best model.

**CV (cross-validation)** is how variants get ranked without touching VAL.
Split the 19 TRAIN videos into 5 folds. Train on 15, score the 4 held back,
rotate five times. Every training video ends up scored exactly once by a model
that never saw it — 19 scores instead of 5. VAL is then scored **once**, for
the winner only.

The folds are frozen literals in `src/pitvis/data/folds.py`, chosen so no fold
leaves a rare class with zero training examples.

**"Honest CV"** means *nothing in the pipeline* saw the held-out videos. Which
is exactly what broke — see §5.

---

## 2. What the AP probe is

Not a model. A **diagnostic**, `uv run pitvis-probe`.

Six instrument classes were never predicted at all. Two very different causes:

- the features contain the information and the decision rule discards it
  → fix the loss or the threshold
- the encoder cannot see the instrument at all
  → fix the encoder

The headline metric cannot distinguish these, because it only ever reports
*decisions*. **Average precision** can: it is computed from the ranking, so it
is independent of both the decision threshold and how rare the class is. AP
near the class's base rate means no signal; far above means signal that
something downstream is throwing away.

It settled the question:

| class | training instances | AP on frozen DINOv2 |
|---|---|---|
| tissue glue | 282 | **0.767** |
| cup forceps | 1,635 | **0.055** |

Tissue glue is *rarer* and nearly separable. So rarity was never the problem —
visibility to the encoder was. That is what justified fine-tuning the backbone.

---

## 3. Three iterations

| | what changed | task 1 (steps) | task 2 (instruments) |
|---|---|---|---|
| start | published reproductions | 0.3425 | 0.2321 |
| **iter 1** | training fixes + backbone swap | — | **0.5572** |
| **iter 2** | same recipe applied to steps | **0.4610** | — |
| **iter 3** | fine-tune the encoder itself | *no valid number yet* | *no valid number yet* |

Each iteration changed one thing at a time against a `control`, so a delta is
attributable.

**Iteration 1 — instruments.** `control` (SANO unchanged) → `weighted` (class
weights in the loss) → `thresholds` (one decision threshold per class instead
of a flat 0.5) → `dinov2` (a better frozen encoder) → `best` (composed).
Winner: weighted + per-class thresholds on DINOv2. Classes never predicted went
**9/19 → 0/19**.

**Iteration 2 — steps.** `control` (ARST unchanged) → `masked` (remove classes
0/11/13 from the argmax) → `weighted` → `dinov2` → `best`. Winner: masking +
class weights on DINOv2. `masked` alone was the largest single lever and is not
a model change at all — the official metric filters by ground truth only, so
predicting an excluded class joins the macro average at F1 = 0.

**Iteration 3 — the encoder.** Everything above rides a *frozen* encoder that
has never seen an endoscope. Both papers fine-tune theirs first. We now can
too: `pitvis-frames` writes pixels to disk, `pitvis-finetune` trains the
backbone on them. A 5-epoch pilot moved **mean AP 0.271 → 0.445 with 19 of 19
classes improving**. No end-to-end number yet — see §5.

### The finding that recurred in both tasks

**Never test a backbone swap first.** DINOv2 alone gained +0.021 macro on
instruments and +0.029 on steps — both inside the fold spread, both looking
like nothing. On both tasks it became the best variant *only after* the loss
and decision rule were fixed. The representation gain is real and it is masked
by the imbalance defect. Testing the encoder first and stopping at the null
would have retired a true hypothesis, twice.

---

## 4. The architecture we landed on

**Task 1 — steps** (CITI's ARST):

```
frame → encoder → 768-d → Linear → 512-d
      → TeCNO      two stacked causal TCNs, ~17 minutes of history
      → ARST       1-layer transformer, auto-regressive over its own past labels,
                   banded causal mask (W=5)
      → argmax with classes 0/11/13 masked, then consistency-constraint inference
      → one step per second
```

**Task 2 — instruments** (SANO's LSTM):

```
frame → encoder → 768-d
      → 5-frame causal window
      → 2-layer unidirectional LSTM → Linear(512, 19) → sigmoid
      → per-class thresholds, capped at 2 by margin
      → up to two instruments per second
```

Both trained with class weighting, both currently reading DINOv2 features. Both
are strictly causal — the challenge permits online models only.

---

## 5. The leak, and why iteration 3 has no number

The backbone was fine-tuned on **all 19 TRAIN videos, using their step and
instrument labels**, reaching 0.944 frame accuracy — substantially memorising
them. Cross-validation then held out videos *from that same set*, so a
"held-out" video's features came from an encoder that already knew its answers.

Steps macro read **0.917** and instruments weighted **0.890**. Those are the
size of the leak, not of an improvement. Both entries were deleted rather than
kept with a caveat, and `crossval.check_no_leak` now refuses the configuration
and names the fix.

**What survives: the AP probe is clean.** It fits on TRAIN and scores on VAL,
and the backbone never saw VAL. So `0.271 → 0.445` stands.

**What honest evaluation costs.** A proper CV over fine-tuned features needs
**one encoder per fold**, each trained with that fold's videos excluded, plus
one on all of TRAIN for the VAL scoring. Six fine-tunes — ~57 h on this laptop
at 50 epochs, a few hours on a rented GPU. That is what `infra/` is for.

---

## 6. The modules, and how they chain

```
pitvis-frames     video      → JPEG frames on disk        (~19 min, 3.6 GB)
pitvis-extract    frames|mp4 → feature vectors            (~20-25 min)
pitvis-finetune   frames     → a surgical-specific encoder  ← the GPU-heavy one
pitvis-probe      features   → per-class AP diagnostic
pitvis-train      features   → a temporal model           (~2 min)
pitvis-eval       score a checkpoint without retraining
pitvis-predict    mp4        → per-second predictions
pitvis-app        watch a case play beside the model
```

They chain `frames → extract → train → predict/app`. Fine-tuning inserts
*before* extract; everything downstream is unchanged and still trains in
minutes. That is the whole design: the expensive stage is paid once.

Two registries, deliberately separate:

- `pitvis-train --list` — what can be trained
- `pitvis-predict --list-models` — what *has* been trained, and where it lives

```sh
uv run pitvis-predict --video V --steps-model arst-v2:best
uv run pitvis-predict --video V --instruments-model instruments-v2:weighted
```

---

## 7. Picking this up on another machine

Code is in git. Data is not — `data/`, `26531686/` and `predictions/` are
gitignored, deliberately (the dataset is CC BY-NC-ND and must not be
redistributed).

| artifact | size | cost to regenerate |
|---|---|---|
| `data/backbone/` | 96 MB | **62 min** |
| `data/features/` (3 spaces) | 2.2 GB | ~75 min |
| `data/arst/` + `data/instruments/` | 461 MB | ~15 min |
| `data/frames/384/` | 3.6 GB | 19 min |
| `26531686/` raw video | **40 GB** | hours to re-download |

**The fine-tuned backbone is the one worth carrying** — 96 MB for an hour of
training. Copy `data/` (6.3 GB total) to a private bucket or an external drive.

You only need the 40 GB of video for `pitvis-app` (it streams the mp4),
`pitvis-frames`, or extracting a new backbone. Training and evaluation need
**features only**.

After copying, confirm nothing corrupted in transit:

```sh
uv sync
uv run pitvis-verify --space resnet50
uv run pitvis-verify --space dinov2_vitb14
uv run pytest                      # 133 tests
```

---

## 8. What to do next

**Option A — get an honest number today, ~10 min.** The pilot backbone was
trained on TRAIN only, and VAL is disjoint from TRAIN, so scoring it on VAL is
legitimate. It spends our one clean VAL touch for this backbone.

```sh
uv run pitvis-train instruments-v2 --variant best --space resnet50_ft
uv run pitvis-train arst-v2       --variant best --space resnet50_ft
```

Compare against the current bests: **0.5572** (instruments, official) and
**0.4610** (steps, challenge metric).

**Option B — the cloud job.** Six fine-tunes at 50 epochs, which makes honest
CV possible on fine-tuned features. See `infra/README.md` for the commands, the
cost table, and the licence constraint (private bucket only).

**Do A before B.** If a 5-epoch pilot already beats the current bests on VAL,
that justifies the cloud spend. If it does not move, you have saved the bill.

### Known open threads

- Four instrument classes are *emitted* but not *usable* — bipolar forceps
  gets 320 predictions for 49 true instances. "0 never predicted" flatters.
- Per-fold CV needs a harness change: each fold must read features from **its
  own** encoder, and `--space` names one space for the whole run.
- Task 1 has no per-class-threshold analogue — steps use a single argmax rather
  than 19 independent sigmoids. A prior-corrected argmax is untried.
