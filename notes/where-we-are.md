# Where we are — orientation snapshot

*Snapshot: 2026-08-16. Read this first after time away, then follow the links.*

**For the system as it stands** — what is trained, what is wired to what, how
a case flows through the app, and what may be claimed in a demo — see
[`current-state.md`](current-state.md). This note is the *research* view.

**This note owns two things and nothing else: the vocabulary, and what to do
next.** Everything else here is a pointer. Results live in the iteration notes,
architectures in the reproduction notes, commands in the README. If you find
yourself restating one of those here, link instead — a scoreboard maintained in
four files is a scoreboard that goes stale in three.

---

## 1. The vocabulary, because most of the confusion lives here

There are **24 usable videos** (video 19 has no annotations), split once and
permanently — the lists are in `data/dataset.py` and `CLAUDE.md`:

```
TRAIN = 19 videos        VAL = 5 videos  [1, 12, 21, 24, 25]
```

**VAL is the exam.** Nothing is ever trained on it. Das et al. benchmark these
exact five videos in Table 8, so it is also the only number comparable to the
paper.

**Why we cannot pick models on VAL.** Five videos give a per-video spread of
about ±0.05 — larger than most differences we are trying to detect. Choosing a
winner by looking at VAL turns VAL into training data: you are then reporting
the best of six noise draws, not the best model.

**CV** ranks variants without touching VAL. The 19 TRAIN videos are split into
5 frozen folds; train on 15, score the 4 held back, rotate. Every training video
is scored exactly once by a model that never saw it — 19 scores instead of 5.
VAL is then scored **once**, for the winner only.

**"Honest CV" means nothing in the pipeline saw the held-out videos** — the
encoder included. That is the distinction that broke in iteration 3, and it is
the whole reason `infra/` exists.

→ full protocol and the `zero_division=1` caveat:
[`instrument-variants.md` §2](models/instrument-variants.md)

---

## 2. Where we got to

| | what changed | steps | instruments (official / macro) |
|---|---|---|---|
| start | published reproductions | 0.3425 | 0.2321 / 0.2556 |
| iter 1+2 | loss, decision rule, DINOv2 | 0.4610 | **0.5572** / 0.3792 |
| iter 3 | fine-tuned ResNet-50 encoder | 0.4425 | 0.3805 / 0.4783 |
| iter 4a | fine-tuned DINOv2 — bad recipe | 0.3500 | 0.2803 / — |
| **iter 4b** | fine-tuned DINOv2 — fixed recipe | **0.5608** | 0.3220 / **0.5333** |

Steps are the challenge metric, instruments the official one, both on VAL.
Instruments carry a second column because the official number and the honest
one disagree — see below.

**Iteration 4b is the largest single gain in the project.** Steps +0.0998 on
the challenge metric outright, which is more than everything in iterations 1+2
combined. Eleven of twelve scored step classes improve and durotomy comes back
from 0.000 F1 to 0.573 — a class the frozen encoder never once predicted
correctly.

**On instruments the same encoder wins every video and the official metric says
it lost.** Macro +0.154 and name-aligned weighted +0.103, both several times the
spread; the official number falls 0.235. That is not a trade-off, it is the
vendored `MultiLabelBinarizer` column defect: on the two videos where the frozen
model's predicted class set happened to match the truth's, official and aligned
agree to four decimals, and those two videos carry its whole 0.5572 mean. The
fine-tuned model never gets that coincidence, so all five videos are penalised.
Aligned, it wins 5 of 5. Full per-video breakdown in
[`instrument-variants.md`](models/instrument-variants.md).

**Iteration 3 did not win, and the reason is interesting rather than
disappointing.** On the *primary* metric it does win instruments — macro F1
0.3792 → **0.4783** — while the support-dominated official number falls. It is
better on the rare classes and worse on the four carrying 91% of positives. On
steps it is a wash overall but trades edit score (−0.061) for macro (+0.024):
the encoder was fine-tuned frame-by-frame with no temporal term, so it names
seconds slightly better and holds segments together worse.

Both arms are **single VAL measurements**, not a ranking — a CV over
`resnet50_ft` is unavailable because one encoder trained on all of TRAIN leaks
into every fold. Treat it as a reason to fine-tune DINOv2 and cross-validate
properly, not as a verdict.

**Iteration 4 fine-tuned DINOv2 itself, twice. The first attempt was the
clearest negative result in the project; the second, after fixing the recipe,
is the clearest positive one.** Same encoder, same data, same augmentation,
same heads — only the optimisation recipe changed.

| | mean AP | classes improved |
|---|---|---|
| frozen DINOv2 | 0.350 | — |
| run 1 — uniform 1e-4, 50 epochs | 0.270 | 3 / 19 |
| **run 2 — 1e-5, layer decay, warmup, early stop @ epoch 2** | **0.523** | **19 / 19** |

Surgical drill went 0.023 → 0.470, which is the clearest single sign the
original diagnosis held: the information was in the pixels and the frozen
encoder could not represent it.

**The AP probe was the falsifier, fixed before the run, and it predicted the
downstream result correctly** — unlike the ResNet-50 case, where 0.445 AP still
lost end to end. That is the one methodological thing worth carrying: an AP
gain this large (+0.172, all 19 classes) carried; a moderate one did not.

Historical, for the contrast: run 1 overwrote a representation better than
84,666 frames can teach — 50 epochs at a uniform lr=1e-4 with no validation
anywhere in the loop. Full account, both runs, in
[`instrument-variants.md` §6](models/instrument-variants.md).

- what was tried, what each variant tested, per-class movement —
  [`step-variants.md`](models/step-variants.md),
  [`instrument-variants.md`](models/instrument-variants.md)
- the reproductions those improve on —
  [`citi-baseline.md`](models/citi-baseline.md), [`instruments.md`](models/instruments.md)
- the architectures as shape traces — [`citi-dataflow.md`](reference/citi-dataflow.md)
- the cross-task finding worth carrying (**never test a backbone swap first**) —
  [`roadmap.md`](roadmap.md#the-finding-worth-carrying-forward)
- the diagnostic that says the encoder is the next lever —
  [`instrument-variants.md` §6](models/instrument-variants.md)

### Two things exist now that are not model work

Both are finished enough to use and neither moves a number, so they sit outside
the table above rather than in it.

**The review surface.** `uv run pitvis-app` plays a case beside the model's
output — the step burned into the frame corners PACS-fashion, a fourteen-row
procedure worklist, an instrument usage record, one progress strip. It reads as
clinical software rather than a video editor, and the reasoning for every part
of that is in [`app.md`](surfaces/app.md). It is also the only thing here that needs the
40 GB of video.

**Serving without Python.** The step cascade exports to ONNX and runs from a
Rust binary, verified **exactly per second** — 4337 of 4337 on video_25.
[`deployment.md`](surfaces/deployment.md) covers where the graph is cut and why, and is
honest about what is not done: task 2 is exported but unserved, and the input is
still a feature blob rather than pixels.

### Why iteration 3 has no number

The backbone was fine-tuned on all 19 TRAIN videos **with their labels**,
reaching 0.944 frame accuracy. Cross-validating over folds drawn from that same
set then held out videos whose features already encoded their answers. Steps
macro read 0.917 and instruments 0.890; those are the size of the leak, not of
an improvement. Both entries were deleted rather than kept with a caveat, and
`crossval.check_no_leak` now refuses the configuration.

The AP probe survives it — it fits on TRAIN and scores VAL, which the backbone
never saw.

→ what an honest version costs, and the job that pays it:
[`infra/README.md`](../infra/README.md)

---

## 3. Picking this up on another machine

Code is in git. Data is not — `data/`, `26531686/` and `predictions/` are
gitignored, deliberately: the dataset is CC BY-NC-ND and must not be
redistributed.

| artifact | size | cost to regenerate |
|---|---|---|
| `data/backbone/` | 96 MB | **62 min** |
| `data/features/<space>/` | ~350 MB–940 MB each | ~25 min each |
| `data/arst/` + `data/instruments/` | 461 MB | ~15 min |
| `data/frames/384/` | 3.6 GB | 19 min |
| `data/onnx/` | small | seconds — a build artifact, never carry it |
| `26531686/` raw video | **40 GB** | hours to re-download |

Four spaces are *defined* (`resnet50`, `resnet50_ft`, `dinov2_vitb14`,
`dinov2_ft`); how many are *extracted* is per machine — `ls data/features/`.
A machine with the ResNet-50 cache only will train and score, but reproduces
the 0.34 / 0.23 reproductions rather than the current bests, which live in
**`dinov2_ft`** — and that space cannot be regenerated from video alone: it
needs `data/backbone/dinov2_ft/backbone.pt`, which is an L4-hour away.

**The fine-tuned backbones are the artifacts worth carrying**, for exactly that
reason — 96 MB standing in for GPU time this machine does not have. Copy
`data/` (6.3 GB) to a private bucket or an external drive.

You only need the 40 GB of video for `pitvis-app`, `pitvis-frames`, or a new
extraction. Training and evaluation need **features only**.

After copying, confirm nothing corrupted in transit:

```sh
uv sync
uv run pitvis-verify --space resnet50
uv run pitvis-verify --space dinov2_vitb14
uv run pytest
```

→ the full command surface: [README](../README.md#usage)

---

## 4. What to do next

**The encoder question is answered.** Options A and B are both done, and the
fixed-recipe rerun (iter 4b) won on both tasks. `dinov2_ft` is now the best
feature space in the repo — by AP, by step metric, and by every defect-free
instrument metric.

**What is live now**, in the order I would take them:

1. **Cross-validate `dinov2_ft` honestly** — the one thing standing between
   iteration 4b and a defensible ranking. Every number in §2 for it is a single
   VAL measurement, because one encoder trained on all of TRAIN leaks into every
   fold and `crossval.check_no_leak` refuses the configuration. The honest
   version is `STAGE=all`: six fine-tunes, ~23 h on an L4, ~6× the cost of the
   run that produced the current encoder. It also needs a **harness change
   first** — each fold must read features from *its own* encoder, which a single
   `--space` cannot express. That change is local and free; write it before
   renting anything. See [`infra/README.md`](../infra/README.md).
2. **Ensembling** — now the cheapest untried lever, and the plan is already
   written with its falsifier fixed:
   [`ensembling-plan.md`](models/ensembling-plan.md). Note its §1 premise
   ("the encoder: two negatives") is now out of date; `dinov2_ft` is a member
   worth having rather than a cautionary tale.
3. **Roadmap 6.5** — four instrument classes are emitted but not usable
   (bipolar forceps: 320 predictions against 49 true instances).
4. **Roadmap 6.8** — a prior-corrected argmax for steps. Tried and it lost
   (`step-variants.md` §8); listed here only so it is not retried by accident.

**One repo defect surfaced while writing this up:** `pitvis-eval` has no
`--space` flag, so it always loads `resnet50` features and cannot score any
checkpoint trained on another space. Every result above was produced through
`pitvis-train`, which does take `--space`, so nothing is wrong with the numbers
— but the standalone scorer is unusable on four of the five checkpoints in
`data/arst/`.

The same list against `roadmap.md`, for anyone reading it alongside:

| item | state |
|---|---|
| **3.6b** | fine-tuning done and it won; the honest CV is what remains — item 1 above |
| **6.5** | four instrument classes emitted but not usable |
| **6.8** | prior-corrected argmax for steps — **tried, lost**, `step-variants.md` §8 |
| **7.4 / 7.5** | serve task 2; take pixels rather than a feature blob |
| **5.4** | the agentic explanation layer — the canvas seam is in place, unused |

Plus the per-fold CV harness change described at the end of `infra/README.md`,
which item 1 depends on.

The honest framing on 7.x: 0.561 served at 200 Hz is still 0.561. The
deployment path was worth building because it proved the auto-regressive
rollout survives the port; extending it competes with making the model better,
and 3.6b is the item that does that.
