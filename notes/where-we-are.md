# Where we are — orientation snapshot

*Snapshot: 2026-08-14. Read this first after time away, then follow the links.*

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

| | what changed | steps | instruments |
|---|---|---|---|
| start | published reproductions | 0.3425 | 0.2321 |
| iter 1+2 | loss, decision rule, DINOv2 | **0.4610** | **0.5572** |
| iter 3 | fine-tuned ResNet-50 encoder | 0.4425 | 0.3805 |

Steps are the challenge metric, instruments the official one, both on VAL.

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

Note the fine-tune is **ResNet-50, not DINOv2** — ViT-B trains at 29 img/s
against ResNet-50's 96, so the pilot went to the cheap backbone to find out
whether fine-tuning helps at all. Nothing has fine-tuned DINOv2 yet.

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
`dinov2_vitb14`.

**The fine-tuned backbone is the one worth carrying** — 96 MB for an hour of
training. Copy `data/` (6.3 GB) to a private bucket or an external drive.

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

**Option A is done.** The pilot backbone was scored on VAL — that is the iter 3
row in §2 — and the verdict was "better on rare classes, worse on the ones
carrying the support, and no honest ranking available". It answered the cheap
question and did not justify itself.

**Option B — the cloud job — is therefore the live one.** Fine-tune the encoder
that already *wins* frozen, which is DINOv2, not the ResNet-50 the pilot used.
Start with `STAGE=full`: one fine-tune instead of six, ~4 h on an L4, which
answers "does a fine-tuned DINOv2 beat frozen DINOv2 on VAL" for a sixth of the
cost. The five per-fold encoders only buy an honest *ranking*, and there is
nothing to rank until the headline moves.

```sh
SPOT=0 STAGE=full BUCKET=gs://your-private-bucket infra/launch.sh
```

Prerequisites, in order — `launch.sh` preflights all of them and stops before
the 3.6 GB upload if any fail: `gcloud` installed, `data/frames` present
(`uv run pitvis-frames`, ~40 min), the branch pushed with `infra/` on it, and
**both** GPU quotas non-zero (`NVIDIA_L4_GPUS` per region and the separate
global `GPUS_ALL_REGIONS`).

→ costs per backbone and per GPU, and why it is six fine-tunes rather than one:
[`infra/README.md`](../infra/README.md)

**Afterwards, delete the instance.** Termination action is STOP so the boot disk
survives a preemption, which means a finished job leaves a stopped instance
billing ~$20/month for a 200 GB disk. `babysit.sh` and `launch.sh` both print
the delete command; nothing enforces it.

### Open threads

Tracked in [`roadmap.md`](roadmap.md). The live ones:

| | |
|---|---|
| **3.6b** | fine-tune DINOv2 and cross-validate it honestly — Option B above |
| **6.5** | four instrument classes emitted but not usable |
| **6.8** | prior-corrected argmax for steps |
| **7.4 / 7.5** | serve task 2; take pixels rather than a feature blob |
| **5.4** | the agentic explanation layer — the canvas seam is in place, unused |

Plus the per-fold CV harness change described at the end of `infra/README.md`.

The honest framing on 7.x: 0.461 served at 200 Hz is still 0.461. The
deployment path was worth building because it proved the auto-regressive
rollout survives the port; extending it competes with making the model better,
and 3.6b is the item that does that.
