# Where we are — orientation snapshot

*Snapshot: 2026-08-10. Read this first after time away, then follow the links.*

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
[`instrument-variants.md` §2](instrument-variants.md)

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
  [`step-variants.md`](step-variants.md),
  [`instrument-variants.md`](instrument-variants.md)
- the reproductions those improve on —
  [`citi-baseline.md`](citi-baseline.md), [`instruments.md`](instruments.md)
- the architectures as shape traces — [`citi-dataflow.md`](citi-dataflow.md)
- the cross-task finding worth carrying (**never test a backbone swap first**) —
  [`roadmap.md`](roadmap.md#the-finding-worth-carrying-forward)
- the diagnostic that says the encoder is the next lever —
  [`instrument-variants.md` §6](instrument-variants.md)

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
| `data/features/` (3 spaces) | 2.2 GB | ~75 min |
| `data/arst/` + `data/instruments/` | 461 MB | ~15 min |
| `data/frames/384/` | 3.6 GB | 19 min |
| `26531686/` raw video | **40 GB** | hours to re-download |

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

**Option A — an honest number today, ~10 min.** The pilot backbone was trained
on TRAIN only and VAL is disjoint from TRAIN, so scoring it on VAL is
legitimate. It spends our one clean VAL touch for this backbone.

```sh
uv run pitvis-train instruments-v2 --variant best --space resnet50_ft
uv run pitvis-train arst-v2       --variant best --space resnet50_ft
```

Compare against the current bests in §2.

**Option B — the cloud job.** Six fine-tunes, which is what makes honest CV
possible on fine-tuned features. See [`infra/README.md`](../infra/README.md).

**Do A before B.** If a 5-epoch pilot already beats the current bests on VAL,
that justifies the spend. If it does not move, you have saved the bill.

### Open threads

Tracked in [`roadmap.md`](roadmap.md) — the live ones are 6.5 (four instrument
classes emitted but not usable), 6.8 (prior-corrected argmax for steps), and
the per-fold CV harness change described at the end of `infra/README.md`.
