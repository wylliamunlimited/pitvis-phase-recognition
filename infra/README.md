# Running the fine-tuning job on a cloud GPU

Everything else in this repo runs on a laptop in seconds to minutes. Backbone
fine-tuning does not, and it is the only thing here that needs a rented GPU.

## What the job is, and why it is six fine-tunes

A backbone fine-tuned on all 19 training videos and then cross-validated over
folds drawn from that same set produces features that already encode the
held-out videos' labels. We measured it: steps macro went **0.504 → 0.917**,
which is the size of the leak, not an improvement. `crossval.check_no_leak`
now refuses that configuration outright.

An honest cross-validation therefore needs **one encoder per fold**, each
trained with that fold's videos excluded, plus **one trained on all of TRAIN**
for the single VAL scoring — VAL being the one split no TRAIN-fitted encoder
has ever seen.

That is 6 fine-tunes, and on this laptop the cheapest of them is ~57 h. That is
the entire reason this directory exists. Costs per backbone are below.

## Which backbone, and why it is now the default

`BACKBONE` defaults to **DINOv2 ViT-B/14**, not ResNet-50.

The pilot fine-tuned ResNet-50, because it trains ~3x faster and could answer
"does fine-tuning help at all" cheaply. It answered yes — mean AP 0.271 → 0.445
with 19/19 classes improving. But end to end it does not beat a **frozen**
DINOv2:

| VAL, `best` recipe | frozen DINOv2 | fine-tuned ResNet-50 |
|---|---|---|
| steps · challenge metric | **0.4610** | 0.4425 |
| instruments · official | **0.5572** | 0.3805 |
| instruments · macro | 0.3792 | **0.4783** |

So the untested combination — and the one worth GPU money — is fine-tuning the
encoder that already wins frozen. `BACKBONE=resnet50` still runs the cheaper
arm; the script maps each backbone to its tag, space, frozen counterpart and
input size, and refuses one it has no mapping for rather than guessing.

## Cost

Measured locally on MPS over the 84,666 training frames: **ResNet-50 at 96
img/s** (11.3 min/epoch), **DINOv2 ViT-B/14 at 29 img/s** (~49 min/epoch). The
table scales those by rough GPU throughput — **verify against current pricing
and your own first run**, because both rates and prices move, and the job
prints its actual img/s in the log.

| | ResNet-50, 6 × 50 ep | DINOv2 ViT-B, 6 × 50 ep |
|---|---|---|
| MPS (this laptop) | ~57 h | ~7 days — not viable |
| T4 | ~18 h | ~60 h |
| L4 | ~7 h | ~23 h |
| A100 40GB | ~3 h | ~10 h |

ViT-B is ~3.3x the cost of ResNet-50 at equal epochs, and 85.7M parameters
against 23.5M. Two ways to cut it before committing to the full run:

- **`EPOCHS=10`.** The 5-epoch ResNet-50 pilot already moved mean AP by 0.174,
  so the curve is steep early and 50 may be well past diminishing returns.
  Note the tag/space guard: `dinov2_ft` loads `backbone/dinov2-50ep/`, so a
  different EPOCHS needs the space's path updated. The job checks this in its
  first second rather than after the last hour.
- **`STAGE=full`.** One fine-tune instead of six — the column above divided by
  six. It answers "does a fine-tuned DINOv2 beat a frozen one", which is the
  question; the other five encoders exist only to make an honest *ranking*
  possible later. **Start here.**

Two things dominate the bill more than the card choice, and both are handled by
flags the [runbook](#2-launch) passes: leaving the instance running (the
shutdown fires from a `trap`, so it survives failure) and not using spot (the
job is idempotent, so a preemption costs the fold in flight, not the run).
Neither is optional — check them against that section rather than improvising.

## What moves

```
up    frames/     ~3.6 GB    the 1 fps JPEG cache
down  backbone/   ~2.1 GB    6 fine-tuned ViT-B encoders (~350 MB each;
                              ~600 MB total if BACKBONE=resnet50)
down  features/   ~350 MB    the re-extracted space, 768-d (or re-extract
                              locally from the backbone, which is smaller)
```

**Not the 40 GB of video.** The job never decodes — it reads JPEGs. That is
the whole reason the frame cache is built locally first.

## Dataset licence — read before creating the bucket

PitVis is **CC BY-NC-ND 4.0**: attribution, non-commercial, **no derivatives**.
The frame cache is unambiguously derived from it.

Running your own copy on your own rented compute is not distribution and is
fine. A **public** bucket would be redistribution of a no-derivatives dataset.
Keep the bucket private, do not share the URL, and delete it when the job is
done. See `NOTICE` at the repo root.

## Runbook

### 0. Before you spend anything

- [ ] **Decide the pass.** `STAGE=full` (one fine-tune) unless you already have
      a headline win and need a ranking. On an L4 that is roughly 4 h against
      23 h — see [Cost](#cost).
- [ ] **The frame cache exists locally.** `du -sh data/frames/384` should read
      **~3.6 GB across 25 video directories**. If not:
      ```sh
      uv run pitvis-frames          # ~19 min, needs the 40 GB of video
      ```
      This is the one step that needs the raw videos, and it is done on your
      machine, not the rented one — decoding 40 GB on a GPU instance would bill
      an accelerator to run ffmpeg.
- [ ] **A private bucket.** See [Dataset licence](#dataset-licence--read-before-creating-the-bucket).
      Not optional and not a formality.
- [ ] **`gcloud` is authenticated** and the project has GPU quota in the zone
      you are about to use. Quota denial is the most common way this fails, and
      it fails *after* the instance create call appears to succeed.

### The short version

`infra/launch.sh` does steps 1 and 2 below, with every preflight this file
argues for, and is idempotent:

```sh
BUCKET=gs://your-private-bucket infra/launch.sh
BUCKET=gs://your-private-bucket infra/babysit.sh    # in a second terminal
```

The rest of this section is what those two scripts do, and why — read it before
trusting them with an accelerator.

### 1. Upload the frames — once, ~3.6 GB

```sh
BUCKET=gs://your-private-bucket
gsutil -m rsync -r data/frames "$BUCKET/frames"
```

Only the frames go up. **Not the 40 GB of video** — the job never decodes.

### 2. Launch

```sh
gcloud compute instances create pitvis-ft \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --max-run-duration=8h \
  --scopes=storage-rw \
  --metadata=BUCKET="$BUCKET",STAGE=full,EPOCHS=50,BACKBONE=vit_base_patch14_dinov2.lvd142m,BRANCH=main \
  --metadata-from-file=startup-script=infra/startup.sh
```

Three flags carry the cost discipline, and none is optional:

- `--provisioning-model=SPOT` — the job is idempotent, `run_job.sh` uploads
  each encoder the moment it finishes, and `startup.sh` re-syncs them on boot,
  so a preemption costs the fold in flight and nothing else.
- `--maintenance-policy=TERMINATE` — required with an accelerator.
- `--scopes=storage-rw` — without it every upload fails and the instance shuts
  down having thrown away hours of GPU time. The job now says so loudly rather
  than exiting quietly, but the flag is what prevents it.
- `--max-run-duration=8h` — the backstop that does not depend on our code.
  `startup.sh` shuts itself down from a trap, but a trap cannot fire if the
  script wedges before reaching it (a hung `uv sync`, a stuck `gsutil`). GCE
  enforces this one. Raise it above your expected runtime — 8h suits
  `STAGE=full` on an L4; a six-fold run needs more.
- `--instance-termination-action=STOP`, **not DELETE**. This is what makes
  preemption cheap: STOP keeps the boot disk, so the half-finished encoder's
  `resume.pt` and the 3.6 GB frame cache are still there when you start the
  instance again. DELETE would throw both away and re-download the frames on
  every restart. The cost is a stopped instance and its disk sitting there
  until you delete it — see [after a preemption](#after-a-preemption).

`BRANCH` must name a branch that is **pushed**; the instance clones from the
remote and cannot see your working tree.

### 3. Watch

```sh
gcloud compute ssh pitvis-ft --zone=us-central1-a -- tail -f /var/log/pitvis-job.log
```

A healthy log reaches these within about two minutes:

```
backbone vit_base_patch14_dinov2.lvd142m   epochs 50   space dinov2_ft   tag dinov2-50ep
NVIDIA L4, 23034 MiB
torch 2.x.x cuda True
backbone accepts 224x224 -> 768-d
holding out [...]            # STAGE=all only
  ep1 6,400/84,666  loss 2.41  step-acc 0.31  (312 img/s)
```

Check `img/s` against the [cost table](#cost) on the first epoch — that is when
a wrong estimate is still cheap to act on. `cuda True` and the `224x224 -> 768-d`
probe are the two lines that mean the ViT was built correctly; without
`--img-size` the job aborts here rather than an epoch later.

### 4. Confirm it shut itself down

```sh
gcloud compute instances list --filter="name=pitvis-ft"
```

Expect `TERMINATED`, or no rows. The shutdown fires from a `trap`, so it also
runs on failure — but confirm anyway. An idle accelerator left running over a
weekend costs more than the whole experiment.

### 5. Pull the results back and score

```sh
gsutil -m rsync -r "$BUCKET/out/backbone" data/backbone
uv run pitvis-extract --space dinov2_ft        # or rsync features down instead
uv run pitvis-verify  --space dinov2_ft

uv run pitvis-train arst-v2        --variant best --space dinov2_ft
uv run pitvis-train instruments-v2 --variant best --space dinov2_ft
```

Those land in `v2/best@dinov2_ft/` and leave the current winners alone.

## After a preemption

**A preempted spot VM is STOPPED, and nothing in GCE restarts it.**
`--provisioning-model=SPOT` buys the discount and the eviction, not the
recovery. Left alone, the instance sits at TERMINATED with a half-finished
encoder on its disk.

So either restart it yourself:

```sh
gcloud compute instances start pitvis-ft --zone=us-central1-a
```

or leave `infra/babysit.sh` running, which polls for exactly that state and
restarts it until the `DONE` marker appears in the bucket:

```sh
BUCKET=gs://your-private-bucket infra/babysit.sh
```

Stopping the watcher does not stop the job — it holds no state; everything
lives in the bucket and on the instance's boot disk.

### Does my laptop have to stay awake?

For `babysit.sh`, yes — it runs on your machine, so a sleeping laptop is a
watcher that has stopped. That is a real constraint, and there are three honest
ways out of it. Pick by arithmetic, not by habit:

| | preemption | needs a watcher | when |
|---|---|---|---|
| **`SPOT=0`** (on-demand) | cannot happen | no | a single ~4 h encoder |
| **spot + `babysit.sh`** | recovers | yes, on your machine | you are around anyway |
| **spot + Cloud Scheduler** | recovers | no | the unattended six-fold run |

**For `STAGE=full` I would just use on-demand.** Spot is meaningfully cheaper
per hour, but on one L4 for about four hours the absolute saving is small, and
"my laptop has to stay awake for four hours" is a poor thing to buy with it:

```sh
BUCKET=gs://your-private-bucket SPOT=0 infra/launch.sh
```

Nothing else changes — the resume checkpointing, the incremental uploads and
the shutdown trap all still apply. They just stop being load-bearing.

**For `STAGE=all`** — six encoders, roughly a day — the calculus flips. Eviction
becomes near-certain rather than unlucky, so the saving is worth having and the
watcher has to survive your laptop closing. Move it off your machine with a
Cloud Scheduler job that starts the instance on a schedule; starting an already
running instance is a no-op, so a blunt every-10-minutes is fine:

```sh
gcloud scheduler jobs create http pitvis-ft-resume \
  --schedule="*/10 * * * *" --location=us-central1 \
  --uri="https://compute.googleapis.com/compute/v1/projects/$(gcloud config get-value project)/zones/us-central1-a/instances/pitvis-ft/start" \
  --http-method=POST --oauth-service-account-email=SA_EMAIL
```

The service account needs `roles/compute.instanceAdmin.v1`. **Delete the
scheduler job when the run finishes**, or it will keep restarting an instance
you thought you were done with — the exact failure this whole file exists to
prevent, arriving from the other direction.

The startup script runs again from the top and three things make that cheap
rather than a fresh start:

| | what survives | why |
|---|---|---|
| finished encoders | uploaded as each one completes | `save_now` in run_job.sh |
| the encoder in flight | `resume.pt`, written after **every epoch** | the boot disk (hence STOP) |
| the frame cache | already on the disk | the boot disk |

So the worst case is the epoch in flight — minutes, not the four hours a
whole ViT encoder costs. `pitvis-finetune` prints `resuming from ... at epoch
N/50` when it picks the state up; if you do not see that line, it is starting
from scratch and something above did not survive.

The resume state carries the optimiser and scheduler, not just the weights.
AdamW holds per-parameter moments and cosine decay holds its position;
restoring weights alone would silently resume at the wrong learning rate with
the moments zeroed — a different training run that happens to start from the
same numbers. It also records `trained_on` and refuses to resume into a run
that holds out a different set of videos.

**When the job is genuinely finished, delete the instance** — STOP leaves the
disk billing:

```sh
gcloud compute instances delete pitvis-ft --zone=us-central1-a
```

## Reading the result honestly

**The bar to beat**, VAL, `best` recipe. The right-hand column is what the job
described here has already produced — it is the bar now, not the frozen one:

| | frozen DINOv2 | **`dinov2_ft` (run 2)** |
|---|---|---|
| steps · challenge metric | 0.4610 ± 0.043 | **0.5608** ± 0.052 |
| instruments · official | **0.5572** ± 0.225 | 0.3220 ± 0.089 |
| instruments · name-aligned weighted | 0.7383 ± 0.041 | **0.8416** ± 0.036 |
| instruments · macro | 0.3792 | **0.5333** |
| probe · mean AP | 0.350 | **0.523** |

The instrument `official` row falling while both defect-free rows rise is the
vendored column-ordering defect, not a regression —
[`instrument-variants.md`](../notes/models/instrument-variants.md) has the
per-video proof. Point 2 below was written before that was understood and is
still the right rule; it just does not apply to this particular disagreement.

Four things to hold on to when the numbers come back:

1. **It is one measurement per arm, not a ranking.** Five videos, per-video std
   around 0.05 on steps and 0.23 on the instruments headline. A gap smaller
   than that spread is not a result. This is precisely what the CV protocol
   exists to avoid, and `STAGE=full` cannot give you a CV — only the six-fold
   run can, and only via the harness change described below.
2. **Expect the metrics to disagree, and decide in advance which one rules.**
   The fine-tuned ResNet-50 raised instrument macro by 0.099 while dropping the
   official number by 0.177: better on the rare classes macro weights equally,
   worse on the four carrying ~91% of positives. The pre-registered rule is
   **primary `macro_f1`, guarded on the official metric** — apply it rather than
   picking the flattering column afterwards.
3. **Watch the edit score on steps, not just macro.** Fine-tuning is frame-wise
   with no temporal term, and on ResNet-50 it bought +0.024 macro for −0.061
   edit — better at naming a second, worse at holding a segment together. If
   DINOv2 repeats that shape, the fix is a temporal objective, not more epochs.
4. **VAL is not the test set.** Das et al. measure a −47-point val→test collapse
   for instruments against −7 for steps. Nothing here is comparable to the
   leaderboard.

## After it lands

`STAGE=full` gives you a headline number and a shippable encoder. It does
**not** give you an honest cross-validation: the full-TRAIN backbone has seen
every fold's held-out videos, and `crossval.check_no_leak` will refuse to
pretend otherwise.

The six-fold run makes that possible for the first time, but the harness is not
there yet. Each fold's temporal model has to read features from *its own*
encoder — the one that never saw its held-out videos — which is more than a
single `--space` can express. That is the next piece of work, and it is a
change to `crossval.py`, not something this job completes.
