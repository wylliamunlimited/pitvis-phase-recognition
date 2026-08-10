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

That is 6 fine-tunes. On MPS at 11.3 min/epoch it is ~57 h at 50 epochs. That
is the entire reason this directory exists.

## Cost

Measured locally: ResNet-50 trains at **96 img/s on MPS**, 11.3 min/epoch over
84,666 frames. The table below scales that by rough GPU throughput — **verify
against current pricing and your own first run**, because both rates and prices
move, and the job prints its actual img/s in the log.

| | approx img/s | 6 × 50 epochs | at spot | at on-demand |
|---|---|---|---|---|
| MPS (this laptop) | 96 | ~57 h | — | — |
| T4 | ~300 | ~18 h | low | low-ish |
| L4 | ~800 | ~7 h | moderate | moderate |
| A100 40GB | ~2000 | ~3 h | higher | highest |

Two things dominate the bill more than the card choice:

- **Forgetting to shut down.** `startup.sh` shuts the instance down from a
  `trap`, so it fires on success, on failure, and on signal. An idle A100 costs
  more per forgotten day than this whole experiment costs to run.
- **Not using spot/preemptible.** The job is idempotent — every stage skips
  work already on disk and `startup.sh` re-syncs finished backbones before
  starting — so a preemption costs the current fold, not the run.

A cheaper first pass: `EPOCHS=10`. Our 5-epoch pilot already moved mean AP from
0.271 to 0.445 with 19/19 classes improving, so the curve is steep early and
50 epochs may be well past the point of diminishing returns.

## What moves

```
up    frames/     ~3.6 GB    the 1 fps JPEG cache
down  backbone/   ~600 MB    6 fine-tuned encoders
down  features/   ~940 MB    the re-extracted space (or re-extract locally)
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

## Running it

```sh
BUCKET=gs://your-private-bucket

gsutil -m rsync -r data/frames "$BUCKET/frames"      # ~3.6 GB, once

gcloud compute instances create pitvis-ft \
  --zone=us-central1-a \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --scopes=storage-rw \
  --metadata=BUCKET="$BUCKET",EPOCHS=50,BACKBONE=resnet50,BRANCH=main \
  --metadata-from-file=startup-script=infra/startup.sh
```

Watch it:

```sh
gcloud compute ssh pitvis-ft --zone=us-central1-a -- tail -f /var/log/pitvis-job.log
```

It shuts itself down when finished. Confirm, and confirm again:

```sh
gcloud compute instances list --filter="name=pitvis-ft"
```

Then pull the results back:

```sh
gsutil -m rsync -r "$BUCKET/out/backbone" data/backbone
uv run pitvis-extract --space resnet50_ft      # or rsync features down instead
uv run pitvis-verify  --space resnet50_ft
```

## After it lands

The per-fold backbones make an honest cross-validation possible for the first
time on fine-tuned features. Each fold's temporal model must read features from
*its own* encoder — the one that never saw its held-out videos — which is a
change to the harness beyond what a single `--space` expresses, and is the next
piece of work rather than something this job completes.

The full-TRAIN backbone is immediately usable for the single VAL scoring, since
VAL is disjoint from TRAIN.
