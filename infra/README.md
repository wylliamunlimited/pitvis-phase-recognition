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
- **Run stage 2 only first.** Comment out the per-fold loop, fine-tune once on
  all of TRAIN, and score VAL. That is 1 run instead of 6, and it answers "does
  a fine-tuned DINOv2 beat a frozen one" — which is the question. The other
  five exist only to make an honest *ranking* possible.

Two things dominate the bill more than the card choice:

- **Forgetting to shut down.** `startup.sh` shuts the instance down from a
  `trap`, so it fires on success, on failure, and on signal. An idle A100 costs
  more per forgotten day than this whole experiment costs to run.
- **Not using spot/preemptible.** The job is idempotent — every stage skips
  work already on disk and `startup.sh` re-syncs finished backbones before
  starting — so a preemption costs the current fold, not the run.

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
  --metadata=BUCKET="$BUCKET",EPOCHS=50,BACKBONE=vit_base_patch14_dinov2.lvd142m,BRANCH=main \
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
uv run pitvis-extract --space dinov2_ft        # or rsync features down instead
uv run pitvis-verify  --space dinov2_ft

# then the comparison the job exists to enable — VAL, scored once
uv run pitvis-train arst-v2        --variant best --space dinov2_ft
uv run pitvis-train instruments-v2 --variant best --space dinov2_ft
```

## After it lands

The per-fold backbones make an honest cross-validation possible for the first
time on fine-tuned features. Each fold's temporal model must read features from
*its own* encoder — the one that never saw its held-out videos — which is a
change to the harness beyond what a single `--space` expresses, and is the next
piece of work rather than something this job completes.

The full-TRAIN backbone is immediately usable for the single VAL scoring, since
VAL is disjoint from TRAIN.
