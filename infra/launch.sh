#!/usr/bin/env bash
# Everything between "I have a GCP project" and "the GPU is training".
#
# Run from the repo root on YOUR machine:
#
#     BUCKET=gs://your-private-bucket infra/launch.sh
#
# Idempotent: the bucket, the frame upload and the instance are each created
# only if missing, so re-running after a failed preflight costs nothing.
#
# Every check here exists because it fails LATE otherwise — after the frames
# are uploaded, or after the accelerator is already billing. The expensive
# mistakes in this job are all mistakes of ordering.

set -euo pipefail

# Zones are tried in order until one has capacity. A GPU stockout
# (ZONE_RESOURCE_POOL_EXHAUSTED) is not a quota problem and not a permanent
# one — it means that zone has no L4 free at this moment — so the useful
# response is to try the next zone, not to give up or to wait blindly.
#
# Same-region zones come first: the bucket is regional, and pulling 3.7 GB
# from us-central1 into another region is a cross-region egress charge plus a
# slower boot. Setting ZONE pins a single zone and disables the fallback.
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-east1-b us-east1-c us-east1-d us-west1-a us-west1-b}"
[ -n "${ZONE:-}" ] && ZONES="$ZONE"
ZONE="${ZONES%% *}"          # first zone, used by preflight
NAME="${NAME:-pitvis-ft}"
MACHINE="${MACHINE:-g2-standard-8}"
ACCEL="${ACCEL:-type=nvidia-l4,count=1}"
DISK="${DISK:-200GB}"
MAX_RUN="${MAX_RUN:-8h}"
# Deep Learning VM families are versioned and RETIRE. `pytorch-latest-gpu` was
# the documented one and no longer resolves at all, which failed the create
# call after the 3.7 GB upload had already been paid for. Pinned to a current
# family, overridable, and checked in preflight so a retirement is a five-second
# error instead of a late one.
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
STAGE="${STAGE:-full}"
EPOCHS="${EPOCHS:-50}"
BACKBONE="${BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

# SPOT=1 is cheaper per hour and can be evicted at any moment; a preempted VM
# is STOPPED and something must start it again (infra/babysit.sh, or a
# scheduler — see the README). SPOT=0 cannot be preempted and therefore needs
# no watcher at all.
#
# The trade is worth doing by arithmetic, not by habit. Spot is the obvious
# choice for the six-fold run, where eviction over ~23 h is near-certain and
# the saving is real. For a single ~4 h encoder the absolute saving is small,
# and "my laptop has to stay awake" is a poor thing to buy with it.
SPOT="${SPOT:-1}"
if [ "$SPOT" = 1 ]; then
  PROVISION=(--provisioning-model=SPOT --instance-termination-action=STOP)
  PROVISION_NOTE="SPOT (evictable — keep infra/babysit.sh running)"
else
  # STOP is required here too: GCE rejects --max-run-duration without a
  # termination action on ANY provisioning model, not just SPOT. STOP rather
  # than DELETE so the boot disk (and therefore any resume state) survives the
  # backstop firing.
  PROVISION=(--provisioning-model=STANDARD --instance-termination-action=STOP)
  PROVISION_NOTE="STANDARD (not evictable — no watcher needed)"
fi

die() { echo "ERROR: $*" >&2; exit 1; }
ok()  { echo "  ok   $*"; }

echo "=== preflight ==="
[ -n "${BUCKET:-}" ] || die "set BUCKET, e.g. BUCKET=gs://my-private-bucket $0"
command -v gcloud >/dev/null || die "gcloud not installed"
command -v gsutil >/dev/null || die "gsutil not installed"

PROJECT="$(gcloud config get-value project 2>/dev/null)"
[ -n "$PROJECT" ] && [ "$PROJECT" != "(unset)" ] || die "no project set: gcloud config set project PROJECT_ID"
ok "project $PROJECT"

# The Compute API is not on by default in a fresh project, and enabling it
# takes a few minutes to propagate. Finding that out here costs nothing;
# finding it out from `instances create` costs the frame upload first.
gcloud services list --enabled --format="value(config.name)" 2>/dev/null \
  | grep -q "^compute.googleapis.com$" \
  || die "Compute Engine API is not enabled. Run:
    gcloud services enable compute.googleapis.com storage.googleapis.com
  then wait a few minutes and re-run this script."
ok "compute API enabled"

# The instance clones from the REMOTE, so a branch that only exists locally —
# or one without infra/ on it — produces an instance that boots, installs
# everything, and then cannot find the job.
git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1 \
  || die "branch '$BRANCH' is not pushed. The instance clones from origin, not from your working tree."
git cat-file -e "origin/$BRANCH:infra/run_job.sh" 2>/dev/null \
  || die "origin/$BRANCH has no infra/run_job.sh. The instance would boot, install
  everything, and fail. Push the branch that carries infra/, or pass BRANCH=..."
ok "branch $BRANCH is pushed and carries infra/"

RAWDIR="${RAWDIR:-26531686}"
[ -d "$RAWDIR" ] || die "no $RAWDIR/ — the annotation CSVs live there and the
  instance needs them for labels. Set RAWDIR if yours is elsewhere."
[ -d data/frames ] || die "no data/frames — run: uv run pitvis-frames"
FRAME_DIRS=$(find data/frames -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ')
[ "$FRAME_DIRS" -ge 20 ] || die "only $FRAME_DIRS video dirs under data/frames; expected ~25.
  Re-run: uv run pitvis-frames"
ok "frame cache present ($FRAME_DIRS videos, $(du -sh data/frames | cut -f1))"

# GPU quota is per project per region and a NEW project has none. This is the
# one preflight that can take days to clear, so it is checked before anything
# is uploaded.
REGION="${ZONE%-*}"
# Read the quota out of JSON. The `--format="value(quotas.filter(...))"` form
# looks right and silently yields nothing, which reported "no quota" on an
# account that had it — a warning that cries wolf is worse than no warning.
QUOTA=$(gcloud compute regions describe "$REGION" --format="json(quotas)" 2>/dev/null \
  | python3 -c "import json,sys
try:
    q = json.load(sys.stdin).get('quotas', [])
except Exception:
    q = []
print(next((str(x['limit']) for x in q if x['metric'] == 'NVIDIA_L4_GPUS'), ''))" 2>/dev/null || echo "")
if [ -z "$QUOTA" ] || [ "${QUOTA%%.*}" = "0" ]; then
  echo "  WARN no NVIDIA_L4_GPUS quota in $REGION (reported: '${QUOTA:-none}')."
  echo "       Request it at IAM & Admin > Quotas before this can run."
  echo "       Continuing — the create call below is what confirms it."
else
  ok "L4 quota in $REGION: $QUOTA"
fi

# GPUS_ALL_REGIONS is a SECOND, GLOBAL quota, separate from the per-region one
# above, and a fresh project has it at zero even when the regional quota has
# been granted. Missing it is why a create call fails with "Quota
# 'GPUS_ALL_REGIONS' exceeded" after every regional check has passed — and it
# fails after the 3.7 GB upload, which is the expensive place to learn it.
GLOBAL_GPU=$(gcloud compute project-info describe --format="json(quotas)" 2>/dev/null \
  | python3 -c "import json,sys
try:
    q = json.load(sys.stdin).get('quotas', [])
except Exception:
    q = []
print(next((str(x['limit']) for x in q if x['metric'] == 'GPUS_ALL_REGIONS'), ''))" 2>/dev/null || echo "")
if [ -z "$GLOBAL_GPU" ] || [ "${GLOBAL_GPU%%.*}" = "0" ]; then
  echo "  WARN GPUS_ALL_REGIONS is ${GLOBAL_GPU:-unreadable} — this is a SEPARATE"
  echo "       global quota from NVIDIA_L4_GPUS above, and BOTH must be non-zero."
  echo "       Request it: IAM & Admin > Quotas > filter 'GPUS_ALL_REGIONS'."
else
  ok "GPUS_ALL_REGIONS: $GLOBAL_GPU"
fi

# The image family is checked here, not discovered at create time, because the
# create call happens AFTER the upload.
gcloud compute images describe-from-family "$IMAGE_FAMILY" \
  --project="$IMAGE_PROJECT" --format="value(name)" >/dev/null 2>&1 \
  || die "image family '$IMAGE_FAMILY' does not exist in $IMAGE_PROJECT.
  These do:
$(gcloud compute images list --project="$IMAGE_PROJECT" --filter='family~pytorch OR family~cu1' --format='value(family)' 2>/dev/null | sort -u | sed 's/^/    /')
  Re-run with IMAGE_FAMILY=<one of those>."
ok "image family $IMAGE_FAMILY"

echo
echo "=== bucket ==="
if gsutil ls -b "$BUCKET" >/dev/null 2>&1; then
  ok "$BUCKET exists"
else
  # Uniform access + no public access: the frame cache is a derivative of a
  # CC BY-NC-ND dataset, so a public object here would be redistribution.
  gsutil mb -l "$REGION" -b on "$BUCKET"
  gsutil pap set enforced "$BUCKET"
  ok "created $BUCKET (region $REGION, public access prevented)"
fi

# --scopes=storage-rw grants the instance an OAUTH SCOPE. It does not grant the
# service account an IAM ROLE, and they are different things: the scope says
# "this VM may present storage credentials", the role says "those credentials
# may touch this bucket". Older projects hid the gap because the default compute
# SA came with project Editor; newer ones grant it nothing at all — so the VM
# boots, authenticates cleanly, then 403s on the first read, which surfaces
# inside the trainer as "no frames at .../video_02" several minutes later.
echo
echo "=== bucket access for the instance ==="
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
if gcloud storage buckets get-iam-policy "$BUCKET" --format=json 2>/dev/null | grep -q "$SA"; then
  ok "$SA can already reach $BUCKET"
else
  gcloud storage buckets add-iam-policy-binding "$BUCKET" \
    --member="serviceAccount:$SA" --role=roles/storage.objectAdmin >/dev/null
  ok "granted roles/storage.objectAdmin on $BUCKET to $SA"
fi

echo
echo "=== frames -> bucket ==="
# ONE OBJECT, NOT 120,018.
#
# The frame cache is 3.6 GB in ~120k files of ~31 KB. Uploaded per-file that is
# 120k HTTPS round trips, and the wall clock is set by request count, not by
# bytes — it takes hours on a connection that would move 3.6 GB in minutes.
# A tar is a single object, so the transfer becomes bandwidth-bound, and the
# instance pulls it back the same way.
#
# Streamed through a pipe on both ends: no 3.6 GB temp copy on either disk.
# Not gzipped — JPEG is already compressed, so it would buy nothing and cost
# CPU on both sides.
if gsutil -q stat "$BUCKET/frames.tar" 2>/dev/null; then
  ok "frames.tar already uploaded"
else
  echo "  packing and streaming $(du -sh data/frames | cut -f1) as a single object..."
  # COPYFILE_DISABLE=1 stops macOS tar emitting AppleDouble entries. Measured:
  # it changes nothing for this cache — the archive is clean either way, with
  # or without it — because the frames carry only `com.apple.provenance`, which
  # bsdtar stores as a PAX header rather than as a `._` entry. Kept as hygiene,
  # NOT as the fix: the `._*.jpg` files that broke the first run appear when
  # LINUX extracts those PAX headers, so the repair belongs on the instance and
  # lives in startup.sh.
  COPYFILE_DISABLE=1 tar -cf - -C data frames | gsutil cp - "$BUCKET/frames.tar"
  ok "uploaded $BUCKET/frames.tar"
fi

# THE LABELS TRAVEL TOO. `Frames` reads the frame pixels from data/frames and
# the labels from 26531686/annotations_NN.csv — both are gitignored, so neither
# reaches the instance through the clone. Shipping 3.7 GB of pixels and not the
# 1.8 MB that says what they are failed the job after boot, install, and a full
# frame extract, with a FileNotFoundError several layers deep in pandas.
echo
echo "=== annotations -> bucket (1.8 MB) ==="
CSVS=$(ls "$RAWDIR"/annotations_*.csv 2>/dev/null | wc -l | tr -d ' ')
[ "$CSVS" -ge 20 ] || die "only $CSVS annotation CSVs in $RAWDIR — expected 24."
gsutil -m -q cp "$RAWDIR"/annotations_*.csv "$RAWDIR"/map_*.csv "$BUCKET/annotations/"
ok "uploaded $CSVS annotation CSVs + the map files"

echo
echo "=== instance ==="
# Look for the instance in EVERY zone, not just the first candidate. The zone
# fallback means it may live anywhere in $ZONES, and a zone-scoped describe
# reported "does not exist" for an instance that was merely somewhere else —
# so this created a SECOND instance, which then failed to start because the
# first was holding the single GPU of quota.
FOUND_ZONE=$(gcloud compute instances list --filter="name=$NAME" \
  --format="value(zone.basename())" 2>/dev/null | head -1)
if [ -n "$FOUND_ZONE" ]; then
  ZONE="$FOUND_ZONE"
  STATUS=$(gcloud compute instances describe "$NAME" --zone="$ZONE" --format="value(status)")
  echo "  found $NAME in $ZONE"
  echo "  $NAME already exists (status $STATUS)."
  # THE STARTUP SCRIPT LIVES IN INSTANCE METADATA, captured at create time.
  # Restarting re-runs that frozen copy, so a fix landed in this repo never
  # reaches an existing instance — it silently re-runs the old bug, which is
  # exactly what happened after the clone fix. Refresh it here so "start" and
  # "create" run the same code.
  gcloud compute instances add-metadata "$NAME" --zone="$ZONE" \
    --metadata=BUCKET="$BUCKET",STAGE="$STAGE",EPOCHS="$EPOCHS",BACKBONE="$BACKBONE",BRANCH="$BRANCH" \
    --metadata-from-file=startup-script=infra/startup.sh >/dev/null
  ok "refreshed its startup script and metadata from this checkout"
  echo "  Start it:      gcloud compute instances start $NAME --zone=$ZONE"
  echo "  Or start over: gcloud compute instances delete $NAME --zone=$ZONE"
  exit 0
fi

BUCKET_REGION=$(gcloud storage buckets describe "$BUCKET" --format="value(location)" 2>/dev/null | tr 'A-Z' 'a-z')
CREATED=""
for Z in $ZONES; do
  case "$Z" in
    "$BUCKET_REGION"-*) ;;
    *) echo "  note: $Z is outside the bucket's region ($BUCKET_REGION) — the"
       echo "        instance will pull frames.tar cross-region (egress + slower boot)" ;;
  esac
  echo "  trying $Z ..."
  if gcloud compute instances create "$NAME" \
  --zone="$Z" \
  --machine-type="$MACHINE" \
  --accelerator="$ACCEL" \
  --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="$DISK" \
  --maintenance-policy=TERMINATE \
  "${PROVISION[@]}" \
  --max-run-duration="$MAX_RUN" \
  --scopes=storage-rw \
  --labels=exp=pitvis-ft \
  --metadata=BUCKET="$BUCKET",STAGE="$STAGE",EPOCHS="$EPOCHS",BACKBONE="$BACKBONE",BRANCH="$BRANCH" \
  --metadata-from-file=startup-script=infra/startup.sh 2>&1 | sed 's/^/    /'; then
    CREATED="$Z"; break
  fi
  echo "    no capacity in $Z, trying the next zone"
done

[ -n "$CREATED" ] || die "no zone in the list had capacity for $MACHINE + $ACCEL.
  This is a STOCKOUT, not a quota problem — capacity comes back. Options:
    * wait and re-run; stockouts usually clear within hours
    * widen the search:  ZONES='\''us-east4-a us-east4-c europe-west4-a'\'' $0
    * try a different accelerator, e.g. ACCEL=type=nvidia-tesla-t4,count=1
      MACHINE=n1-standard-8 (slower, but T4 capacity is usually easier)"
ZONE="$CREATED"

cat <<EOF

=== running ===
  stage $STAGE   epochs $EPOCHS   backbone $BACKBONE   branch $BRANCH
  provisioning: $PROVISION_NOTE

Watch it:
  gcloud compute ssh $NAME --zone=$ZONE -- tail -f /var/log/pitvis-job.log

$( [ "$SPOT" = 1 ] && cat <<SPOTNOTE
Keep it alive across preemptions — spot VMs do NOT restart themselves, and
this runs on THIS machine, so it stops if the laptop sleeps:
  BUCKET=$BUCKET infra/babysit.sh
SPOTNOTE
)

When the DONE marker appears, pull the results:
  gsutil -m rsync -r $BUCKET/out/backbone data/backbone
  gcloud compute instances delete $NAME --zone=$ZONE
EOF
