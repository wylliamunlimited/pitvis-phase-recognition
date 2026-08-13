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

ZONE="${ZONE:-us-central1-a}"
NAME="${NAME:-pitvis-ft}"
MACHINE="${MACHINE:-g2-standard-8}"
ACCEL="${ACCEL:-type=nvidia-l4,count=1}"
DISK="${DISK:-200GB}"
MAX_RUN="${MAX_RUN:-8h}"
STAGE="${STAGE:-full}"
EPOCHS="${EPOCHS:-50}"
BACKBONE="${BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
BRANCH="${BRANCH:-$(git rev-parse --abbrev-ref HEAD)}"

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

[ -d data/frames ] || die "no data/frames — run: uv run pitvis-frames"
FRAME_DIRS=$(find data/frames -mindepth 2 -maxdepth 2 -type d | wc -l | tr -d ' ')
[ "$FRAME_DIRS" -ge 20 ] || die "only $FRAME_DIRS video dirs under data/frames; expected ~25.
  Re-run: uv run pitvis-frames"
ok "frame cache present ($FRAME_DIRS videos, $(du -sh data/frames | cut -f1))"

# GPU quota is per project per region and a NEW project has none. This is the
# one preflight that can take days to clear, so it is checked before anything
# is uploaded.
REGION="${ZONE%-*}"
QUOTA=$(gcloud compute regions describe "$REGION" \
  --format="value(quotas.filter(metric:NVIDIA_L4_GPUS).limit)" 2>/dev/null || echo "")
if [ -z "$QUOTA" ] || [ "${QUOTA%%.*}" = "0" ]; then
  echo "  WARN no NVIDIA_L4_GPUS quota in $REGION (reported: '${QUOTA:-none}')."
  echo "       Request it at IAM & Admin > Quotas before this can run."
  echo "       Continuing — the create call below is what confirms it."
else
  ok "L4 quota in $REGION: $QUOTA"
fi

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

echo
echo "=== frames -> bucket (~3.6 GB, skipped if already there) ==="
gsutil -m rsync -r data/frames "$BUCKET/frames"

echo
echo "=== instance ==="
if gcloud compute instances describe "$NAME" --zone="$ZONE" >/dev/null 2>&1; then
  STATUS=$(gcloud compute instances describe "$NAME" --zone="$ZONE" --format="value(status)")
  echo "  $NAME already exists (status $STATUS)."
  echo "  To resume it:  gcloud compute instances start $NAME --zone=$ZONE"
  echo "  To start over: gcloud compute instances delete $NAME --zone=$ZONE"
  exit 0
fi

gcloud compute instances create "$NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator="$ACCEL" \
  --image-family=pytorch-latest-gpu --image-project=deeplearning-platform-release \
  --boot-disk-size="$DISK" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --max-run-duration="$MAX_RUN" \
  --scopes=storage-rw \
  --labels=exp=pitvis-ft \
  --metadata=BUCKET="$BUCKET",STAGE="$STAGE",EPOCHS="$EPOCHS",BACKBONE="$BACKBONE",BRANCH="$BRANCH" \
  --metadata-from-file=startup-script=infra/startup.sh

cat <<EOF

=== running ===
  stage $STAGE   epochs $EPOCHS   backbone $BACKBONE   branch $BRANCH

Watch it:
  gcloud compute ssh $NAME --zone=$ZONE -- tail -f /var/log/pitvis-job.log

Keep it alive across preemptions (spot VMs do NOT restart themselves):
  BUCKET=$BUCKET infra/babysit.sh

When the DONE marker appears, pull the results:
  gsutil -m rsync -r $BUCKET/out/backbone data/backbone
  gcloud compute instances delete $NAME --zone=$ZONE
EOF
