#!/usr/bin/env bash
# GCE startup script. Runs as root on boot, pulls inputs, runs the job, pushes
# results back, and SHUTS THE INSTANCE DOWN.
#
# The shutdown is the most important line in this file. A forgotten A100 costs
# more per idle day than this entire experiment costs to run, and the failure
# mode is silent — the job succeeds and the meter keeps running. It fires from
# a trap, so it also happens on error.
#
# Inputs pulled from $BUCKET (private, see infra/README.md on the dataset
# licence): the frame cache only. Not the 40 GB of video — the job never
# decodes, it reads JPEGs.
#
# Metadata expected on the instance:
#   BUCKET   gs://your-bucket        required
#   EPOCHS   50                      optional
#   BACKBONE vit_base_patch14_dinov2.lvd142m   optional; `resnet50` for the
#            cheaper arm. run_job.sh maps it to a tag, a space and an input
#            size, and refuses a backbone it has no mapping for.
#   STAGE    all | full             optional. `full` runs ONE fine-tune instead
#            of six — the ~1/6 cost pass that answers whether a fine-tuned
#            encoder beats the frozen one. Start here.
#   BATCH    64                     optional; lower it if a ViT runs out of
#            VRAM on a smaller card.
#   REPO     https://github.com/...  optional, defaults to origin
#   BRANCH   main

set -uo pipefail
exec > >(tee -a /var/log/pitvis-job.log) 2>&1

meta() { curl -fsH "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }

BUCKET="$(meta BUCKET)"
EPOCHS="$(meta EPOCHS)"; EPOCHS="${EPOCHS:-50}"
BACKBONE="$(meta BACKBONE)"; BACKBONE="${BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
STAGE="$(meta STAGE)"; STAGE="${STAGE:-all}"
BATCH="$(meta BATCH)"; BATCH="${BATCH:-64}"
REPO="$(meta REPO)"; REPO="${REPO:-https://github.com/wylliamunlimited/pitvis-phase-recognition.git}"
BRANCH="$(meta BRANCH)"; BRANCH="${BRANCH:-main}"

# Fires on success, failure, or signal. Results are pushed first so a crash
# mid-job still surfaces whatever completed.
#
# TERM and INT are trapped as well as EXIT because a spot preemption arrives as
# a signal with roughly 30 seconds of grace, not as a normal exit. That grace is
# also why run_job.sh pushes each encoder as it finishes: 2 GB of backbones does
# not reliably reach the bucket in 30 seconds, so the last-moment push is a
# backstop, never the plan.
_finished=0
finish() {
  # JOB_STATUS if the job ran to completion, otherwise whatever killed us.
  # Plain $? here reports the LAST COMMAND's status, which after the job is the
  # echo — so a failed run logged "exit 0" and looked successful in the log
  # someone reads to decide whether to trust the outputs.
  code="${JOB_STATUS:-$?}"
  [ "$_finished" = 1 ] && return          # EXIT still fires after TERM
  _finished=1
  echo "=== finishing (exit $code), pushing results ==="
  local failed=0
  if [ -n "$BUCKET" ] && [ -d /opt/pitvis/data ]; then
    push_dir backbone || failed=1
    push_dir features || failed=1
    gsutil cp /var/log/pitvis-job.log "$BUCKET/out/" || failed=1
  else
    echo "!!! no BUCKET or no data dir — NOTHING WAS SAVED"
    failed=1
  fi
  # Say so loudly rather than shutting down looking successful. The whole point
  # of the bucket is that an hour of GPU time survives the instance.
  if [ "$failed" = 1 ]; then
    echo "!!! ONE OR MORE UPLOADS FAILED — results may exist only on this disk."
    echo "!!! Check $BUCKET/out/ BEFORE deleting the instance."
  else
    echo "=== all results uploaded to $BUCKET/out/ ==="
  fi
  echo "=== shutting down ==="
  shutdown -h now
}
trap finish EXIT INT TERM

# Used by finish(). run_job.sh has its own save_now() for the per-fold pushes,
# because it runs as a child and cannot see this function.
push_dir() {
  local d="$1"
  [ -d "/opt/pitvis/data/$d" ] || return 0
  gsutil -m rsync -r "/opt/pitvis/data/$d" "$BUCKET/out/$d"
}

[ -n "$BUCKET" ] || { echo "BUCKET metadata is required"; exit 1; }

apt-get update -qq && apt-get install -y -qq git ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# A plain clone FAILS on every restart — "destination path '/opt/pitvis'
# already exists" — and because this script runs without `set -e`, it carried
# on and ran whatever code the FIRST boot happened to clone. So fixing a bug
# here and restarting the instance would silently re-run the old bug. Fetch and
# hard-reset instead, so a restart always lands on the branch tip.
if [ -d /opt/pitvis/.git ]; then
  git -C /opt/pitvis fetch --depth 1 origin "$BRANCH"
  git -C /opt/pitvis reset --hard FETCH_HEAD
else
  git clone --branch "$BRANCH" --depth 1 "$REPO" /opt/pitvis
fi
cd /opt/pitvis
echo "=== code at $(git rev-parse --short HEAD) on $BRANCH ==="

# Swap the default (CPU + MPS) torch for the CUDA build. pyproject ships these
# blocks commented out so the Mac build stays the default; here we want them.
python3 - <<'PY'
import pathlib
p = pathlib.Path("pyproject.toml"); s = p.read_text()
s = s.replace("# [[tool.uv.index]]", "[[tool.uv.index]]")
s = s.replace('# name = "pytorch-cu124"', 'name = "pytorch-cu124"')
s = s.replace('# url = "https://download.pytorch.org/whl/cu124"',
              'url = "https://download.pytorch.org/whl/cu124"')
s = s.replace("# explicit = true", "explicit = true")
s = s.replace("# [tool.uv.sources]", "[tool.uv.sources]")
s = s.replace('# torch = { index = "pytorch-cu124" }', 'torch = { index = "pytorch-cu124" }')
s = s.replace('# torchvision = { index = "pytorch-cu124" }',
              'torchvision = { index = "pytorch-cu124" }')
p.write_text(s)
PY

uv sync
mkdir -p data

# Only the frame cache. ~3.6 GB, versus 40 GB of video the job never touches.
# Prefer the single-object tar: ~120k separate GETs is request-bound and slow,
# and the instance is billing an accelerator while it waits. Falls back to the
# per-file layout so a bucket populated by an older launch still works.
echo "=== pulling frames from $BUCKET ==="
if gsutil -q stat "$BUCKET/frames.tar" 2>/dev/null; then
  # --no-xattrs is the actual fix for the AppleDouble problem. The archive is
  # clean — verified by listing it — but it carries PAX
  # `LIBARCHIVE.xattr.com.apple.provenance` headers from macOS, and extracting
  # those here materialises a `._00001.jpg` beside every `00001.jpg`. Those
  # match the `*.jpg` glob the Frames dataset uses and sort BEFORE the real
  # frames, so every label would pair with the wrong image. Measured on the
  # first run: 59,658 sidecars. Dropping the xattrs on extract avoids creating
  # them at all; the sweep below catches any that slip through an older tar.
  gsutil cat "$BUCKET/frames.tar" | tar --no-xattrs -xf - -C data 2>/dev/null \
    || gsutil cat "$BUCKET/frames.tar" | tar -xf - -C data
  find data/frames -name '._*' -delete 2>/dev/null || true
else
  echo "  no frames.tar — falling back to per-file rsync (slower)"
  gsutil -m rsync -r "$BUCKET/frames" data/frames
fi

# CHECK THE FRAMES BEFORE SPENDING GPU TIME. Both of this job's real failures
# were invisible here and became confusing errors minutes later:
#   * a 403 on the bucket left data/frames empty, and the trainer reported
#     "no frames at .../video_02" — which points at the frame cache on a laptop
#     that has one, rather than at the instance's access to the bucket;
#   * a tar built on macOS carried AppleDouble sidecars, and `._00001.jpg`
#     matches the `*.jpg` glob, so every label would pair with the wrong image.
# Ten seconds of counting turns both into an immediate, named failure.
# The labels. Small, but the job is dead without them — and it dies deep inside
# pandas, after the frames have already been extracted.
echo "=== pulling annotations ==="
mkdir -p 26531686
gsutil -m -q cp "$BUCKET/annotations/*" 26531686/ || true
CSVS=$(ls 26531686/annotations_*.csv 2>/dev/null | wc -l | tr -d ' ')
if [ "${CSVS:-0}" -lt 20 ]; then
  echo "!!! only ${CSVS:-0} annotation CSVs in 26531686/ — expected 24."
  echo "!!! The frames are pixels; these are what says which step each one is."
  echo "!!! They are gitignored, so they do NOT arrive with the clone —"
  echo "!!! infra/launch.sh uploads them to $BUCKET/annotations/. Re-run it."
  exit 1
fi
echo "=== annotations ready: $CSVS files ==="

DIRS=$(find data/frames -mindepth 2 -maxdepth 2 -type d 2>/dev/null | wc -l | tr -d ' ')
BAD=$(find data/frames -name '._*' 2>/dev/null | wc -l | tr -d ' ')
echo "=== frames: ${DIRS} video dirs, ${BAD} AppleDouble sidecars ==="
if [ "${DIRS:-0}" -lt 20 ]; then
  echo "!!! only ${DIRS:-0} video directories — expected ~25. The frames did not"
  echo "!!! arrive, so there is nothing to train on. Most likely the instance's"
  echo "!!! service account cannot read $BUCKET: --scopes=storage-rw is an OAuth"
  echo "!!! scope, NOT an IAM role. infra/launch.sh grants the role; re-run it."
  exit 1
fi
if [ "${BAD:-0}" -gt 0 ]; then
  echo "!!! ${BAD} AppleDouble (._*) files in the frame cache. These match the"
  echo "!!! *.jpg glob and would misalign every label. The tar was built on macOS"
  echo "!!! without COPYFILE_DISABLE=1. Rebuild and re-upload it:"
  echo "!!!   COPYFILE_DISABLE=1 tar -cf - -C data frames | gsutil cp - $BUCKET/frames.tar"
  exit 1
fi
# Resume anything a previous (possibly preempted) run finished.
gsutil -m rsync -r "$BUCKET/out/backbone" data/backbone || true

export EPOCHS BACKBONE STAGE BATCH BUCKET

# NOT `exec`. `exec` replaces this shell with the job, which DESTROYS the EXIT
# trap above — so the instance would finish its work, push nothing, and then
# bill until somebody noticed. Running it as a child keeps the trap, and the
# job's exit status reaches finish() through $?.
uv run bash infra/run_job.sh; JOB_STATUS=$?
echo "=== job finished with status $JOB_STATUS ==="
