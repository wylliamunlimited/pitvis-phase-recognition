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
#   REPO     https://github.com/...  optional, defaults to origin
#   BRANCH   main

set -uo pipefail
exec > >(tee -a /var/log/pitvis-job.log) 2>&1

meta() { curl -fsH "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null || true; }

BUCKET="$(meta BUCKET)"
EPOCHS="$(meta EPOCHS)"; EPOCHS="${EPOCHS:-50}"
BACKBONE="$(meta BACKBONE)"; BACKBONE="${BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
REPO="$(meta REPO)"; REPO="${REPO:-https://github.com/wylliamunlimited/pitvis-phase-recognition.git}"
BRANCH="$(meta BRANCH)"; BRANCH="${BRANCH:-main}"

# Fires on success, failure, or signal. Results are pushed first so a crash
# mid-job still surfaces whatever completed.
finish() {
  code=$?
  echo "=== finishing (exit $code), pushing results ==="
  if [ -n "$BUCKET" ] && [ -d /opt/pitvis/data ]; then
    gsutil -m rsync -r /opt/pitvis/data/backbone  "$BUCKET/out/backbone"  || true
    gsutil -m rsync -r /opt/pitvis/data/features  "$BUCKET/out/features"  || true
    gsutil cp /var/log/pitvis-job.log "$BUCKET/out/" || true
  fi
  echo "=== shutting down ==="
  shutdown -h now
}
trap finish EXIT

[ -n "$BUCKET" ] || { echo "BUCKET metadata is required"; exit 1; }

apt-get update -qq && apt-get install -y -qq git ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

git clone --branch "$BRANCH" --depth 1 "$REPO" /opt/pitvis
cd /opt/pitvis

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
echo "=== pulling frames from $BUCKET ==="
gsutil -m rsync -r "$BUCKET/frames" data/frames
# Resume anything a previous (possibly preempted) run finished.
gsutil -m rsync -r "$BUCKET/out/backbone" data/backbone || true

export EPOCHS BACKBONE
exec uv run bash infra/run_job.sh
