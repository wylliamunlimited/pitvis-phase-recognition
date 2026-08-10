#!/usr/bin/env bash
# The GPU job: per-fold backbone fine-tuning, then extraction and scoring.
#
# WHY PER FOLD. A backbone fine-tuned on all 19 training videos, then
# cross-validated over folds drawn from that same set, produces features that
# already encode the held-out videos' labels. We measured that: steps macro
# went 0.504 -> 0.917, which is the size of the leak, not an improvement.
# `crossval.check_no_leak` now refuses that configuration outright.
#
# So: one encoder per fold, each trained with that fold's videos excluded, plus
# one trained on all of TRAIN for the single VAL scoring. Six fine-tunes. That
# is ~6x a single run, which is exactly why this is a cloud job and not a
# laptop one.
#
# Idempotent: every stage skips work already on disk, so a preempted instance
# resumes rather than restarting.

set -euo pipefail

EPOCHS="${EPOCHS:-50}"
BACKBONE="${BACKBONE:-resnet50}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
SIZE="${SIZE:-384}"

cd "$(dirname "$0")/.."
echo "=== pitvis GPU job ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

FOLDS=$(python -c "from pitvis.data.folds import FOLDS_5; print('|'.join(' '.join(map(str,f)) for f in FOLDS_5))")

# ---- 1. one backbone per fold, each blind to that fold's held-out videos ----
i=0
IFS='|' read -ra ARR <<< "$FOLDS"
for fold in "${ARR[@]}"; do
  tag="fold${i}"
  if [ -f "data/backbone/${tag}/backbone.pt" ]; then
    echo "--- ${tag}: exists, skipping"
  else
    echo "--- ${tag}: fine-tuning, holding out ${fold}"
    pitvis-finetune --backbone "$BACKBONE" --epochs "$EPOCHS" --batch "$BATCH" \
      --workers "$WORKERS" --size "$SIZE" --device cuda \
      --exclude $fold --tag "$tag"
  fi
  i=$((i+1))
done

# ---- 2. one trained on all of TRAIN, for the single VAL scoring -------------
if [ ! -f "data/backbone/${BACKBONE}-${EPOCHS}ep/backbone.pt" ]; then
  echo "--- full: fine-tuning on all of TRAIN (VAL is never in TRAIN)"
  pitvis-finetune --backbone "$BACKBONE" --epochs "$EPOCHS" --batch "$BATCH" \
    --workers "$WORKERS" --size "$SIZE" --device cuda
fi

# ---- 3. features + the diagnostic that decided this was worth doing ---------
echo "--- extracting features for the full backbone"
pitvis-extract --space "${SPACE:-resnet50_ft}" --device cuda
pitvis-verify --space "${SPACE:-resnet50_ft}"
pitvis-probe --space resnet50 --space "${SPACE:-resnet50_ft}" \
  | tee "data/backbone/probe-${BACKBONE}-${EPOCHS}ep.txt"

echo "=== job complete ==="
du -sh data/backbone data/features
