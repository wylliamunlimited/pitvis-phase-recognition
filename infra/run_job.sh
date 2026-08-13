#!/usr/bin/env bash
# The GPU job: per-fold backbone fine-tuning, then extraction and scoring.
#
# WHY PER FOLD. A backbone fine-tuned on all 19 training videos, then
# cross-validated over folds drawn from that same set, produces features that
# already encode the held-out videos' labels. We measured that: steps macro
# went 0.504 -> 0.917, which is the size of the leak, not an improvement.
# `crossval.check_no_leak` refuses that configuration outright.
#
# So: one encoder per fold, each trained with that fold's videos excluded, plus
# one trained on all of TRAIN for the single VAL scoring. Six fine-tunes. That
# is ~6x a single run, which is exactly why this is a cloud job and not a
# laptop one.
#
# WHICH BACKBONE. Default is DINOv2 ViT-B/14, not ResNet-50. The pilot went to
# ResNet-50 because it trains 3x faster, and it answered the question it was
# meant to (mean AP 0.271 -> 0.445, 19/19 classes) — but end to end it does not
# beat a FROZEN DINOv2: 0.4425 steps / 0.3805 instruments against 0.4610 /
# 0.5572. Fine-tuning the encoder that already wins frozen is the untested
# combination. `BACKBONE=resnet50` still reproduces the cheaper arm.
#
# Idempotent: every stage skips work already on disk, so a preempted instance
# resumes rather than restarting.

set -euo pipefail

BACKBONE="${BACKBONE:-vit_base_patch14_dinov2.lvd142m}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"
SIZE="${SIZE:-384}"

# STAGE=full runs ONE fine-tune (all of TRAIN) instead of six, then extracts
# and scores. That is the ~1/6 cost pass that answers the actual question —
# "does a fine-tuned encoder beat the frozen one on VAL" — and it is the right
# first spend. The five per-fold encoders exist only to make an honest RANKING
# possible later, and there is no point ranking until the headline moves.
STAGE="${STAGE:-all}"
case "$STAGE" in
  all|full) ;;
  *) echo "STAGE must be 'all' (6 fine-tunes) or 'full' (1)" >&2; exit 1 ;;
esac

# One choice drives the tag, the space and the input size, so they cannot
# drift. They MUST agree: a space names the checkpoint path it loads, and a
# fine-tuned encoder inferred at a resolution it was not tuned at measures the
# mismatch as much as the model.
case "$BACKBONE" in
  resnet50)
    FT_TAG=resnet50; SPACE=resnet50_ft; FROZEN=resnet50; IMG=() ;;
  vit_base_patch14_dinov2.lvd142m)
    # ViT: without --img-size the model is built at the checkpoint's native
    # 518 and rejects the 224 crop the frame dataset produces.
    FT_TAG=dinov2; SPACE=dinov2_ft; FROZEN=dinov2_vitb14; IMG=(--img-size 224) ;;
  *)
    echo "unknown BACKBONE '$BACKBONE' — add a case here with its tag, space," \
         "frozen counterpart and input size before running a multi-hour job" >&2
    exit 1 ;;
esac

FULL_TAG="${FT_TAG}-${EPOCHS}ep"

# Push what is finished, the moment it is finished.
#
# An encoder is the expensive thing here — an hour of GPU time each — and a spot
# preemption gives about 30 seconds of warning, which is not enough to upload
# 2 GB. Saving after each one means a preemption costs the fold in flight and
# nothing else, and startup.sh re-syncs `out/backbone` on boot so the next
# instance skips everything already done. Without this, "idempotent and
# resumable" is a claim the job cannot actually honour.
save_now() {                       # save_now <label> [dir=backbone]
  [ -n "${BUCKET:-}" ] || return 0
  local dir="${2:-backbone}"
  echo "--- saving $1 -> $BUCKET/out/$dir"
  gsutil -m rsync -r "data/$dir" "$BUCKET/out/$dir" \
    || echo "!!! upload of $1 FAILED — it exists only on this disk"
}

cd "$(dirname "$0")/.."
echo "=== pitvis GPU job ==="
echo "backbone $BACKBONE   epochs $EPOCHS   space $SPACE   tag $FULL_TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# Fail in the first second, not the seventh hour. The space hardcodes the
# checkpoint path it will load; if EPOCHS makes FULL_TAG disagree with it, the
# job would fine-tune for hours and then extract from the wrong weights (or
# none). Checked before any GPU time is spent.
WANT=$(python -c "from pitvis.data import spaces; print(spaces.get('$SPACE').checkpoint)")
if [ "$WANT" != "backbone/${FULL_TAG}/backbone.pt" ]; then
  echo "MISMATCH: space '$SPACE' loads '$WANT', this job would write" \
       "'backbone/${FULL_TAG}/backbone.pt'." >&2
  echo "Fix: set EPOCHS so the tags agree, or update the space's checkpoint" \
       "path in src/pitvis/data/spaces.py." >&2
  exit 1
fi

# ---- 1. one backbone per fold, each blind to that fold's held-out videos ----
# Tags carry the backbone: two backbones' folds are different encoders and must
# not land on the same directory. (A variant/space collision of exactly this
# shape overwrote a winning checkpoint once already.)
if [ "$STAGE" = all ]; then
  FOLDS=$(python -c "from pitvis.data.folds import FOLDS_5; print('|'.join(' '.join(map(str,f)) for f in FOLDS_5))")
  i=0
  IFS='|' read -ra ARR <<< "$FOLDS"
  for fold in "${ARR[@]}"; do
    tag="${FT_TAG}-fold${i}"
    if [ -f "data/backbone/${tag}/backbone.pt" ]; then
      echo "--- ${tag}: exists, skipping"
    else
      echo "--- ${tag}: fine-tuning, holding out ${fold}"
      pitvis-finetune --backbone "$BACKBONE" "${IMG[@]}" --epochs "$EPOCHS" \
        --batch "$BATCH" --workers "$WORKERS" --size "$SIZE" --device cuda \
        --exclude $fold --tag "$tag"
      save_now "$tag"
    fi
    i=$((i+1))
  done
else
  echo "--- STAGE=full: skipping the 5 per-fold encoders (no honest CV from this run)"
fi

# ---- 2. one trained on all of TRAIN, for the single VAL scoring -------------
if [ -f "data/backbone/${FULL_TAG}/backbone.pt" ]; then
  echo "--- ${FULL_TAG}: exists, skipping"
else
  echo "--- ${FULL_TAG}: fine-tuning on all of TRAIN (VAL is never in TRAIN)"
  pitvis-finetune --backbone "$BACKBONE" "${IMG[@]}" --epochs "$EPOCHS" \
    --batch "$BATCH" --workers "$WORKERS" --size "$SIZE" --device cuda \
    --tag "$FULL_TAG"
  save_now "$FULL_TAG"
fi

# ---- 3. features + the diagnostic that decided this was worth doing ---------
echo "--- extracting features for $SPACE"
pitvis-extract --space "$SPACE" --device cuda
pitvis-verify  --space "$SPACE"
# ~350 MB, and cheap to re-extract from the backbone — but only if the backbone
# survived. Pushed here anyway so a laptop can skip the re-extraction entirely.
save_now "features ($SPACE)" features

# Frozen vs fine-tuned of the SAME backbone. Comparing a fine-tuned ViT against
# a frozen ResNet-50 would confound the two changes we are trying to separate.
pitvis-probe --space "$FROZEN" --space "$SPACE" \
  | tee "data/backbone/probe-${FULL_TAG}.txt"
save_now "probe report"

echo "=== job complete ==="
echo "next, on a machine with the features: uv run pitvis-train arst-v2 \\"
echo "        --variant best --space $SPACE      # lands in v2/best@${SPACE}/"
du -sh data/backbone data/features
