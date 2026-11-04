"""The feature spaces the cache can hold — one entry per backbone.

A *feature space* is the whole image->numbers function: which backbone, at what
resolution, with which preprocessing, sampled at what rate. Two caches built by
different spaces are not interchangeable, and mixing them in one training run
produces a model that is silently wrong rather than loudly broken.

`extract_features.py` already computes a content hash of that definition
(`space_id`) and refuses to write into a cache whose hash disagrees. What it
could not do was hold two spaces at once: the guard's remedy was "delete
data/features/ and re-extract", so trying a second backbone meant destroying
the first. This module names the spaces, and `paths.features_dir` gives each
one its own directory, so they coexist.

Same shape as `training/registry.py`, for the same reason: one list is the
source of truth, `--space` choices derive from it, and adding a backbone is one
entry rather than an edit in six modules.

THE HASHED PAYLOAD IS FROZEN. `space_id` hashes
`{backbone, feature_dim, target_fps, transform}` and nothing else. `name` is
deliberately absent from it — adding a key would move the existing cache's id
off `67912d3efc6852e7`, and the guard would then reject 940 MB of correct
features and demand a 25-minute re-extract for no reason. The human-readable
name lives in the directory and in this registry; never in the hash.

PER-FOLD SPACES ARE DERIVED, NOT LISTED. An honest cross-validation of a
fine-tuned encoder needs *one encoder per fold*, each blind to that fold's
held-out videos, so fold `i` must read its features from a different cache than
fold `j` — which a single `--space` cannot express. `fold(base, i)` derives
`<base>@fold<i>` from a template the fine-tuned space declares
(`fold_checkpoint`), and `get` resolves that name, so every `--space` reader
reaches a fold space with no change. They are derived rather than five more
registry entries because the fold *count* is owned by `data/folds.py`: hand-
listing them would be a second place for k to be wrong.

The payload is untouched by this. `fold_checkpoint` is not hashed — only the
resolved `checkpoint`'s WEIGHT digest is, and that already differs per fold
because each fold trained on a different subset. So the fold caches get
distinct ids for the right reason, and `resnet50`'s id does not move.
"""

import re

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Space:
    """One backbone's feature space.

    `model_kwargs` reaches `timm.create_model` untouched. DINOv2 needs it:
    its checkpoint ships at 518x518, which costs 6.4x the compute of 224 for a
    37x37 patch grid that we then average-pool down to the same 768 numbers.
    At 224 the grid is 16x16 — still 5x ResNet-50's 7x7, and measured *faster*
    than ConvNeXtV2 at the same resolution.
    """

    name: str
    backbone: str
    summary: str
    # Declared, not discovered. The hashed payload still takes the width from
    # `model.num_features` at extraction time, so `space_id` is untouched by
    # this field -- but that number only exists once the backbone has been
    # built with pretrained weights, i.e. after a download, and only survives
    # in the cache's manifest.json. Anything that needs the width WITHOUT a
    # cache (the `pitvis-models` trace on a fresh clone) had nowhere to read
    # it and fell back to a hardcoded 2048, silently tracing the wrong model
    # for a 768-d space. `build_model` asserts the two agree, so the
    # declaration cannot drift from the backbone it describes.
    feature_dim: int
    target_fps: int = 1
    model_kwargs: dict = field(default_factory=dict)
    # Fine-tuned weights, relative to data/. None means the timm pretrained
    # weights, i.e. an off-the-shelf encoder.
    checkpoint: str | None = None
    # Where the PER-FOLD encoders live, as a template over the fold index —
    # `fold(name, i)` fills `{fold}`. Only a fine-tuned space has them, and
    # only `infra/run_job.sh` STAGE=all produces them, one per fold of
    # `data/folds.py`. Declared rather than derived from `checkpoint` because
    # the fold tags carry no epoch count (`dinov2-fold0`, not
    # `dinov2-fold0-50ep`), so no string edit of the full tag reaches them.
    # `tests/test_spaces.py` pins these against the tags the job writes.
    fold_checkpoint: str | None = None
    # "video" decodes the mp4; "frames" reads the JPEG cache. A fine-tuned
    # encoder MUST read frames: it was tuned on 384px centre-square JPEGs
    # cropped to 224, and feeding it pixels framed differently would measure
    # the preprocessing mismatch as much as the model.
    source: str = "video"
    frame_size: int = 384


SPACES: dict[str, Space] = {
    s.name: s
    for s in [
        Space(
            name="resnet50",
            feature_dim=2048,
            backbone="resnet50",
            summary="ImageNet ResNet-50, 2048-d, 224px — the original cache",
        ),
        Space(
            name="resnet50_ft",
            feature_dim=2048,
            backbone="resnet50",
            summary="ResNet-50 fine-tuned on PitVis frames — surgical-specific",
            checkpoint="backbone/resnet50-5ep/backbone.pt",
            fold_checkpoint="backbone/resnet50-fold{fold}/backbone.pt",
            source="frames",
        ),
        Space(
            name="dinov2_vitb14",
            feature_dim=768,
            backbone="vit_base_patch14_dinov2.lvd142m",
            summary="DINOv2 ViT-B/14, 768-d, 224px — self-supervised, 16x16 grid",
            model_kwargs={"img_size": 224},
        ),
        # The one the evidence points at. Fine-tuning was piloted on ResNet-50
        # because ViT-B trains at 29 img/s against 96 — the cheap backbone
        # answers "does fine-tuning help at all" for a quarter of the cost, and
        # it does (mean AP 0.271 -> 0.445, 19/19 classes). But it was applied
        # to the encoder that LOSES frozen: end to end the fine-tuned ResNet-50
        # scores 0.4425 steps / 0.3805 instruments against frozen DINOv2's
        # 0.4610 / 0.5572. So this space is that lever aimed at the winner.
        #
        # `img_size` MUST match what pitvis-finetune was given, or the encoder
        # is tuned at one resolution and inferred at another. Produced by:
        #   uv run pitvis-finetune --backbone vit_base_patch14_dinov2.lvd142m \
        #       --img-size 224 --tag dinov2-50ep --epochs 50 --device cuda
        Space(
            name="dinov2_ft",
            feature_dim=768,
            backbone="vit_base_patch14_dinov2.lvd142m",
            summary="DINOv2 ViT-B/14 fine-tuned on PitVis frames — 768-d, 224px",
            model_kwargs={"img_size": 224},
            checkpoint="backbone/dinov2-50ep/backbone.pt",
            fold_checkpoint="backbone/dinov2-fold{fold}/backbone.pt",
            source="frames",
        ),
    ]
}

# What every reader gets when it is not told otherwise. Changing this changes
# which cache the whole project trains on, so it is a one-line decision rather
# than a default scattered across call sites.
DEFAULT = "resnet50"


# `<base>@fold<i>`. The separator is a sigil rather than a `-` or a `_` so a
# derived name can never collide with a registered one, and so the base is
# recoverable from the name by a split -- which is what lets a CV entry record
# "these five caches are one family" without a second field.
FOLD_SEP = "@fold"
_FOLD_RE = re.compile(rf"^(?P<base>.+){re.escape(FOLD_SEP)}(?P<fold>\d+)$")


def fold_name(base: str, fold: int) -> str:
    """The cache name fold `fold` reads: `dinov2_ft@fold0`."""
    return f"{base}{FOLD_SEP}{fold}"


def split_fold(name: str) -> tuple[str, int] | None:
    """`('dinov2_ft', 0)` for a per-fold name, None for a plain one."""
    m = _FOLD_RE.match(name)
    return (m["base"], int(m["fold"])) if m else None


def fold(base: str, index: int) -> Space:
    """The per-fold space derived from `base` — a different ENCODER, not a view.

    Fold `index`'s encoder was fine-tuned with that fold's held-out videos
    excluded, so its features are a different function of the same pixels and
    belong in a different cache. `crossval.check_no_leak` then has something to
    check: the manifest of this cache records which videos its encoder saw.

    Refuses a frozen base rather than inventing a path. A frozen encoder has
    nothing to hold out and needs no fold spaces — the same cache is honest for
    every fold, which is why `cross_validate` only derives these when asked.
    """
    parent = get(base)
    if parent.fold_checkpoint is None:
        raise SystemExit(
            f"space {base!r} declares no per-fold encoders"
            + (" — it is frozen, so every fold may read it directly and "
               "`--per-fold-encoders` is unnecessary."
               if parent.checkpoint is None else
               f" — it is fine-tuned ({parent.checkpoint!r}) but no "
               f"`fold_checkpoint` template is declared for it in "
               f"data/spaces.py, so the per-fold caches cannot be named.")
        )
    return replace(parent, name=fold_name(base, index),
                   checkpoint=parent.fold_checkpoint.format(fold=index),
                   summary=f"{parent.summary} — fold {index}'s encoder only")


def get(name: str) -> Space:
    """Look up a space, naming the registered ones when it is missing.

    Resolves a derived `<base>@fold<i>` too, so `pitvis-extract --space
    dinov2_ft@fold0` and everything downstream of it work without the registry
    listing five more entries per backbone.
    """
    if name in SPACES:
        return SPACES[name]
    derived = split_fold(name)
    if derived:
        return fold(*derived)
    raise SystemExit(
        f"unknown feature space {name!r}. Registered: {', '.join(names())}"
        f" (plus <space>{FOLD_SEP}<i> for a fine-tuned space's per-fold "
        f"encoders)"
    )


def space_name(value: str) -> str:
    """argparse `type=` for `--space`: validate, return the name unchanged.

    Replaces `choices=spaces.names()` at every call site. `choices` could not
    admit a derived `@fold<i>` name without listing ten of them in every
    `--help`, and eight copies of the same choices list was already one owner
    too many. Validation still happens at parse time, before any work.

    `argparse` is imported here, not at module scope: this module sits on
    `dataset.py`'s import path, so it is reached by `pitvis-app` and by
    `pitvis-predict --list-models`, and 5.7 ms of argparse for a function only
    a parser ever calls is the same toll the sklearn imports were.
    """
    import argparse

    try:
        get(value)
    except SystemExit as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return value


def names() -> list[str]:
    """Registered space names, default first. Derived fold names are excluded."""
    rest = sorted(n for n in SPACES if n != DEFAULT)
    return [DEFAULT, *rest]


def describe() -> str:
    """One line per space, for `--list`-style output."""
    w = max(len(n) for n in SPACES)
    lines = [f"  {s.name:<{w}}  {s.summary}" for s in (SPACES[n] for n in names())]
    folded = [n for n in names() if SPACES[n].fold_checkpoint]
    if folded:
        lines.append(f"  {'':<{w}}  (each of {', '.join(folded)} also derives "
                     f"<space>{FOLD_SEP}<i> — one encoder per CV fold)")
    return "\n".join(lines)
