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
"""

from dataclasses import dataclass, field


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
    target_fps: int = 1
    model_kwargs: dict = field(default_factory=dict)
    # Fine-tuned weights, relative to data/. None means the timm pretrained
    # weights, i.e. an off-the-shelf encoder.
    checkpoint: str | None = None
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
            backbone="resnet50",
            summary="ImageNet ResNet-50, 2048-d, 224px — the original cache",
        ),
        Space(
            name="resnet50_ft",
            backbone="resnet50",
            summary="ResNet-50 fine-tuned on PitVis frames — surgical-specific",
            checkpoint="backbone/resnet50-5ep/backbone.pt",
            source="frames",
        ),
        Space(
            name="dinov2_vitb14",
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
            backbone="vit_base_patch14_dinov2.lvd142m",
            summary="DINOv2 ViT-B/14 fine-tuned on PitVis frames — 768-d, 224px",
            model_kwargs={"img_size": 224},
            checkpoint="backbone/dinov2-50ep/backbone.pt",
            source="frames",
        ),
    ]
}

# What every reader gets when it is not told otherwise. Changing this changes
# which cache the whole project trains on, so it is a one-line decision rather
# than a default scattered across call sites.
DEFAULT = "resnet50"


def get(name: str) -> Space:
    """Look up a space, naming the registered ones when it is missing."""
    try:
        return SPACES[name]
    except KeyError:
        raise SystemExit(
            f"unknown feature space {name!r}. Registered: {', '.join(names())}"
        ) from None


def names() -> list[str]:
    """Registered space names, default first — suitable for argparse choices."""
    rest = sorted(n for n in SPACES if n != DEFAULT)
    return [DEFAULT, *rest]


def describe() -> str:
    """One line per space, for `--list`-style output."""
    w = max(len(n) for n in SPACES)
    return "\n".join(f"  {s.name:<{w}}  {s.summary}" for s in
                     (SPACES[n] for n in names()))
