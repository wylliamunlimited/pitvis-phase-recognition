"""PitVis per-video feature/label loading and the train/val split.

Split from Das et al. 2024 (arXiv 2409.01184): 20 train / 5 val, chosen for an
approximate 4:1 per-class annotation ratio. Video 19 has no annotations (gap in
the download), so the effective split is 19 train / 5 val. Kept as explicit
constants — do not derive by arithmetic.

Labels use the 15-way encoding: 0 = background (-1 raw), k = step k (1..14).
"""

import numpy as np

from pitvis.data import spaces
from pitvis.paths import video_dir

VAL = [1, 12, 21, 24, 25]
TRAIN = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 22, 23]

NUM_CLASSES = 15
BACKGROUND = 0

# The one definition. These strings existed twice — once keyed by the encoded
# label and once by the raw one — which is two chances for them to drift and no
# way to tell which copy a caller meant. They live here because both key spaces
# are defined here, and because `evaluation/metric.py` already imports from this
# module, so consolidating adds no dependency edge.
#
# Keyed ENCODED (0..14). The raw labels the CSVs and `predictions.csv` use are
# the same integers except that background is -1, not 0 — ask `step_name` for
# that key space rather than building a second dict.
#
# Names are the cleaned forms: `map_steps.csv` has a trailing space on step 1
# and snake_case on step 9, and it maps -1 to three different strings, so it is
# not loadable as an int -> str dict at all. See notes/data-dictionary.md.
STEP_NAMES = {
    0: "background", 1: "nasal corridor creation", 2: "anterior sphenoidotomy",
    3: "septum displacement", 4: "sphenoid sinus clearance", 5: "sellotomy",
    6: "durotomy", 7: "tumour excision", 8: "haemostasis",
    9: "synthetic graft placement", 10: "fat graft placement",
    11: "gasket seal construct", 12: "dural sealant", 13: "nasal packing",
    14: "debris clearance",
}


def step_name(k: int, *, raw: bool = False, default: str = "?") -> str:
    """Human-readable name for a step label.

    `raw=True` reads `k` in the challenge's own encoding, where background is
    -1 rather than 0 — the encoding used by `annotations_*.csv`, by
    `predictions.csv` and by `segments.csv`.
    """
    if raw and k == -1:
        k = BACKGROUND
    return STEP_NAMES.get(k, default)


def load_video(vid: int, space: str = spaces.DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Return (features (T, D) float32, labels (T,) int64) for one video.

    `space` selects which backbone's cache to read. D varies by space — 2048
    for resnet50, 768 for dinov2_vitb14 — so a model trained on one is not
    loadable against the other, which is the point of keeping them separate.
    """
    d = video_dir(space, vid)
    features = np.load(d / "features.npy")
    labels = np.load(d / "labels.npy")
    assert len(features) == len(labels), \
        f"video {vid}: {len(features)} features vs {len(labels)} labels"
    return features, labels


def load_split(videos: list[int],
               space: str = spaces.DEFAULT) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return [(vid, features, labels), ...] for the given video list."""
    return [(vid, *load_video(vid, space)) for vid in videos]


def load_video_instruments(vid: int,
                           space: str = spaces.DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """Return (features (T, D) float32, instruments (T, 2) int64) for one video.

    Deliberately a second function rather than a wider return from `load_video`:
    the `(vid, features, labels)` triple is destructured positionally in five
    call sites, so widening it would break every one silently.

    Instrument values are RAW — -1 (out of patient), -2 (no secondary) and 0
    (nothing visible) are three distinct states. Convert to multi-hot at the
    point of use, not here.
    """
    d = video_dir(space, vid)
    features = np.load(d / "features.npy")
    path = d / "instruments.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `uv run pitvis-extract` to backfill it "
            f"(features are reused, nothing is re-decoded)"
        )
    instruments = np.load(path)
    assert len(features) == len(instruments), \
        f"video {vid}: {len(features)} features vs {len(instruments)} instrument rows"
    return features, instruments


def load_split_instruments(
    videos: list[int], space: str = spaces.DEFAULT
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return [(vid, features, instruments), ...] for the given video list."""
    return [(vid, *load_video_instruments(vid, space)) for vid in videos]
