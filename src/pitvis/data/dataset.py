"""PitVis per-video feature/label loading and the train/val split.

Split from Das et al. 2024 (arXiv 2409.01184): 20 train / 5 val, chosen for an
approximate 4:1 per-class annotation ratio. Video 19 has no annotations (gap in
the download), so the effective split is 19 train / 5 val. Kept as explicit
constants — do not derive by arithmetic.

Labels use the 15-way encoding: 0 = background (-1 raw), k = step k (1..14).
"""

import numpy as np

from pitvis.paths import FEATURES

VAL = [1, 12, 21, 24, 25]
TRAIN = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 20, 22, 23]

NUM_CLASSES = 15
BACKGROUND = 0


def load_video(vid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (features (T, D) float32, labels (T,) int64) for one video."""
    d = FEATURES / f"video_{vid:02d}"
    features = np.load(d / "features.npy")
    labels = np.load(d / "labels.npy")
    assert len(features) == len(labels), \
        f"video {vid}: {len(features)} features vs {len(labels)} labels"
    return features, labels


def load_split(videos: list[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return [(vid, features, labels), ...] for the given video list."""
    return [(vid, *load_video(vid)) for vid in videos]


def load_video_instruments(vid: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (features (T, D) float32, instruments (T, 2) int64) for one video.

    Deliberately a second function rather than a wider return from `load_video`:
    the `(vid, features, labels)` triple is destructured positionally in five
    call sites, so widening it would break every one silently.

    Instrument values are RAW — -1 (out of patient), -2 (no secondary) and 0
    (nothing visible) are three distinct states. Convert to multi-hot at the
    point of use, not here.
    """
    d = FEATURES / f"video_{vid:02d}"
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


def load_split_instruments(videos: list[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return [(vid, features, instruments), ...] for the given video list."""
    return [(vid, *load_video_instruments(vid)) for vid in videos]
