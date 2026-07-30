"""PitVis per-video feature/label loading and the train/val split.

Split from Das et al. 2024 (arXiv 2409.01184): 20 train / 5 val, chosen for an
approximate 4:1 per-class annotation ratio. Video 19 has no annotations (gap in
the download), so the effective split is 19 train / 5 val. Kept as explicit
constants — do not derive by arithmetic.

Labels use the 15-way encoding: 0 = background (-1 raw), k = step k (1..14).
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"

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
