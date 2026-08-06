"""Predict surgical steps for a video, from an mp4 to per-second labels.

This is the piece that makes the repo usable rather than only measurable.
Everything else assumes a video that is already in the feature cache *with*
annotations; this accepts a path to any video and needs no labels at all.

The path:

    video.mp4 --ffmpeg 1 fps--> frames --frozen resnet50--> (T, 2048)
              --standardize--> --spatial--> --TeCNO--> --ARST + CCI--> (T,)

Two output shapes, because they answer different questions:

- **per-second labels** in the challenge's own `int_time,int_step` encoding
  (background is -1, not 0) — directly comparable to `annotations_{n}.csv`
- **segments** `(start_s, end_s, step, duration_s)` — what a human or a
  downstream app actually reads; a 7,201-row CSV is not a result anyone looks at

Labels are optional. Supply them and the official metric is computed too.
"""

from __future__ import annotations

import subprocess
from itertools import groupby
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from pitvis.data.dataset import NUM_CLASSES
from pitvis.evaluation.metric import decode
from pitvis.models.arst import ARST, SpatialEmbedding, TeCNO
from pitvis.paths import CKPT, FEATURES, MANIFEST


def require_ffmpeg() -> None:
    """Fail with a fixable message rather than a FileNotFoundError traceback."""
    import shutil
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} not found on PATH. These are not Python "
            f"packages — install them separately (macOS: brew install ffmpeg)."
        )


def cached_features(video: Path) -> np.ndarray | None:
    """Return cached features for `video` if the cache holds this exact file.

    Only reuses the cache when the manifest records this video at this path in
    the *current* feature space — a cache from a different backbone would
    silently produce predictions the checkpoint was never trained for.
    """
    import json
    if not MANIFEST.exists():
        return None
    manifest = json.loads(MANIFEST.read_text())
    for key, entry in manifest.get("videos", {}).items():
        if Path(entry["source"]).resolve() != video.resolve():
            continue
        path = FEATURES / key / "features.npy"
        if not path.exists():
            return None
        feats = np.load(path)
        if len(feats) != entry["frames"]:
            return None
        return feats
    return None


def embed(video: Path, device: torch.device) -> np.ndarray:
    """Decode and embed `video` — the same code path extraction uses."""
    from pitvis.data.extract_features import build_model, embed_video
    require_ffmpeg()
    model, transform, _ = build_model(device)
    features, _ = embed_video(video, model, transform, device, tag=video.name)
    return features


def load_checkpoint(ckpt_path: Path, std_path: Path, feature_dim: int,
                    device: torch.device, width: int | None = None):
    """Rebuild the three frozen stages from a checkpoint."""
    for p, what in [(ckpt_path, "checkpoint"), (std_path, "standardisation stats")]:
        if not p.exists():
            raise SystemExit(
                f"{what} not found at {p} — train a model first:\n"
                f"    uv run pitvis-train arst"
            )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    trained = ckpt["args"]
    w = width if width is not None else trained["width"]

    spatial = SpatialEmbedding(feature_dim, num_classes=NUM_CLASSES).to(device)
    tecno = TeCNO(num_classes=NUM_CLASSES).to(device)
    arst = ARST(num_classes=NUM_CLASSES, width=w).to(device)
    spatial.load_state_dict(ckpt["spatial"])
    tecno.load_state_dict(ckpt["tecno"])
    arst.load_state_dict(ckpt["arst"])
    spatial.eval(), tecno.eval(), arst.eval()

    stats = np.load(std_path)
    return spatial, tecno, arst, stats["mean"], stats["std"], trained, w


@torch.no_grad()
def predict(features: np.ndarray, spatial, tecno, arst, mean, std,
            device: torch.device, chunk: int, cci: bool,
            mask_excluded: bool) -> np.ndarray:
    """Run the cascade. Returns (T,) encoded predictions (0 = background)."""
    from pitvis.training.arst import cci_decode
    x = torch.from_numpy((features - mean) / std).to(device)
    z, _ = spatial(x)
    _, ft = tecno(z.unsqueeze(0))
    opts = SimpleNamespace(chunk=chunk, cci=cci, mask_excluded=mask_excluded)
    return cci_decode(arst, ft, opts, device)


def to_segments(preds: np.ndarray) -> pd.DataFrame:
    """Collapse per-second predictions into contiguous runs.

    Emitted in the challenge's raw encoding (background -1), so a segment table
    reads the same way the annotations do.
    """
    rows, t = [], 0
    for step, group in groupby(decode(preds)):
        n = len(list(group))
        rows.append({"start_s": t, "end_s": t + n - 1, "int_step": step,
                     "duration_s": n})
        t += n
    return pd.DataFrame(rows)


def load_labels(path: Path, expected: int) -> np.ndarray:
    """Load ground truth for scoring, from a .npy or an annotations CSV.

    Applies the same truncation extraction does — annotation files carry one
    row more than there are frames, and the extra row is background.
    """
    if path.suffix == ".npy":
        labels = np.load(path)
    else:
        steps = pd.read_csv(path)["int_step"].to_numpy()
        if len(steps) == expected + 1:
            if steps[-1] != -1:
                raise SystemExit(f"{path}: dropped trailing row is not background")
            steps = steps[:expected]
        labels = steps.copy()
        labels[labels == -1] = 0
    if len(labels) != expected:
        raise SystemExit(
            f"{path}: {len(labels)} labels but {expected} predicted frames"
        )
    return labels.astype(np.int64)
