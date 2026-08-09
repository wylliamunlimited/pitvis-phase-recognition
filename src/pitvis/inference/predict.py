"""Predict surgical steps for a video, from an mp4 to per-second labels.

This is the piece that makes the repo usable rather than only measurable.
Everything else assumes a video that is already in the feature cache *with*
annotations; this accepts a path to any video and needs no labels at all.

The path:

    video.mp4 --ffmpeg 1 fps--> frames --frozen resnet50--> (T, 2048)
              --standardize--> --spatial--> --TeCNO--> --ARST + CCI--> (T,)

Both challenge tasks run off the same features — one decode, two models:

    task 1  spatial -> TeCNO -> ARST + CCI    -> one step per second
    task 2  causal window -> LSTM -> sigmoid  -> up to two instruments per second

Two output shapes for steps, because they answer different questions:

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
from pitvis.data import spaces
from pitvis.paths import CKPT, CKPT_INSTRUMENTS, manifest_path, video_dir


def require_ffmpeg() -> None:
    """Fail with a fixable message rather than a FileNotFoundError traceback."""
    import shutil
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise SystemExit(
            f"{' and '.join(missing)} not found on PATH. These are not Python "
            f"packages — install them separately (macOS: brew install ffmpeg)."
        )


def cached_features(video: Path, space: str = spaces.DEFAULT) -> np.ndarray | None:
    """Return cached features for `video` if the cache holds this exact file.

    Only reuses the cache when the manifest records this video at this path in
    the *current* feature space — a cache from a different backbone would
    silently produce predictions the checkpoint was never trained for.
    """
    import json
    mpath = manifest_path(space)
    if not mpath.exists():
        return None
    manifest = json.loads(mpath.read_text())
    for key, entry in manifest.get("videos", {}).items():
        if Path(entry["source"]).resolve() != video.resolve():
            continue
        path = video_dir(space, int(key.removeprefix("video_"))) / "features.npy"
        if not path.exists():
            return None
        feats = np.load(path)
        if len(feats) != entry["frames"]:
            return None
        return feats
    return None


def embed(video: Path, device: torch.device,
          space: str = spaces.DEFAULT) -> np.ndarray:
    """Decode and embed `video` — the same code path extraction uses."""
    from pitvis.data.extract_features import build_model, embed_video
    require_ffmpeg()
    model, transform, _ = build_model(device, spaces.get(space))
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
    # Tags recorded at training time, each defaulting to what citi.pt -- which
    # predates all of them -- was trained with. `mask_excluded` matters: the
    # step winner masks 0/11/13 out of the argmax, and ignoring that here would
    # quietly discard most of its advantage.
    meta = {"space": ckpt.get("space", spaces.DEFAULT),
            "variant": ckpt.get("variant", "reproduction"),
            "mask_excluded": bool(ckpt.get("mask_excluded", False))}

    spatial = SpatialEmbedding(feature_dim, num_classes=NUM_CLASSES).to(device)
    tecno = TeCNO(num_classes=NUM_CLASSES).to(device)
    arst = ARST(num_classes=NUM_CLASSES, width=w).to(device)
    spatial.load_state_dict(ckpt["spatial"])
    tecno.load_state_dict(ckpt["tecno"])
    arst.load_state_dict(ckpt["arst"])
    spatial.eval(), tecno.eval(), arst.eval()

    stats = np.load(std_path)
    return spatial, tecno, arst, stats["mean"], stats["std"], trained, w, meta


@torch.no_grad()
def predict(features: np.ndarray, spatial, tecno, arst, mean, std,
            device: torch.device, chunk: int, cci: bool,
            mask_excluded: bool, *, return_probs: bool = False):
    """Run the cascade. Returns (T,) encoded predictions (0 = background).

    With `return_probs`, also returns the decoder's (T, 15) softmax — see
    `cci_decode` for why that distribution is pre-CCI and what that means.
    """
    from pitvis.training.arst import cci_decode
    x = torch.from_numpy((features - mean) / std).to(device)
    z, _ = spatial(x)
    _, ft = tecno(z.unsqueeze(0))
    opts = SimpleNamespace(chunk=chunk, cci=cci, mask_excluded=mask_excluded)
    return cci_decode(arst, ft, opts, device, return_probs=return_probs)


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


def step_space(ckpt_path: Path) -> str:
    """Which feature space a task-1 checkpoint expects.

    Read before any features are computed, because it decides which backbone
    has to run. citi.pt predates the multi-space cache and carries no `space`.
    """
    if not ckpt_path.exists():
        return spaces.DEFAULT
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("space", spaces.DEFAULT)


def instrument_space(ckpt_path: Path) -> str:
    """Which feature space a task-2 checkpoint expects.

    Read BEFORE the features are computed, because it decides which backbone
    has to run. SANO's original sano.pt predates the multi-space cache and
    carries no `space` key at all — absent means resnet50, which is what it
    was trained on.
    """
    if not ckpt_path.exists():
        return spaces.DEFAULT
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt.get("space", spaces.DEFAULT)


def load_instrument_checkpoint(ckpt_path: Path, std_path: Path, feature_dim: int,
                               device: torch.device):
    """Rebuild a task-2 model from a checkpoint, dispatching on its tags.

    Returns None rather than raising when the checkpoint is absent — a video
    should still get step predictions on a machine where only task 1 has been
    trained. The caller reports the skip.

    Three tags decide how the checkpoint is used, and all three are absent from
    SANO's original sano.pt, which is why each has a default that reproduces
    it: `arch` (sano-lstm), `space` (resnet50) and `thresholds` (None, meaning
    the caller's global threshold).
    """
    if not ckpt_path.exists() or not std_path.exists():
        return None

    from pitvis.models.lstm import SanoLSTM

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = ckpt["args"]
    arch = ckpt.get("arch", "sano-lstm")
    if arch != "sano-lstm":
        raise SystemExit(
            f"{ckpt_path} declares arch {arch!r}, which this inference path "
            f"does not know how to build. Add it here rather than guessing."
        )
    model = SanoLSTM(
        in_dim=feature_dim, hidden=a["hidden"], layers=a["layers"],
        window=a["window"], dropout=a["dropout"], aux_step=a["aux_step"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    stats = np.load(std_path)
    taus = ckpt.get("thresholds")
    meta = {"arch": arch, "space": ckpt.get("space", spaces.DEFAULT),
            "thresholds": np.asarray(taus, dtype=np.float32) if taus else None,
            "variant": ckpt.get("variant", "sano")}
    return model, stats["mean"], stats["std"], a, meta


@torch.no_grad()
def predict_instruments(features: np.ndarray, model, mean, std,
                        device: torch.device, threshold: float,
                        chunk: int, *, return_probs: bool = False,
                        thresholds: np.ndarray | None = None):
    """Run SANO. Returns (T, 2) instrument pairs in the raw challenge encoding.

    Delegates to the training module's `predict_video` so inference here and at
    training time cannot diverge — the same rule the workflow runners follow.
    With `return_probs`, also returns the (T, 19) sigmoid and the binary mask.
    """
    from pitvis.training.instruments import predict_video
    x = torch.from_numpy((features - mean) / std).float()
    if thresholds is None:
        return predict_video(model, x, threshold, chunk, device,
                             return_probs=return_probs)

    # Per-class thresholds need the margin-based cap, so this cannot delegate
    # to predict_video — that one is pinned to the SANO reproduction's global
    # rule. Same windows, same chunking; only the decision differs.
    from pitvis.evaluation.instruments import multihot_to_pairs
    from pitvis.models.lstm import causal_windows, decide_per_class
    tt = torch.from_numpy(thresholds).to(device)
    xs = x.unsqueeze(0)
    w = causal_windows(xs, model.window)
    keeps, probs = [], []
    with torch.no_grad():
        for s in range(0, xs.shape[1], chunk):
            e = min(s + chunk, xs.shape[1])
            wc = w[:, s:e].reshape((e - s), model.window, xs.shape[2]).to(device)
            h = model.drop(model.lstm(wc)[0][:, -1])
            logits = model.instruments(h)
            keeps.append(decide_per_class(logits, tt).cpu().numpy())
            if return_probs:
                probs.append(torch.sigmoid(logits).cpu().numpy())
    keep = np.concatenate(keeps)
    pairs = multihot_to_pairs(keep)
    if return_probs:
        return pairs, np.concatenate(probs), keep.astype(np.int8)
    return pairs


def load_instrument_labels(path: Path, expected: int) -> np.ndarray | None:
    """Instrument ground truth from an annotations CSV, or None if unavailable.

    Returns None for a .npy of step labels — that file simply does not carry
    instruments, which is not an error worth stopping a prediction over.
    """
    if path.suffix == ".npy":
        arr = np.load(path)
        return arr.astype(np.int64) if arr.ndim == 2 and arr.shape[1] == 2 else None

    df = pd.read_csv(path)
    cols = ["int_instrument1", "int_instrument2"]
    if not all(c in df.columns for c in cols):
        return None
    inst = df[cols].to_numpy()
    if len(inst) == expected + 1:       # the documented trailing row
        inst = inst[:expected]
    if len(inst) != expected:
        raise SystemExit(
            f"{path}: {len(inst)} instrument rows but {expected} predicted frames"
        )
    return inst.astype(np.int64)


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
