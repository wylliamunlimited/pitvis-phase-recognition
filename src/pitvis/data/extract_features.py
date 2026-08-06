"""Extract 1 fps frozen ResNet-50 features for every PitVis video.

For each video: decode every round(fps)-th frame (frames 0, r, 2r, ...) via an
ffmpeg rawvideo pipe, run them through a frozen ImageNet-pretrained timm
resnet50 (num_classes=0 -> 2048-d pooled features), and save

    data/features/video_{n}/features.npy   (T, 2048) float32
    data/features/video_{n}/labels.npy     (T,)      int64   (labeled videos only)

where T = ceil(nb_frames / round(fps)). Labels are the annotation rows
truncated to T (the dropped trailing row is verified background) with the
15-way encoding: -1 -> 0, k -> k.

Resumable: videos whose features.npy already exists with the expected length
are skipped. Video 19 gets features but no labels (annotations missing).

data/features/manifest.json records the feature space (backbone, transform,
target fps, content-hash id) plus per-video provenance. Extraction refuses to
write into a cache whose manifest describes a different feature space — that
would silently mix incompatible features in one training run.

Usage: uv run pitvis-extract [video_numbers...]
"""

import hashlib
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pitvis.paths import FEATURES as OUT
from pitvis.paths import MANIFEST, RAW

BACKBONE = "resnet50"
TARGET_FPS = 1
WIDTH, HEIGHT = 1280, 720  # uniform across all 25 videos (verified)
FRAME_BYTES = WIDTH * HEIGHT * 3
BATCH = 64


def probe(video: Path) -> tuple[int, int]:
    """Return (nb_frames, round(fps))."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=r_frame_rate,nb_read_packets",
        "-of", "json", str(video),
    ]
    s = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["nb_read_packets"]), round(int(num) / int(den))


def space_id(space: dict) -> str:
    """Content hash of a feature-space dict (without its 'id' key)."""
    payload = {k: v for k, v in space.items() if k != "id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_model(device: torch.device):
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(BACKBONE, pretrained=True, num_classes=0)
    model.eval().to(device)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg)
    space = {
        "backbone": BACKBONE,
        "feature_dim": model.num_features,
        "target_fps": TARGET_FPS,
        "transform": {
            k: cfg[k]
            for k in ("crop_mode", "crop_pct", "input_size", "interpolation", "mean", "std")
        },
    }
    space["id"] = space_id(space)
    return model, transform, space


def load_manifest(space: dict) -> dict:
    """Load the manifest, or start one. Refuses a different feature space."""
    if not MANIFEST.exists():
        return {"space": space, "videos": {}}
    manifest = json.loads(MANIFEST.read_text())
    canonical = json.loads(json.dumps(space))  # tuples -> lists, as stored
    if manifest["space"] != canonical:
        raise SystemExit(
            f"manifest feature space {manifest['space']['id']} != current "
            f"{space['id']} — the cache holds features from a different "
            f"backbone/transform. Delete data/features/ and re-extract."
        )
    return manifest


def record_video(manifest: dict, vid: int, frames: int, fps_rounded: int,
                 has_labels: bool, source: Path) -> None:
    manifest["videos"][f"video_{vid:02d}"] = {
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fps_rounded": fps_rounded,
        "frames": frames,
        "labels": has_labels,
        "source": str(source),
    }
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(MANIFEST)


@torch.no_grad()
def extract_video(vid: int, model, transform, device: torch.device, manifest: dict) -> None:
    video = RAW / f"video_{vid:02d}.mp4"
    out_dir = OUT / f"video_{vid:02d}"
    nb_frames, r = probe(video)
    expected = math.ceil(nb_frames / r)

    feat_path = out_dir / "features.npy"
    if feat_path.exists():
        existing = np.load(feat_path, mmap_mode="r")
        if len(existing) == expected:
            entry = manifest["videos"].get(f"video_{vid:02d}")
            if entry is None or entry["frames"] != expected:
                record_video(manifest, vid, expected, r,
                             (out_dir / "labels.npy").exists(), video)
                print(f"video {vid:02d}: exists ({expected} frames), manifest entry added")
            else:
                print(f"video {vid:02d}: exists ({expected} frames), skipping")
            return
        print(f"video {vid:02d}: found {len(existing)} != {expected} frames, redoing")

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"select=not(mod(n\\,{r}))", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=FRAME_BYTES * 4)

    t0 = time.time()
    feats, batch = [], []

    def flush():
        if batch:
            x = torch.stack(batch).to(device)
            feats.append(model(x).float().cpu().numpy())
            batch.clear()

    n = 0
    while True:
        buf = proc.stdout.read(FRAME_BYTES)
        if len(buf) < FRAME_BYTES:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        batch.append(transform(Image.fromarray(frame)))
        n += 1
        if len(batch) == BATCH:
            flush()
        if n % 500 == 0:
            print(f"  video {vid:02d}: {n}/{expected} ({n / (time.time() - t0):.1f} fps)")
    flush()
    proc.wait()

    features = np.concatenate(feats) if feats else np.empty((0, 2048), np.float32)
    assert len(features) == expected, \
        f"video {vid:02d}: extracted {len(features)}, expected {expected}"

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, features.astype(np.float32))

    ann_path = RAW / f"annotations_{vid:02d}.csv"
    if ann_path.exists():
        steps = pd.read_csv(ann_path)["int_step"].to_numpy()
        assert len(steps) == expected + 1, \
            f"video {vid:02d}: {len(steps)} ann rows, expected {expected + 1}"
        assert steps[-1] == -1, f"video {vid:02d}: dropped row is not background"
        labels = steps[:expected].copy()
        labels[labels == -1] = 0
        np.save(out_dir / "labels.npy", labels.astype(np.int64))

    record_video(manifest, vid, expected, r, ann_path.exists(), video)
    print(f"video {vid:02d}: done, {expected} frames in {time.time() - t0:.0f}s")


def main() -> None:
    vids = [int(a) for a in sys.argv[1:]] or list(range(1, 26))
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}, videos: {vids}")
    model, transform, space = build_model(device)
    manifest = load_manifest(space)
    for vid in vids:
        extract_video(vid, model, transform, device, manifest)


if __name__ == "__main__":
    main()
