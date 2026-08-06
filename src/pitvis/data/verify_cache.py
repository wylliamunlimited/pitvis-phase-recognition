"""Verify the integrity of the feature cache in data/features/.

Extraction asserts its invariants at write time; nothing re-checks them
afterwards, so a truncated npy, a stale labels file, or a manifest that
drifted from what is on disk would surface only as silent training
misbehaviour. This script re-establishes "the cache is good" as a fact.

Per video 01..25:
- features.npy exists, float32, shape (T, feature_dim), every value finite
- labels.npy (24 labeled videos): int64, length T, values in 0..14, and
  byte-identical to re-deriving from annotations_{n}.csv (drop the trailing
  background row, encode -1 -> 0). This re-verifies the off-by-one alignment
  from the raw annotations, independently of what extraction wrote.
- video 19 (no annotations) must NOT have a labels.npy
- the manifest entry exists and matches: frames == T, labels flag == presence

Manifest-level:
- space.id matches the recomputed content hash (mirrors extract_features.py)
- no manifest entries for videos that are not on disk

--probe additionally re-runs ffprobe per video and checks
T == ceil(nb_frames / round(fps)). This is the only length check that is
independent of the annotations, and the only independent one available for
video 19 — but it re-reads every container, so it is slow. Run it when the
raw videos or the cache have been touched; skip it day to day.

Exit code 0 iff every check passes.

Usage: uv run pitvis-verify [--probe]
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from pitvis.data.dataset import TRAIN, VAL
from pitvis.paths import FEATURES, MANIFEST, RAW

ALL_VIDEOS = list(range(1, 26))
LABELED = set(TRAIN) | set(VAL)  # 24 videos; 19 has no annotations
NUM_CLASSES = 15
FINITE_CHUNK = 50_000  # rows per np.isfinite pass over the memmap


def space_id(space: dict) -> str:
    """Content hash of the feature-space dict — mirrors extract_features.py."""
    payload = {k: v for k, v in space.items() if k != "id"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def probe(video: Path) -> tuple[int, int]:
    """Return (nb_frames, round(fps)).

    Deliberately independent of `extract_features.probe` (which also returns
    resolution): this is the check, and a check that imports the thing it
    checks verifies nothing.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=r_frame_rate,nb_read_packets",
        "-of", "json", str(video),
    ]
    s = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return int(s["nb_read_packets"]), round(int(num) / int(den))


def expected_labels(vid: int) -> np.ndarray:
    """Re-derive labels from the raw annotation CSV, independently of the cache."""
    steps = pd.read_csv(RAW / f"annotations_{vid:02d}.csv")["int_step"].to_numpy()
    assert steps[-1] == -1, f"video {vid:02d}: trailing annotation row is not background"
    labels = steps[:-1].copy()
    labels[labels == -1] = 0
    return labels.astype(np.int64)


def check_video(vid: int, manifest: dict, do_probe: bool) -> list[str]:
    errors = []
    key = f"video_{vid:02d}"
    d = FEATURES / key
    feat_path = d / "features.npy"
    label_path = d / "labels.npy"

    if not feat_path.exists():
        return [f"{key}: features.npy missing"]

    feats = np.load(feat_path, mmap_mode="r")
    feature_dim = manifest["space"]["feature_dim"]
    if feats.dtype != np.float32:
        errors.append(f"{key}: features dtype {feats.dtype}, expected float32")
    if feats.ndim != 2 or feats.shape[1] != feature_dim:
        errors.append(f"{key}: features shape {feats.shape}, expected (T, {feature_dim})")
    for start in range(0, len(feats), FINITE_CHUNK):
        if not np.isfinite(feats[start:start + FINITE_CHUNK]).all():
            errors.append(f"{key}: non-finite feature values near row {start}")
            break
    t = len(feats)

    if vid in LABELED:
        if not label_path.exists():
            errors.append(f"{key}: labels.npy missing")
        else:
            labels = np.load(label_path)
            if labels.dtype != np.int64:
                errors.append(f"{key}: labels dtype {labels.dtype}, expected int64")
            if labels.shape != (t,):
                errors.append(f"{key}: labels shape {labels.shape}, expected ({t},)")
            if labels.min() < 0 or labels.max() >= NUM_CLASSES:
                errors.append(f"{key}: label values outside 0..{NUM_CLASSES - 1}")
            derived = expected_labels(vid)
            if len(derived) != t:
                errors.append(
                    f"{key}: annotations give {len(derived)} labels, features give {t} "
                    f"— off-by-one alignment broken"
                )
            elif not np.array_equal(labels, derived):
                errors.append(f"{key}: labels.npy differs from re-derived annotations")
    elif label_path.exists():
        errors.append(f"{key}: has labels.npy but no annotation CSV exists")

    entry = manifest["videos"].get(key)
    if entry is None:
        errors.append(f"{key}: no manifest entry")
    else:
        if entry["frames"] != t:
            errors.append(f"{key}: manifest frames {entry['frames']} != on-disk {t}")
        if entry["labels"] != label_path.exists():
            errors.append(f"{key}: manifest labels flag {entry['labels']} != on-disk")

    if do_probe:
        nb_frames, r = probe(RAW / f"{key}.mp4")
        expected = math.ceil(nb_frames / r)
        if t != expected:
            errors.append(f"{key}: probe expects {expected} frames, cache has {t}")
        if entry is not None and entry["fps_rounded"] != r:
            errors.append(f"{key}: manifest fps {entry['fps_rounded']} != probed {r}")

    return errors


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", action="store_true",
                        help="also re-probe every video with ffprobe (slow)")
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        sys.exit("manifest.json missing — cache has no provenance; re-run extraction")
    manifest = json.loads(MANIFEST.read_text())

    errors = []
    if manifest["space"]["id"] != space_id(manifest["space"]):
        errors.append(
            f"manifest space id {manifest['space']['id']} != recomputed "
            f"{space_id(manifest['space'])}"
        )
    orphans = set(manifest["videos"]) - {f"video_{v:02d}" for v in ALL_VIDEOS}
    if orphans:
        errors.append(f"manifest entries for unknown videos: {sorted(orphans)}")

    total_frames = 0
    for vid in ALL_VIDEOS:
        errs = check_video(vid, manifest, args.probe)
        errors.extend(errs)
        key = f"video_{vid:02d}"
        if not errs:
            t = len(np.load(FEATURES / key / "features.npy", mmap_mode="r"))
            total_frames += t
            print(f"{key}: OK ({t} frames{', labeled' if vid in LABELED else ''})")
        else:
            for e in errs:
                print(f"{key}: FAIL — {e}", file=sys.stderr)

    if errors:
        print(f"\nFAILED: {len(errors)} problem(s)", file=sys.stderr)
        sys.exit(1)
    print(f"\nOK: 25 videos, {total_frames} frames, "
          f"space {manifest['space']['id']}"
          f"{', probed' if args.probe else ''}")


if __name__ == "__main__":
    main()
