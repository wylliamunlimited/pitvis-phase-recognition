"""Decode 1 fps frames to disk — the data path fine-tuning needs.

`extract_features.py` embeds each frame once and throws the pixels away. That
is the right trade for training temporal models on frozen features, and it is
exactly what makes fine-tuning the backbone impossible: there is nothing to
push through it a second time.

Fine-tuning needs every frame on every epoch. The alternative to storing them
is re-decoding, and decoding one full pass costs ~18 minutes (measured), so 50
epochs would burn **15 hours on decode alone** — before any gradient. Writing
JPEGs once buys all of that back.

    data/frames/{size}/video_{n}/{index:05d}.jpg

THE CROP IS NOT INCIDENTAL. A PitVis frame is a centred endoscopic circle
inside black pillarbox bars: 1280x720 where roughly 720x720 carries signal.
Centre-cropping to a square discards ~44% of every frame that is pure black,
which is why the cache lands at 1.67 GB rather than 3 GB, and it is the same
crop the feature pipeline already applies (see notes/embeddings.md).

SIZE IS STORED LARGER THAN THE TRAINING CROP, on purpose. ARST resizes to 250
and random-crops 224; SANO uses 384. Storing exactly 224 would leave no room
for random crops or rotation, and augmentation is one of the three things our
faithfulness tables mark as not reproduced. Re-decoding to recover resolution
is the expensive mistake to avoid, so the default is 384.

Measured on real frames: 14.6 KB each at 384 px (1.67 GB for all 120,018),
9.2 KB at 256 (1.05 GB). Raw uint8 at 384 would be 49 GB, which is why JPEG.

Resumable: a video whose directory already holds the expected frame count is
skipped.

Usage: uv run pitvis-frames [video_numbers...] [--size 384]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from pitvis.paths import DATA, RAW

FRAMES = DATA / "frames"
DEFAULT_SIZE = 384
QUALITY = 90


def frames_dir(size: int) -> Path:
    return FRAMES / str(size)


def video_frames(size: int, vid: int) -> Path:
    return frames_dir(size) / f"video_{vid:02d}"


def manifest_path(size: int) -> Path:
    return frames_dir(size) / "manifest.json"


def probe(video: Path) -> tuple[int, int, int, int]:
    """(nb_frames, round(fps), width, height) — independent of extract_features.

    Deliberately a second implementation rather than an import: this module
    writes a different artifact and should not inherit a bug from the other.
    """
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=r_frame_rate,nb_read_packets,width,height",
        "-of", "json", str(video),
    ]
    s = json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return (int(s["nb_read_packets"]), round(int(num) / int(den)),
            int(s["width"]), int(s["height"]))


def square(frame: np.ndarray, size: int) -> Image.Image:
    """Centre-crop the endoscopic circle out of the pillarbox, then resize."""
    h, w = frame.shape[:2]
    side = min(h, w)
    x0, y0 = (w - side) // 2, (h - side) // 2
    im = Image.fromarray(frame[y0:y0 + side, x0:x0 + side])
    return im if side == size else im.resize((size, size), Image.BICUBIC)


def extract_video(vid: int, size: int, manifest: dict) -> None:
    video = RAW / f"video_{vid:02d}.mp4"
    out = video_frames(size, vid)
    nb_frames, r, width, height = probe(video)
    expected = math.ceil(nb_frames / r)

    if out.exists() and len(list(out.glob("*.jpg"))) == expected:
        print(f"video {vid:02d}: exists ({expected} frames), skipping")
        record(manifest, size, vid, expected, r, video)
        return

    out.mkdir(parents=True, exist_ok=True)
    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"select=not(mod(n\\,{r}))", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    t0, n = time.time(), 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        arr = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        square(arr, size).save(out / f"{n:05d}.jpg", "JPEG", quality=QUALITY)
        n += 1
        if n % 1000 == 0:
            print(f"  video {vid:02d}: {n}/{expected} ({n / (time.time() - t0):.0f} fps)")
    proc.wait()

    if n != expected:
        raise RuntimeError(
            f"video {vid:02d}: wrote {n} frames, expected {expected} "
            f"(ceil({nb_frames}/{r})) — the ffmpeg pipe desynchronised"
        )
    record(manifest, size, vid, expected, r, video)
    print(f"video {vid:02d}: done, {expected} frames in {time.time() - t0:.0f}s")


def record(manifest: dict, size: int, vid: int, frames: int, fps: int,
           source: Path) -> None:
    manifest["videos"][f"video_{vid:02d}"] = {
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frames": frames, "fps_rounded": fps, "source": str(source),
    }
    p = manifest_path(size)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(p)


def load_manifest(size: int) -> dict:
    p = manifest_path(size)
    if p.exists():
        return json.loads(p.read_text())
    return {"size": size, "quality": QUALITY, "crop": "centre square",
            "target_fps": 1, "videos": {}}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", type=int, metavar="N",
                    help="video numbers (default: all 25)")
    ap.add_argument("--size", type=int, default=DEFAULT_SIZE,
                    help=f"stored square edge in px (default: {DEFAULT_SIZE}); "
                         f"keep it larger than the training crop so random "
                         f"cropping has headroom")
    args = ap.parse_args(argv)

    vids = args.videos or list(range(1, 26))
    bad = [v for v in vids if not 1 <= v <= 25]
    if bad:
        raise SystemExit(f"video numbers must be in 1..25, got {bad}")

    manifest = load_manifest(args.size)
    if manifest.get("size") != args.size:
        raise SystemExit(
            f"{frames_dir(args.size)} holds {manifest.get('size')}px frames, "
            f"not {args.size}px"
        )
    print(f"frames -> {frames_dir(args.size)}  ({args.size}px, q{QUALITY}, "
          f"centre square)")
    for vid in vids:
        extract_video(vid, args.size, manifest)


if __name__ == "__main__":
    main()
