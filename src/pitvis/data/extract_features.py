"""Extract 1 fps frozen backbone features for every PitVis video.

For each video: decode every round(fps)-th frame (frames 0, r, 2r, ...) via an
ffmpeg rawvideo pipe, run them through a frozen timm backbone
(num_classes=0 -> pooled features), and save

    data/features/{space}/video_{n}/features.npy   (T, D) float32
    data/features/{space}/video_{n}/labels.npy     (T,)   int64  (labeled only)

where T = ceil(nb_frames / round(fps)). Labels are the annotation rows
truncated to T (the dropped trailing row is verified background) with the
15-way encoding: -1 -> 0, k -> k.

ONE DIRECTORY PER FEATURE SPACE. `D` and the preprocessing depend on the
backbone, so two spaces are not interchangeable — 2048 for resnet50, 768 for
dinov2_vitb14. Each owns a directory and a manifest, which is what lets a
second backbone be extracted without destroying the first. The spaces
themselves are named in `pitvis.data.spaces`.

Each manifest records its space (backbone, transform, target fps, content-hash
id) plus per-video provenance, and extraction still refuses to write into a
manifest describing a different space — but the remedy is now `--space`
rather than deleting the cache.

Resumable: videos whose features.npy already exists with the expected length
are skipped. Video 19 gets features but no labels (annotations missing).

Usage: uv run pitvis-extract [video_numbers...] [--space NAME]
       uv run pitvis-extract --migrate      # pre-space cache -> data/features/resnet50/
"""

import argparse
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from pitvis.data import spaces
from pitvis.paths import (DATA, FEATURES, RAW, features_dir, manifest_path,
                          video_dir)

BATCH = 64


def probe(video: Path) -> tuple[int, int, int, int]:
    """Return (nb_frames, round(fps), width, height).

    Resolution is probed rather than assumed: the 25 challenge videos are all
    1280x720, but `embed_video` accepts arbitrary files and the raw ffmpeg pipe
    is read in fixed-size frames — a wrong size desynchronises every frame
    after the first.
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


def space_id(space: dict) -> str:
    """Content hash of a feature-space dict.

    Keys starting with `_` are provenance carried alongside rather than
    identity: `_trained_on` is already implied by the checkpoint digest, and
    hashing it would make the id depend on how the list happened to be sorted.
    """
    payload = {k: v for k, v in space.items()
               if k != "id" and not k.startswith("_")}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def build_model(device: torch.device, space: spaces.Space):
    """Instantiate a space's backbone and return (model, transform, space_dict).

    THE RETURNED DICT IS THE HASHED PAYLOAD and its keys are frozen at
    {backbone, feature_dim, target_fps, transform}. `space.name` and
    `space.model_kwargs` are deliberately absent: adding either would move the
    existing cache's id off 67912d3efc6852e7 and the guard below would reject
    940 MB of correct features. `model_kwargs` needs no separate entry anyway —
    everything it changes (DINOv2's img_size) surfaces in `transform`.
    """
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(space.backbone, pretrained=True, num_classes=0,
                              **space.model_kwargs)
    if space.checkpoint:
        ck = torch.load(DATA / space.checkpoint, map_location="cpu",
                        weights_only=False)
        missing, unexpected = model.load_state_dict(ck["backbone"], strict=False)
        if unexpected:
            raise SystemExit(f"{space.checkpoint}: unexpected keys {unexpected[:4]}")
        print(f"loaded fine-tuned weights from {space.checkpoint}"
              + (f" ({len(missing)} head keys absent, as expected)" if missing else ""))
    model.eval().to(device)

    # `resolve_data_config` reports the CHECKPOINT's native config, not the
    # model we just built. DINOv2's weights ship at 518, so without this
    # override the transform resizes to 518 and the 224 model rejects it:
    # "Input height (518) doesn't match model (224)". Overriding input_size is
    # the documented way to reconcile the two, and it is what makes the hashed
    # transform describe the tensor the backbone actually sees.
    overrides: dict = {}
    if "img_size" in space.model_kwargs:
        s = space.model_kwargs["img_size"]
        overrides["input_size"] = (3, s, s)
    if space.source == "frames":
        # The frame cache is already a centre square, and fine-tuning resized
        # it straight to 224. crop_pct=1.0 reproduces that exactly; the default
        # 0.95 would crop again and re-introduce the mismatch this avoids.
        overrides["crop_pct"] = 1.0
    cfg = resolve_data_config(overrides, model=model)
    transform = create_transform(**cfg)
    payload = {
        "backbone": space.backbone,
        "feature_dim": model.num_features,
        "target_fps": space.target_fps,
        "transform": {
            k: cfg[k]
            for k in ("crop_mode", "crop_pct", "input_size", "interpolation", "mean", "std")
        },
    }
    # Extra keys ONLY when they are non-default, so resnet50's payload -- and
    # therefore its id 67912d3efc6852e7 and its 940 MB of cached features --
    # is bit-for-bit what it always was.
    if space.checkpoint:
        # Digest the WEIGHTS, not the file. Hashing the container made the id
        # move when a provenance key was added to it, invalidating 940 MB of
        # features whose contents were bit-identical. Identity is what the
        # encoder computes, not how it was serialised.
        h = hashlib.sha256()
        for key in sorted(ck["backbone"]):
            h.update(key.encode())
            h.update(ck["backbone"][key].cpu().numpy().tobytes())
        payload["checkpoint"] = h.hexdigest()[:16]
        # Carried out of band, not hashed — it is already implied by the
        # digest. crossval reads it to refuse folds the encoder has seen.
        payload["_trained_on"] = ck.get("trained_on")
    if space.source != "video":
        payload["source"] = space.source
        payload["frame_size"] = space.frame_size
    payload["id"] = space_id(payload)
    return model, transform, payload


def load_manifest(payload: dict, path: Path) -> dict:
    """Load one space's manifest, or start it. Refuses a different space.

    Still full-dict equality, unchanged — but the remedy is no longer "delete
    the cache". Each space owns a directory, so a second backbone is a second
    manifest rather than a collision.
    """
    if not path.exists():
        return {"space": payload, "videos": {}}
    manifest = json.loads(path.read_text())
    canonical = json.loads(json.dumps(payload))  # tuples -> lists, as stored

    # Compare only the keys that define identity. Provenance (`_`-prefixed) is
    # carried alongside and refreshed in place: it is already implied by the
    # checkpoint digest, so letting it force a mismatch would reject a cache
    # for a reason the id itself does not see -- and produce the memorable
    # error "space X != current X".
    ident = lambda d: {k: v for k, v in d.items() if not k.startswith("_")}
    if ident(manifest["space"]) != ident(canonical):
        raise SystemExit(
            f"manifest feature space {manifest['space']['id']} != current "
            f"{payload['id']} — {path.parent} holds features from a different "
            f"backbone/transform. Extract to a different --space, or delete "
            f"that directory and re-extract."
        )
    manifest["space"] = canonical      # refresh provenance without re-extracting
    return manifest


def record_video(manifest: dict, path: Path, vid: int, frames: int, fps_rounded: int,
                 has_labels: bool, source: Path, has_instruments: bool = False) -> None:
    manifest["videos"][f"video_{vid:02d}"] = {
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fps_rounded": fps_rounded,
        "frames": frames,
        "labels": has_labels,
        "instruments": has_instruments,
        "source": str(source),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def legacy_layout() -> list[Path]:
    """Video directories sitting directly under data/features/ — pre-space.

    The cache used to be `data/features/video_NN/`. Finding one now means the
    migration has not run, and the right response is to say so rather than
    silently re-decode 40 GB into the new layout.
    """
    if not FEATURES.exists():
        return []
    return sorted(p for p in FEATURES.iterdir()
                  if p.is_dir() and p.name.startswith("video_"))


def require_migration() -> None:
    stray = legacy_layout()
    if stray:
        raise SystemExit(
            f"found {len(stray)} video directories directly under {FEATURES} "
            f"(pre-space layout). Run `uv run pitvis-extract --migrate` to move "
            f"them into data/features/{spaces.DEFAULT}/ — it is a rename, "
            f"nothing is re-decoded."
        )


def migrate() -> None:
    """Move a pre-space cache into data/features/<DEFAULT>/.

    A same-filesystem rename of ~940 MB: instant, and it re-decodes nothing.
    Opt-in rather than automatic — moving a user's cache is not something to
    do as a side effect of an unrelated command.
    """
    stray = legacy_layout()
    old_manifest = FEATURES / "manifest.json"
    if not stray and not old_manifest.exists():
        print(f"nothing to migrate — no pre-space layout under {FEATURES}")
        return

    dest = features_dir(spaces.DEFAULT)
    dest.mkdir(parents=True, exist_ok=True)
    for d in stray:
        d.rename(dest / d.name)
    if old_manifest.exists():
        old_manifest.rename(manifest_path(spaces.DEFAULT))
    print(f"migrated {len(stray)} video directories + manifest -> {dest}")


def write_annotations(vid: int, out_dir: Path, expected: int) -> bool:
    """Write labels.npy and instruments.npy from the raw CSV. Returns has-labels.

    Split out of `extract_video` so it can also run on the resumable path: the
    feature cache predates instrument support, and extraction returns early when
    features already exist, so a caller with a warm cache needs a way to backfill
    labels without re-decoding 40 GB of video.

    Steps use the 15-way encoding (-1 -> 0). Instruments keep their RAW
    sentinels: -1 (out of patient), -2 (no secondary) and 0 (nothing visible)
    are three different things, and collapsing them corrupts the target. See
    notes/data-dictionary.md §4.
    """
    ann_path = RAW / f"annotations_{vid:02d}.csv"
    if not ann_path.exists():
        return False

    df = pd.read_csv(ann_path)
    steps = df["int_step"].to_numpy()
    assert len(steps) == expected + 1, \
        f"video {vid:02d}: {len(steps)} ann rows, expected {expected + 1}"
    assert steps[-1] == -1, f"video {vid:02d}: dropped row is not background"

    labels = steps[:expected].copy()
    labels[labels == -1] = 0
    np.save(out_dir / "labels.npy", labels.astype(np.int64))

    inst = df[["int_instrument1", "int_instrument2"]].to_numpy()[:expected]
    np.save(out_dir / "instruments.npy", inst.astype(np.int64))
    return True


@torch.no_grad()
def embed_video(video: Path, model, transform, device: torch.device,
                tag: str | None = None) -> tuple[np.ndarray, int]:
    """Decode `video` at 1 fps and embed every frame. Returns (features, fps).

    The one path from pixels to a feature matrix — used both by cache
    extraction and by `pitvis.inference.predict`, so a prediction is computed
    from exactly the feature space the model was trained on. Accepts any video
    file; nothing here assumes the challenge's naming or resolution.
    """
    tag = tag or video.stem
    nb_frames, r, width, height = probe(video)
    expected = math.ceil(nb_frames / r)
    frame_bytes = width * height * 3

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(video),
        "-vf", f"select=not(mod(n\\,{r}))", "-vsync", "0",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    t0 = time.time()
    feats, batch = [], []

    def flush():
        if batch:
            x = torch.stack(batch).to(device)
            feats.append(model(x).float().cpu().numpy())
            batch.clear()

    n = 0
    while True:
        buf = proc.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 3)
        batch.append(transform(Image.fromarray(frame)))
        n += 1
        if len(batch) == BATCH:
            flush()
        if n % 500 == 0:
            print(f"  {tag}: {n}/{expected} ({n / (time.time() - t0):.1f} fps)")
    flush()
    proc.wait()

    features = np.concatenate(feats) if feats else np.empty((0, model.num_features), np.float32)
    if len(features) != expected:
        raise RuntimeError(
            f"{tag}: decoded {len(features)} frames, expected {expected} "
            f"(ceil({nb_frames}/{r})) — the ffmpeg pipe desynchronised"
        )
    return features.astype(np.float32), r


@torch.no_grad()
def embed_frames(vid: int, size: int, model, transform, device: torch.device,
                 tag: str | None = None) -> np.ndarray:
    """Embed a video from the JPEG frame cache rather than the mp4.

    The path a fine-tuned encoder must take. It was tuned on these exact
    files -- same centre-square crop, same 384px source -- so re-deriving
    pixels from the video would introduce a preprocessing difference and the
    measurement would confound the model with the framing. It also skips the
    ~18-minute ffmpeg decode entirely.
    """
    from pitvis.data.extract_frames import video_frames
    d = video_frames(size, vid)
    paths = sorted(d.glob("*.jpg"))
    if not paths:
        raise SystemExit(
            f"no frames at {d} — run `uv run pitvis-frames --size {size}` first"
        )
    tag = tag or f"video {vid:02d}"
    t0, feats, batch = time.time(), [], []

    def flush():
        if batch:
            feats.append(model(torch.stack(batch).to(device)).float().cpu().numpy())
            batch.clear()

    for n, path in enumerate(paths, 1):
        batch.append(transform(Image.open(path).convert("RGB")))
        if len(batch) == BATCH:
            flush()
        if n % 1000 == 0:
            print(f"  {tag}: {n}/{len(paths)} ({n / (time.time() - t0):.0f} fps)")
    flush()
    return np.concatenate(feats).astype(np.float32)


@torch.no_grad()
def extract_video(vid: int, model, transform, device: torch.device, manifest: dict,
                  space: spaces.Space | None = None) -> None:
    space = space or spaces.get(spaces.DEFAULT)
    name = space.name
    video = RAW / f"video_{vid:02d}.mp4"
    out_dir = video_dir(name, vid)
    mpath = manifest_path(name)
    nb_frames, r, _, _ = probe(video)
    expected = math.ceil(nb_frames / r)

    feat_path = out_dir / "features.npy"
    if feat_path.exists():
        existing = np.load(feat_path, mmap_mode="r")
        if len(existing) == expected:
            # Features are the expensive part and they are already here. Still
            # backfill any annotation artifact that is missing — instruments.npy
            # postdates this cache, and re-decoding 40 GB to add a column read
            # from a CSV would defeat the resumability rule in CLAUDE.md.
            missing = not (out_dir / "instruments.npy").exists()
            has_labels = (out_dir / "labels.npy").exists()
            if missing and (RAW / f"annotations_{vid:02d}.csv").exists():
                has_labels = write_annotations(vid, out_dir, expected)
                print(f"video {vid:02d}: exists ({expected} frames), "
                      f"backfilled instruments.npy")
            entry = manifest["videos"].get(f"video_{vid:02d}")
            if entry is None or entry["frames"] != expected or missing \
                    or "instruments" not in entry:
                record_video(manifest, mpath, vid, expected, r, has_labels, video,
                             (out_dir / "instruments.npy").exists())
                if not missing:
                    print(f"video {vid:02d}: exists ({expected} frames), manifest updated")
            elif not missing:
                print(f"video {vid:02d}: exists ({expected} frames), skipping")
            return
        print(f"video {vid:02d}: found {len(existing)} != {expected} frames, redoing")

    t0 = time.time()
    if space.source == "frames":
        features = embed_frames(vid, space.frame_size, model, transform, device,
                                tag=f"video {vid:02d}")
        if len(features) != expected:
            raise RuntimeError(
                f"video {vid:02d}: frame cache holds {len(features)}, expected "
                f"{expected} — the frame cache and the video disagree"
            )
    else:
        features, r = embed_video(video, model, transform, device,
                                  tag=f"video {vid:02d}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(feat_path, features)

    has_labels = write_annotations(vid, out_dir, expected)

    record_video(manifest, mpath, vid, expected, r, has_labels, video,
                 (out_dir / "instruments.npy").exists())
    print(f"video {vid:02d}: done, {expected} frames in {time.time() - t0:.0f}s")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", type=int, metavar="N",
                    help="video numbers to extract (default: all 25)")
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"),
                    help="override device autodetection")
    ap.add_argument("--space", default=spaces.DEFAULT, choices=spaces.names(),
                    help=f"feature space to extract into (default: {spaces.DEFAULT})")
    ap.add_argument("--migrate", action="store_true",
                    help="move a pre-space data/features/video_NN cache into "
                         "data/features/<default>/ and exit (a rename, no decode)")
    args = ap.parse_args(argv)

    if args.migrate:
        migrate()
        return
    require_migration()

    vids = args.videos or list(range(1, 26))
    bad = [v for v in vids if not 1 <= v <= 25]
    if bad:
        raise SystemExit(f"video numbers must be in 1..25, got {bad}")

    device = torch.device(args.device) if args.device else torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    space = spaces.get(args.space)
    print(f"device: {device}, space: {space.name} ({space.backbone}), videos: {vids}")
    model, transform, payload = build_model(device, space)
    manifest = load_manifest(payload, manifest_path(space.name))
    print(f"feature space id: {payload['id']}  dim: {payload['feature_dim']}")
    for vid in vids:
        extract_video(vid, model, transform, device, manifest, space)


if __name__ == "__main__":
    main()
