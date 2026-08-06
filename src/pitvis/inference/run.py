"""Point a trained model at a video: `uv run pitvis-predict --video case.mp4`.

    uv run pitvis-predict --video 26531686/video_19.mp4
    uv run pitvis-predict --video case.mp4 --out results/case
    uv run pitvis-predict --video 26531686/video_01.mp4 \
                          --labels 26531686/annotations_01.csv
    uv run pitvis-predict --video case.mp4 --no-cache --no-cci

Writes `predictions.csv` (`int_time,int_step`, the challenge's own submission
format) and `segments.csv` (`start_s,end_s,int_step,duration_s`). With
`--labels`, also scores against ground truth with the official metric.

Uses the feature cache when the manifest records this exact video in the
current feature space, so re-running on one of the 25 challenge videos takes
seconds instead of re-decoding. `--no-cache` forces a full decode — use it to
verify the end-to-end path really works from pixels.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pitvis.evaluation.metric import decode, report
from pitvis.inference import predict as P
from pitvis.paths import CKPT

STEP_NAMES = {
    -1: "background", 1: "nasal corridor creation", 2: "anterior sphenoidotomy",
    3: "septum displacement", 4: "sphenoid sinus clearance", 5: "sellotomy",
    6: "durotomy", 7: "tumour excision", 8: "haemostasis",
    9: "synthetic graft placement", 10: "fat graft placement",
    11: "gasket seal construct", 12: "dural sealant", 13: "nasal packing",
    14: "debris clearance",
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", required=True, type=Path, help="path to a video file")
    ap.add_argument("--out", type=Path,
                    help="output directory (default: predictions/<video stem>/)")
    ap.add_argument("--labels", type=Path,
                    help="ground truth (.npy or annotations CSV) — enables scoring")
    ap.add_argument("--ckpt", type=Path, default=CKPT / "citi.pt")
    ap.add_argument("--standardize", type=Path, default=CKPT / "standardize.npz")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="always decode from the video, never reuse the feature cache")
    ap.add_argument("--no-cci", dest="cci", action="store_false",
                    help="disable the consistency constraint (strictly causal)")
    ap.add_argument("--mask-excluded", action="store_true",
                    help="remove classes 0/11/13 from the argmax")
    ap.add_argument("--width", type=int, help="override the checkpoint's band width")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"))
    ap.add_argument("--confusion", action="store_true",
                    help="with --labels, also print the 15-way confusion matrix")
    args = ap.parse_args(argv)

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    import torch
    from pitvis.training.arst import device_of
    dev = torch.device(args.device) if args.device else device_of()

    out_dir = args.out or Path("predictions") / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"video   {args.video}")
    print(f"device  {dev}")

    t0 = time.time()
    features = P.cached_features(args.video) if args.cache else None
    if features is not None:
        print(f"features  cache hit — {features.shape} ({time.time() - t0:.1f}s)")
    else:
        print("features  decoding at 1 fps (no cache entry for this video)"
              if args.cache else "features  decoding at 1 fps (--no-cache)")
        features = P.embed(args.video, dev)
        print(f"          {features.shape} in {time.time() - t0:.0f}s")

    spatial, tecno, arst, mean, std, trained, width = P.load_checkpoint(
        args.ckpt, args.standardize, features.shape[1], dev, args.width
    )
    print(f"model   {args.ckpt}  (trained W={trained['width']}, "
          f"seed={trained['seed']})")
    print(f"        W={width}, CCI={'on' if args.cci else 'off'}, "
          f"mask-excluded={'on' if args.mask_excluded else 'off'}")

    t1 = time.time()
    preds = P.predict(features, spatial, tecno, arst, mean, std, dev,
                      args.chunk, args.cci, args.mask_excluded)
    print(f"predict {len(preds)} seconds in {time.time() - t1:.0f}s")

    # per-second, in the challenge's own encoding (background is -1)
    raw = decode(preds)
    pd.DataFrame({"int_time": np.arange(len(raw)), "int_step": raw}).to_csv(
        out_dir / "predictions.csv", index=False)

    segments = P.to_segments(preds)
    segments.to_csv(out_dir / "segments.csv", index=False)

    print(f"\n{len(segments)} segments over {len(preds)} s "
          f"({len(preds) / 60:.1f} min)")
    print(f"  {'start':>7} {'end':>7} {'dur':>6}  step")
    for r in segments.itertuples():
        if r.duration_s < 5:            # keep the console readable
            continue
        name = STEP_NAMES.get(r.int_step, "?")
        print(f"  {r.start_s:>7} {r.end_s:>7} {r.duration_s:>6}  "
              f"{r.int_step:>3}  {name}")
    short = (segments.duration_s < 5).sum()
    if short:
        print(f"  ... plus {short} segment(s) under 5 s, omitted here "
              f"but present in segments.csv")

    summary = {
        "video": str(args.video),
        "frames": int(len(preds)),
        "segments": int(len(segments)),
        "width": width,
        "cci": args.cci,
        "mask_excluded": args.mask_excluded,
        "checkpoint": str(args.ckpt),
    }

    if args.labels:
        labels = P.load_labels(args.labels, len(preds))
        m = report([(args.video.stem, labels, preds)],
                   title=f"{args.video.name} vs {args.labels.name}",
                   show_confusion=args.confusion)
        summary["metric"] = m["mean"]
        summary["frame_accuracy"] = float((preds == labels).mean())

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {out_dir}/predictions.csv, segments.csv, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
