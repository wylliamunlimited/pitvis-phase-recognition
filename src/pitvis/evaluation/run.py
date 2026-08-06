"""Score a trained checkpoint with the official metric — no retraining.

`evaluation/` is a library (the vendored metric plus our reporting), not a
pipeline. Its workflow is the one thing you repeatedly want and that training
makes expensive: take a checkpoint that already exists and score it.

That separation matters because two of the interesting ablations are
*inference-time* choices, not training choices:

    --no-cci          drop the consistency constraint -> strictly causal
    --mask-excluded   remove classes 0/11/13 from the argmax

Neither changes a weight. Running them through `pitvis-train-arst` retrains all
three stages (~112 s) to answer a question about the decoder, and — because MPS
kernels are not bit-deterministic — returns a model that differs slightly from
the one you meant to ablate. Scoring one fixed checkpoint keeps the weights
constant, so the difference you measure is the difference you asked about.

Usage:
    uv run pitvis-eval                          # score data/arst/citi.pt on val
    uv run pitvis-eval --no-cci                 # strictly causal
    uv run pitvis-eval --mask-excluded          # the scoring-rule exploit
    uv run pitvis-eval --split train --confusion
"""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

from pitvis.data.dataset import NUM_CLASSES, TRAIN, VAL, load_split
from pitvis.evaluation.metric import report
from pitvis.models.arst import ARST, SpatialEmbedding, TeCNO
from pitvis.paths import CKPT


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--ckpt", default=str(CKPT / "citi.pt"),
                    help="checkpoint to score (default: data/arst/citi.pt)")
    ap.add_argument("--standardize", default=str(CKPT / "standardize.npz"),
                    help="train-split mean/std (default: data/arst/standardize.npz)")
    ap.add_argument("--split", choices=("val", "train"), default="val")
    ap.add_argument("--width", type=int,
                    help="override the checkpoint's banded-mask width")
    ap.add_argument("--chunk", type=int, help="override the checkpoint's chunk length")
    ap.add_argument("--no-cci", dest="cci", action="store_false",
                    help="disable the consistency constraint (strictly causal)")
    ap.add_argument("--mask-excluded", action="store_true",
                    help="remove classes 0/11/13 from the argmax")
    ap.add_argument("--confusion", action="store_true")
    ap.add_argument("--json", metavar="PATH", help="also write the result as JSON")
    args = ap.parse_args(argv)

    # imported here: training imports evaluation.metric, so a module-level
    # import would make the dependency look circular to a reader.
    from pitvis.training.arst import cci_decode, device_of

    from pathlib import Path
    ckpt_path, std_path = Path(args.ckpt), Path(args.standardize)
    for p, what in [(ckpt_path, "checkpoint"), (std_path, "standardisation stats")]:
        if not p.exists():
            raise SystemExit(f"{what} not found at {p} — run `uv run pitvis-train-arst` first")

    dev = device_of()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    trained = ckpt["args"]
    width = args.width if args.width is not None else trained["width"]
    chunk = args.chunk if args.chunk is not None else trained["chunk"]

    print(f"checkpoint: {ckpt_path}")
    print(f"  trained with W={trained['width']}, chunk={trained['chunk']}, "
          f"seed={trained['seed']}")
    print(f"  scoring    W={width}, chunk={chunk}, "
          f"CCI={'on' if args.cci else 'off'}, "
          f"mask-excluded={'on' if args.mask_excluded else 'off'}")
    print(f"  device     {dev}")

    stats = np.load(std_path)
    mean, std = stats["mean"], stats["std"]

    videos = VAL if args.split == "val" else TRAIN
    split = load_split(videos)
    feature_dim = split[0][1].shape[1]

    spatial = SpatialEmbedding(feature_dim, num_classes=NUM_CLASSES).to(dev)
    tecno = TeCNO(num_classes=NUM_CLASSES).to(dev)
    arst = ARST(num_classes=NUM_CLASSES, width=width).to(dev)
    spatial.load_state_dict(ckpt["spatial"])
    tecno.load_state_dict(ckpt["tecno"])
    arst.load_state_dict(ckpt["arst"])
    spatial.eval(), tecno.eval(), arst.eval()

    decode_args = SimpleNamespace(chunk=chunk, cci=args.cci,
                                  mask_excluded=args.mask_excluded)

    preds = []
    with torch.no_grad():
        for vid, f, l in split:
            x = torch.from_numpy((f - mean) / std).to(dev)
            z, _ = spatial(x)
            _, ft = tecno(z.unsqueeze(0))
            p = cci_decode(arst, ft, decode_args, dev)
            preds.append((vid, l, p))
            print(f"video {vid:02d}: frame acc {(p == l).mean():.4f}")

    title = (f"{args.split} (checkpoint, W={width}, "
             f"CCI={'on' if args.cci else 'off'}"
             f"{', masked' if args.mask_excluded else ''})")
    m = report(preds, title=title, show_confusion=args.confusion)

    if args.json:
        from pathlib import Path as _P
        _P(args.json).write_text(json.dumps(
            {"mean": m["mean"], "std": m["std"], "split": args.split,
             "ckpt": str(ckpt_path), "width": width, "cci": args.cci,
             "mask_excluded": args.mask_excluded}, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
