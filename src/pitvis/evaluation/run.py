"""Score a trained checkpoint with the official metric — no retraining.

`evaluation/` is a library (the vendored metric plus our reporting), not a
pipeline. Its workflow is the one thing you repeatedly want and that training
makes expensive: take a checkpoint that already exists and score it.

That separation matters because two of the interesting ablations are
*inference-time* choices, not training choices:

    --no-cci          drop the consistency constraint -> strictly causal
    --mask-excluded   remove classes 0/11/13 from the argmax

Neither changes a weight. Running them through `pitvis-train arst` retrains all
three stages (~112 s) to answer a question about the decoder, and — because MPS
kernels are not bit-deterministic — returns a model that differs slightly from
the one you meant to ablate. Scoring one fixed checkpoint keeps the weights
constant, so the difference you measure is the difference you asked about.

THE CHECKPOINT DECIDES WHAT IT IS SCORED AGAINST. Its `space` tag selects the
feature cache and its `standardize.npz` is taken from beside the weights. Both
used to be assumed — features were always loaded from `resnet50` and the stats
always from `data/arst/` — and neither assumption fails loudly. `resnet50` and
`resnet50_ft` are both 2048-d, as are `dinov2_vitb14` and `dinov2_ft` at 768-d,
so scoring a fine-tuned checkpoint against frozen features loaded cleanly and
reported a wrong number with no warning at all. `--space` still overrides, and
says so when it disagrees with the tag.

Usage:
    uv run pitvis-eval                          # score data/arst/citi.pt on val
    uv run pitvis-eval --no-cci                 # strictly causal
    uv run pitvis-eval --mask-excluded          # the scoring-rule exploit
    uv run pitvis-eval --ckpt data/arst/v2/best@dinov2_ft/model.pt
    uv run pitvis-eval --split train --confusion
"""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import torch

from pitvis.data import spaces
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
    ap.add_argument("--standardize", default=None,
                    help="train-split mean/std (default: standardize.npz beside "
                         "the checkpoint — they are never resolved separately)")
    ap.add_argument("--space", default=None, choices=spaces.names(),
                    help="override the feature space (default: the checkpoint's tag)")
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
    from pitvis.inference.checkpoints import STEPS, read_tags

    ckpt_path = Path(args.ckpt)
    # Stats come from beside the weights unless asked otherwise. Applying one
    # model's mean/std to another silently shifts every feature, so the two are
    # never resolved independently.
    std_path = (Path(args.standardize) if args.standardize
                else ckpt_path.parent / "standardize.npz")
    for p, what in [(ckpt_path, "checkpoint"), (std_path, "standardisation stats")]:
        if not p.exists():
            raise SystemExit(f"{what} not found at {p} — run `uv run pitvis-train arst` first")

    dev = device_of()
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    trained = ckpt["args"]
    meta = read_tags(ckpt, STEPS)
    width = args.width if args.width is not None else trained["width"]
    chunk = args.chunk if args.chunk is not None else trained["chunk"]
    space = args.space or meta["space"]
    # The flag only ever turns masking ON. A checkpoint trained with it is
    # scored with it; --mask-excluded still works on one that was not.
    mask = args.mask_excluded or meta["mask_excluded"]

    print(f"checkpoint: {ckpt_path}")
    print(f"  trained with W={trained['width']}, chunk={trained['chunk']}, "
          f"seed={trained['seed']}, variant={meta['variant']}")
    print(f"  space      {space}"
          + ("  (from the checkpoint)" if not args.space else
             f"  (--space OVERRIDE; the checkpoint says {meta['space']})"))
    print(f"  stats      {std_path}")
    print(f"  scoring    W={width}, chunk={chunk}, "
          f"CCI={'on' if args.cci else 'off'}, "
          f"mask-excluded={'on' if mask else 'off'}"
          + ("  (from the checkpoint)" if meta["mask_excluded"]
             and not args.mask_excluded else "")
          + (f", logit-adjust tau={meta['prior_tau']:g}"
             if meta["logit_adjust"] is not None else ""))
    print(f"  device     {dev}")

    if args.space and args.space != meta["space"]:
        print(f"\nWARNING: scoring a checkpoint tagged {meta['space']!r} against "
              f"{args.space!r} features.\n"
              f"         Same-width spaces load without error and report a wrong "
              f"number — this is only correct if you meant it.")

    stats = np.load(std_path)
    mean, std = stats["mean"], stats["std"]

    videos = VAL if args.split == "val" else TRAIN
    split = load_split(videos, space)
    feature_dim = split[0][1].shape[1]

    # A width mismatch would surface as a load_state_dict shape error further
    # down; naming both sides here turns it into a fixable sentence.
    trained_dim = ckpt["spatial"]["project.weight"].shape[1]
    if trained_dim != feature_dim:
        raise SystemExit(
            f"the checkpoint's spatial stage expects {trained_dim}-d features "
            f"but space {space!r} holds {feature_dim}-d.\n"
            f"Pick the space the checkpoint was trained on, or drop --space to "
            f"use its own tag ({meta['space']!r})."
        )

    spatial = SpatialEmbedding(feature_dim, num_classes=NUM_CLASSES).to(dev)
    tecno = TeCNO(num_classes=NUM_CLASSES).to(dev)
    arst = ARST(num_classes=NUM_CLASSES, width=width).to(dev)
    spatial.load_state_dict(ckpt["spatial"])
    tecno.load_state_dict(ckpt["tecno"])
    arst.load_state_dict(ckpt["arst"])
    spatial.eval(), tecno.eval(), arst.eval()

    decode_args = SimpleNamespace(chunk=chunk, cci=args.cci,
                                  mask_excluded=mask,
                                  logit_adjust=meta["logit_adjust"])

    preds = []
    with torch.no_grad():
        for vid, f, l in split:
            x = torch.from_numpy((f - mean) / std).to(dev)
            z, _ = spatial(x)
            _, ft = tecno(z.unsqueeze(0))
            p = cci_decode(arst, ft, decode_args, dev)
            preds.append((vid, l, p))
            print(f"video {vid:02d}: frame acc {(p == l).mean():.4f}")

    title = (f"{args.split} (checkpoint, space={space}, W={width}, "
             f"CCI={'on' if args.cci else 'off'}"
             f"{', masked' if mask else ''})")
    m = report(preds, title=title, show_confusion=args.confusion)

    if args.json:
        from pathlib import Path as _P
        _P(args.json).write_text(json.dumps(
            {"mean": m["mean"], "std": m["std"], "split": args.split,
             "ckpt": str(ckpt_path), "space": space, "width": width,
             "cci": args.cci, "mask_excluded": mask}, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
