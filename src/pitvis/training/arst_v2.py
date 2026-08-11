"""Step-recognition variants — the same iteration, applied to task 1.

ARST reproduces at **0.3425** on the validation videos against Table 8's
benchmark of **70** for CITI on those same five. That is the same ~50%
shortfall instrument recognition had before `instruments_v2.py`, and the two
reproductions share one deviation from their published counterparts: a frozen
ImageNet backbone.

Task 1 also has the same shape of per-class failure. `notes/citi-baseline.md`
records step 3 (septum displacement) and step 9 (synthetic graft placement) at
**0.000 recall** and step 6 (durotomy) at 0.036, against an unweighted
cross-entropy trained over a distribution running 23.9% (tumour excision) to
0.06% (nasal packing).

The variants mirror task 2's, in the order the task-2 result says to run them:

  control   ARST unchanged — the anchor.
  masked    0/11/13 removed from the argmax. Not a model change at all: the
            official metric filters by GROUND TRUTH only, and calls f1_score
            with no `labels=`, so predicting an excluded class adds it to the
            macro average at F1 = 0. CLAUDE.md has recorded since before any of
            this that masking "can only raise the official metric"; this is the
            first time it is measured under cross-validation.
  weighted  Class-weighted cross-entropy at all three stages. Tests the same
            hypothesis that paid most on task 2.
  dinov2    Same cascade on DINOv2 features.
  best      Composed.

ORDER MATTERS, and task 2 is why. There, DINOv2 alone gained +0.021 macro —
inside the fold spread, indistinguishable from nothing — and only paid (+0.055)
once the loss stopped masking it. Running the backbone swap first and stopping
at the null result would have retired a hypothesis that was true.

`pitvis-train arst` is untouched and still reproduces CITI. Nothing here writes
data/arst/citi.pt.

Usage:
    uv run pitvis-train arst-v2 --variant masked --cv
    uv run pitvis-train arst-v2 --ablations --cv
    uv run pitvis-train arst-v2 --cv-report
    uv run pitvis-train arst-v2 --variant best        # single VAL scoring
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from pitvis.data import spaces
from pitvis.data.dataset import NUM_CLASSES, TRAIN, VAL, load_split
from pitvis.evaluation.metric import report
from pitvis.models.arst import ARST, BAND_WIDTH, SpatialEmbedding, TeCNO
from pitvis.paths import CKPT
from pitvis.training.arst import (EXCLUDED, cci_decode, device_of, embed,
                                  temporal, train_arst, train_tecno)
from pitvis.training.crossval import STEPS, cross_validate, load_entries, summarise

OUT_ROOT = CKPT / "v2"


@dataclass(frozen=True)
class Variant:
    name: str
    summary: str
    space: str = spaces.DEFAULT
    mask: bool = False
    weighted: bool = False


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant("control", "ARST unchanged — the anchor"),
        Variant("masked", "0/11/13 removed from the argmax", mask=True),
        Variant("weighted", "class-weighted CE at all three stages", weighted=True),
        Variant("dinov2", "same cascade on DINOv2 features", space="dinov2_vitb14"),
        # The winner. Masking + class weights on DINOv2 features: macro 0.5044,
        # edit 0.5789, metric 0.5417 out of fold, against control's
        # 0.4047 / 0.4282 / 0.4164.
        Variant("best", "WINNER — argmax masking + class weights on DINOv2",
                space="dinov2_vitb14", mask=True, weighted=True),
    ]
}


def class_weights(train, cap: float, dev: torch.device) -> torch.Tensor:
    """Inverse-frequency weights over the 15 step classes, capped.

    Computed on the fold's own training videos. Capped for the same reason
    task 2's pos_weight is: nasal packing is 0.06% of frames, so an uncapped
    inverse frequency lands in the thousands and the gradient stops being about
    anything else.
    """
    y = np.concatenate([l for _, _, l in train])
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    inv = np.where(counts > 0, len(y) / (NUM_CLASSES * np.maximum(counts, 1)), 1.0)
    return torch.from_numpy(np.clip(inv, 1.0, cap)).float().to(dev)


def train_spatial_w(train, mean, std, args, dev, weights):
    """Stage 1, with an optional class weight. Mirrors arst.train_spatial."""
    X = np.concatenate([f for _, f, _ in train])
    y = np.concatenate([l for _, _, l in train])
    X = torch.from_numpy((X - mean) / std)
    y = torch.from_numpy(y)

    model = SpatialEmbedding(X.shape[1], num_classes=NUM_CLASSES).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_spatial, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    rng = np.random.default_rng(args.seed)

    for _ in range(args.epochs_spatial):
        model.train()
        perm = torch.from_numpy(rng.permutation(len(X)))
        for i in range(0, len(X), args.batch_frames):
            idx = perm[i:i + args.batch_frames]
            _, logits = model(X[idx].to(dev))
            loss = loss_fn(logits, y[idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def train_tecno_w(zs, args, dev, weights):
    """Stage 2, with an optional class weight. Mirrors arst.train_tecno."""
    model = TeCNO(num_classes=NUM_CLASSES, dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_tecno)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    order = np.arange(len(zs))
    rng = np.random.default_rng(args.seed)
    for _ in range(args.epochs_tecno):
        model.train()
        rng.shuffle(order)
        for i in order:
            _, z, y = zs[i]
            stages, _ = model(z)
            loss = sum(loss_fn(s.squeeze(0), y.squeeze(0)) for s in stages)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def train_arst_w(fs, args, dev, weights):
    """Stage 3, with an optional class weight. Mirrors arst.train_arst."""
    model = ARST(num_classes=NUM_CLASSES, width=args.width,
                 dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_arst)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    windows = []
    for _, f, y in fs:
        for s in range(0, f.size(1), args.chunk):
            windows.append((f[:, s:s + args.chunk], y[:, s:s + args.chunk], s))

    rng = np.random.default_rng(args.seed)
    for _ in range(args.epochs_arst):
        model.train()
        rng.shuffle(windows)
        for f, y, off in windows:
            logits = model(f, model.shift(y), offset=off)
            loss = loss_fn(logits.squeeze(0), y.squeeze(0))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def make_fit(variant: Variant):
    """Return a fit function for the CV harness: train the three stages."""
    def fit(train, args, dev):
        X = np.concatenate([f for _, f, _ in train])
        mean, std = X.mean(0), X.std(0) + 1e-6
        del X
        w = class_weights(train, args.weight_cap, dev) if variant.weighted else None

        spatial = train_spatial_w(train, mean, std, args, dev, w)
        zs = embed(spatial, train, mean, std, dev)
        tecno = train_tecno_w(zs, args, dev, w)
        fs = temporal(tecno, zs)
        arst = train_arst_w(fs, args, dev, w)

        opts = argparse.Namespace(chunk=args.chunk, cci=args.cci,
                                  mask_excluded=variant.mask)

        def predict(features: np.ndarray) -> np.ndarray:
            x = torch.from_numpy((features - mean) / std).to(dev)
            with torch.no_grad():
                z, _ = spatial(x)
                _, ft = tecno(z.unsqueeze(0))
            return cci_decode(arst, ft, opts, dev)

        predict.parts = (spatial, tecno, arst)
        predict.stats = (mean, std)
        return predict
    return fit


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="control", choices=sorted(VARIANTS))
    ap.add_argument("--space", default=None, choices=spaces.names())
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--cv-report", action="store_true")
    ap.add_argument("--epochs-spatial", type=int, default=20)
    ap.add_argument("--epochs-tecno", type=int, default=30)
    ap.add_argument("--epochs-arst", type=int, default=20)
    ap.add_argument("--lr-spatial", type=float, default=1e-4)
    ap.add_argument("--lr-tecno", type=float, default=1e-4)
    ap.add_argument("--lr-arst", type=float, default=1e-5)
    ap.add_argument("--batch-frames", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--width", type=int, default=BAND_WIDTH)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-cap", type=float, default=10.0,
                    help="cap on the inverse-frequency class weight")
    ap.add_argument("--no-cci", dest="cci", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confusion", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cv_report:
        print(summarise(load_entries(STEPS), STEPS))
        return

    variant = VARIANTS[args.variant]
    args.space = args.space or variant.space
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device_of()
    print(f"variant: {variant.name} — {variant.summary}")
    print(f"device: {dev}  space: {args.space}  mask: {variant.mask}  "
          f"weighted: {variant.weighted}")

    # A variant carries its own space, so the same variant run on a different
    # one is a DIFFERENT result and needs a different name. CV already did this;
    # the artifact path did not, and `--variant best --space resnet50_ft` wrote
    # straight over the DINOv2 winner's model.pt — the checkpoint
    # `pitvis-predict --steps-model arst-v2:best` resolves to.
    label = (variant.name if args.space == variant.space
             else f"{variant.name}@{args.space}")

    if args.cv:
        cross_validate(make_fit(variant), args, dev, variant=label,
                       space=args.space, task=STEPS)
        return

    t0 = time.time()
    train = load_split(TRAIN, args.space)
    fit = make_fit(variant)(train, args, dev)
    print(f"trained in {time.time() - t0:.0f}s")

    out = OUT_ROOT / label
    out.mkdir(parents=True, exist_ok=True)
    mean, std = fit.stats
    spatial, tecno, arst = fit.parts
    np.savez(out / "standardize.npz", mean=mean, std=std)
    torch.save({"spatial": spatial.state_dict(), "tecno": tecno.state_dict(),
                "arst": arst.state_dict(), "args": vars(args),
                "space": args.space, "variant": variant.name,
                "mask_excluded": variant.mask}, out / "model.pt")

    preds = [(vid, l, fit(f)) for vid, f, l in load_split(VAL, args.space)]
    m = report(preds, title=f"val ({variant.name}, space={args.space})",
               show_confusion=args.confusion)
    (out / "result.json").write_text(json.dumps(
        {"mean": m["mean"], "std": m["std"], "variant": variant.name,
         "space": args.space, "args": vars(args)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
