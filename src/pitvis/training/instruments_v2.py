"""Instrument-recognition variants — the iteration on top of SANO.

SANO reproduces at 0.2321 official / 0.6234 name-aligned weighted / 0.2556
macro on the validation videos, against Table 8's benchmark of **81** on those
same five videos. The measured defect is precise: **9 of the 19 classes are
never predicted at all**, and the 4 that are learned carry ~91% of positives.
That is the signature of unweighted BCE plus a single 0.5 threshold under heavy
imbalance — neither of which the paper specifies, and SANO explicitly *did*
balance (a "not faithful" row in our own table).

Each variant changes exactly one thing, so a delta is attributable:

  control     SANO, unchanged, inside the CV harness — the anchor. Without it
              every delta is unanchored, since a CV mean is not comparable to
              the VAL mean SANO's number came from.
  weighted    BCE gets pos_weight. Tests: the dead classes are a gradient
              imbalance artifact, not missing information.
  thresholds  Per-class decision thresholds instead of a global 0.5, capped by
              margin. Tests: the ranking signal is already in the logits and
              the 0.5 cut is what discards it.
  dinov2      Identical model on DINOv2 ViT-B/14 features. Tests: the frozen
              ImageNet backbone is the shared bottleneck — the one deviation
              both our reproductions have from their published counterparts,
              which both sit near 50% of Table 8.
  best        Whatever cleared the bar, composed.

`pitvis-train instruments` is untouched and still reproduces SANO byte for
byte. Nothing here writes to data/instruments/sano.pt, so pitvis-predict and
the app keep working against the reproduction.

Usage:
    uv run pitvis-train instruments-v2 --variant weighted --cv
    uv run pitvis-train instruments-v2 --ablations --cv      # every variant
    uv run pitvis-train instruments-v2 --cv-report           # the leaderboard
    uv run pitvis-train instruments-v2 --variant best        # single VAL scoring
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from pitvis.data import spaces
from pitvis.data.dataset import TRAIN, VAL, load_split, load_split_instruments
from pitvis.evaluation.instruments import multihot, multihot_to_pairs, report
from pitvis.models.lstm import (HIDDEN, LAYERS, WINDOW, SanoLSTM, causal_windows,
                                decide, decide_per_class)
from pitvis.paths import CKPT_INSTRUMENTS
from pitvis.training.crossval import cross_validate, load_entries, summarise
from pitvis.training.instruments import device_of, gather_windows

NUM_INSTRUMENTS = 19
OUT_ROOT = CKPT_INSTRUMENTS / "v2"

# Thresholds are searched on this grid. Starts well below 0.5 because that is
# the whole point: a class with 184 positives in 84,666 frames will never clear
# a half-probability bar, however well the logits rank it.
TAU_GRID = np.round(np.arange(0.05, 0.91, 0.025), 3)


@dataclass(frozen=True)
class Variant:
    name: str
    summary: str
    space: str = spaces.DEFAULT
    pos_weight: bool = False
    per_class: bool = False
    extras: dict = field(default_factory=dict)


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant("control", "SANO unchanged — the anchor"),
        Variant("weighted", "BCE pos_weight from the fold's own training videos",
                pos_weight=True),
        Variant("thresholds", "per-class tau, cross-fitted; cap by margin",
                per_class=True),
        Variant("dinov2", "same model on DINOv2 ViT-B/14 features",
                space="dinov2_vitb14"),
        Variant("best", "composed from whatever cleared the bar",
                pos_weight=True, per_class=True),
    ]
}


# -- one training run --------------------------------------------------------

def standardise(train) -> tuple[np.ndarray, np.ndarray]:
    X = np.concatenate([f for _, f, _ in train])
    mean, std = X.mean(0), X.std(0) + 1e-6
    del X
    return mean, std


def pos_weight_from(train, cap: float) -> torch.Tensor:
    """neg/pos per class, capped.

    Uncapped, class 1 (184 positives in ~68,000 frames) gets a weight near 370
    and its gradient swamps the batch. The cap is what keeps the variant a
    rebalancing experiment rather than a divergence experiment.
    """
    Y = np.concatenate([multihot(i) for _, _, i in train])
    pos = Y.sum(0).astype(np.float64)
    neg = len(Y) - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1), 1.0)
    return torch.from_numpy(np.clip(w, 1.0, cap)).float()


def train_model(train, args, dev: torch.device, weights: torch.Tensor | None):
    """Fit one SanoLSTM on `train`. Returns (model, mean, std)."""
    mean, std = standardise(train)
    feats = [torch.from_numpy((f - mean) / std).float() for _, f, _ in train]
    targets = [torch.from_numpy(multihot(i)).float() for _, _, i in train]

    steps = {}
    if args.aux_step:
        steps = {vid: l for vid, _, l in load_split(
            [v for v, _, _ in train], args.space)}
    step_t = ([torch.from_numpy(steps[v]).long() for v, _, _ in train]
              if args.aux_step else None)

    index = np.array([(v, t) for v, f in enumerate(feats) for t in range(len(f))])
    model = SanoLSTM(in_dim=feats[0].shape[1], hidden=args.hidden,
                     layers=args.layers, window=args.window,
                     dropout=args.dropout, aux_step=args.aux_step).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss(pos_weight=weights.to(dev) if weights is not None else None)
    ce = nn.CrossEntropyLoss()

    rng = np.random.default_rng(args.seed)
    for _ in range(args.epochs):
        model.train()
        perm = rng.permutation(len(index))
        for i in range(0, len(perm), args.batch):
            batch = index[perm[i:i + args.batch]]
            xb = gather_windows(feats, batch, args.window, dev)
            yb = torch.stack([targets[v][t] for v, t in batch]).to(dev)
            out, _ = model.lstm(xb)
            h = model.drop(out[:, -1])
            loss = bce(model.instruments(h), yb)
            if args.aux_step:
                sb = torch.stack([step_t[v][t] for v, t in batch]).to(dev)
                loss = loss + args.aux_weight * ce(model.steps(h), sb)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), mean, std


@torch.no_grad()
def probabilities(model, feats: np.ndarray, mean, std, args, dev) -> np.ndarray:
    """(T, D) features -> (T, 19) sigmoid probabilities, chunked."""
    x = torch.from_numpy((feats - mean) / std).float().unsqueeze(0)
    w = causal_windows(x, model.window)
    out = []
    for s in range(0, x.shape[1], args.chunk):
        e = min(s + args.chunk, x.shape[1])
        wc = w[:, s:e].reshape((e - s), model.window, x.shape[2]).to(dev)
        h = model.drop(model.lstm(wc)[0][:, -1])
        out.append(torch.sigmoid(model.instruments(h)).cpu().numpy())
    return np.concatenate(out)


# -- per-class thresholds ----------------------------------------------------

def sweep_thresholds(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-class tau maximising that class's F1 on out-of-bag probabilities."""
    taus = np.full(NUM_INSTRUMENTS, 0.5, dtype=np.float32)
    for c in range(NUM_INSTRUMENTS):
        y, p = Y[:, c], P[:, c]
        if y.sum() == 0:
            continue
        best_f1, best_tau = -1.0, 0.5
        for tau in TAU_GRID:
            pred = p >= tau
            tp = float((pred & (y > 0)).sum())
            if tp == 0:
                continue
            prec, rec = tp / pred.sum(), tp / y.sum()
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best_f1:
                best_f1, best_tau = f1, float(tau)
        taus[c] = best_tau
    return taus


def crossfit_thresholds(train, args, dev, weights) -> np.ndarray:
    """Fit thresholds on OUT-OF-BAG probabilities, by 2-fold cross-fitting.

    Fitting tau on data the model trained on would bias every threshold
    upward — the probabilities are optimistic there — which is the exact
    opposite of what a rare class needs. Fitting it on a carved-out holdout
    instead would shrink the training set and confound the comparison with
    `control`, which trains on all of it.

    So: split the fold's training videos in half, train on each half and
    predict the other, and sweep on the union. Every probability used is
    out-of-bag, and the final model still trains on the full fold. Costs two
    extra fits per fold.
    """
    half = max(1, len(train) // 2)
    parts = [train[:half], train[half:]]
    P, Y = [], []
    for i in (0, 1):
        inner = parts[1 - i]
        if not inner or not parts[i]:
            continue
        m, mu, sd = train_model(inner, args, dev, weights)
        for _, f, inst in parts[i]:
            P.append(probabilities(m, f, mu, sd, args, dev))
            Y.append(multihot(inst))
    if not P:
        return np.full(NUM_INSTRUMENTS, 0.5, dtype=np.float32)
    return sweep_thresholds(np.concatenate(P), np.concatenate(Y))


# -- the fit function the CV harness calls -----------------------------------

def make_fit(variant: Variant):
    def fit(train, args, dev):
        weights = (pos_weight_from(train, args.weight_cap)
                   if variant.pos_weight else None)
        taus = (crossfit_thresholds(train, args, dev, weights)
                if variant.per_class else None)
        model, mean, std = train_model(train, args, dev, weights)
        tt = torch.from_numpy(taus).to(dev) if taus is not None else None

        def predict(feats: np.ndarray) -> np.ndarray:
            P = probabilities(model, feats, mean, std, args, dev)
            logits = torch.from_numpy(np.log(P / (1 - P + 1e-12) + 1e-12))
            keep = (decide_per_class(logits, tt.cpu()) if tt is not None
                    else decide(logits, args.threshold))
            return multihot_to_pairs(keep.numpy())

        predict.thresholds = taus            # carried out for the checkpoint
        predict.model = model
        predict.stats = (mean, std)
        return predict
    return fit


# -- entry point -------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="control", choices=sorted(VARIANTS),
                    help="which variant to run (default: control)")
    ap.add_argument("--space", default=None, choices=spaces.names(),
                    help="override the variant's feature space")
    ap.add_argument("--cv", action="store_true",
                    help="cross-validate over the 19 training videos")
    ap.add_argument("--cv-report", action="store_true",
                    help="print the leaderboard from data/instruments/cv/")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="global threshold, used when the variant is not per-class")
    ap.add_argument("--weight-cap", type=float, default=50.0,
                    help="cap on pos_weight; uncapped, class 1 lands near 370")
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--no-aux-step", dest="aux_step", action="store_false")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class-report", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.cv_report:
        print(summarise(load_entries()))
        return

    variant = VARIANTS[args.variant]
    args.space = args.space or variant.space
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device_of()
    print(f"variant: {variant.name} — {variant.summary}")
    print(f"device: {dev}  space: {args.space}  pos_weight: {variant.pos_weight}  "
          f"per-class tau: {variant.per_class}")

    if args.cv:
        # Label by space when it is overridden, so `--variant best --space X`
        # cannot silently overwrite the entry for `best` on its own space.
        label = (variant.name if args.space == variant.space
                 else f"{variant.name}@{args.space}")
        cross_validate(make_fit(variant), args, dev,
                       variant=label, space=args.space)
        return

    # Single split: train on all of TRAIN, score VAL once. This is the run that
    # produces a shippable checkpoint, and for the winner it happens exactly
    # once, after the leaderboard is frozen.
    t0 = time.time()
    train = load_split_instruments(TRAIN, args.space)
    fit = make_fit(variant)(train, args, dev)
    print(f"trained in {time.time() - t0:.0f}s")

    out = OUT_ROOT / variant.name
    out.mkdir(parents=True, exist_ok=True)
    mean, std = fit.stats
    np.savez(out / "standardize.npz", mean=mean, std=std)
    torch.save({"model": fit.model.state_dict(), "args": vars(args),
                "arch": "sano-lstm", "space": args.space, "variant": variant.name,
                "thresholds": None if fit.thresholds is None else fit.thresholds.tolist()},
               out / "model.pt")

    preds = [(vid, inst, fit(f))
             for vid, f, inst in load_split_instruments(VAL, args.space)]
    m = report(preds, title=f"val ({variant.name}, space={args.space})",
               show_per_class=args.per_class_report)
    (out / "result.json").write_text(json.dumps(
        {"mean": m["mean"], "std": m["std"], "variant": variant.name,
         "space": args.space, "args": vars(args)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
