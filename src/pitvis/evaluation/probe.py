"""Per-class average precision on a feature space — is the information there?

THE QUESTION THIS ANSWERS. When a class is never predicted, two very different
things could be true: the features encode it and the loss or the threshold is
throwing that away, or the encoder simply cannot see it. Those have opposite
fixes — rebalancing versus a better representation — and the headline metric
cannot tell them apart, because it only ever reports decisions.

Average precision can. It is computed from the ranking, so it is free of both
the decision threshold and the class prior: AP near the base rate means no
signal, AP far above it means signal that something downstream is discarding.

WHAT IT ALREADY SETTLED. Run against frozen DINOv2 features, one-vs-rest:

    tissue glue          282 train positives    AP 0.767
    micro doppler        679                    AP 0.731
    cup forceps        1,635                    AP 0.055
    retractable knife    492                    AP 0.015

Tissue glue is rarer than four of the weak classes and is nearly separable, so
**rarity does not predict difficulty** — what predicts it is whether the encoder
can see the instrument. That result is why the backbone was fine-tuned: no
threshold, class weight or sampler recovers information that is not present.

The probe is a linear model on frozen features, deliberately. A stronger head
would blur the question by learning its way around a weak representation, and
the question is precisely how good the representation is.

Usage:
    uv run pitvis-probe --space resnet50_ft
    uv run pitvis-probe --space resnet50 --space resnet50_ft   # side by side
"""

from __future__ import annotations

import argparse

import numpy as np

from pitvis.data import spaces
from pitvis.data.dataset import TRAIN, VAL, load_split_instruments
from pitvis.evaluation.instruments import (INSTRUMENT_NAMES, NUM_INSTRUMENTS,
                                           multihot)

MAX_NEG = 40_000        # negatives subsampled per class when fitting; AP is
                        # always scored on the full validation set


def probe_space(space: str, classes: list[int], seed: int = 0,
                quiet: bool = False) -> dict[int, dict]:
    """One-vs-rest AP per class, fitted on TRAIN and scored on VAL."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score

    tr = load_split_instruments(TRAIN, space)
    va = load_split_instruments(VAL, space)
    Xtr = np.concatenate([f for _, f, _ in tr])
    Ytr = np.concatenate([multihot(i) for _, _, i in tr])
    Xva = np.concatenate([f for _, f, _ in va])
    Yva = np.concatenate([multihot(i) for _, _, i in va])

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd

    rng = np.random.default_rng(seed)
    out: dict[int, dict] = {}
    for c in classes:
        y = Ytr[:, c].astype(bool)
        pos = np.flatnonzero(y)
        neg_all = np.flatnonzero(~y)
        neg = rng.choice(neg_all, size=min(MAX_NEG, len(neg_all)), replace=False)
        idx = np.concatenate([pos, neg])

        clf = LogisticRegression(max_iter=300, class_weight="balanced")
        clf.fit(Xtr[idx], Ytr[idx, c])
        scores = clf.decision_function(Xva)

        base = float(Yva[:, c].mean())
        ap = float(average_precision_score(Yva[:, c], scores))
        out[c] = {"train_pos": int(y.sum()), "val_pos": int(Yva[:, c].sum()),
                  "base_rate": base, "ap": ap,
                  "lift": ap / base if base else float("nan")}
        if not quiet:
            print(f"  {c:>3} {INSTRUMENT_NAMES[c][:24]:<26} AP {ap:.3f}")
    return out


def main(argv: list[str] | None = None) -> None:
    ap_ = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap_.add_argument("--space", action="append", default=None,
                     choices=spaces.names(),
                     help="feature space to probe; repeat to compare")
    ap_.add_argument("--classes", type=int, nargs="*", default=None,
                     help="instrument ids (default: all 19)")
    ap_.add_argument("--seed", type=int, default=0)
    args = ap_.parse_args(argv)

    targets = args.space or [spaces.DEFAULT]
    classes = args.classes if args.classes is not None else list(range(NUM_INSTRUMENTS))

    results = {}
    for sp in targets:
        print(f"probing {sp} ...")
        results[sp] = probe_space(sp, classes, args.seed, quiet=True)

    head = f"{'id':>3} {'name':<26}{'train':>7}{'base':>9}"
    for sp in targets:
        head += f"{sp[:14]:>16}"
    print("\n" + head)
    print("-" * len(head))
    for c in classes:
        first = results[targets[0]][c]
        row = (f"{c:>3} {INSTRUMENT_NAMES[c][:25]:<26}"
               f"{first['train_pos']:>7}{first['base_rate']:>9.4f}")
        for sp in targets:
            row += f"{results[sp][c]['ap']:>16.3f}"
        print(row)

    print("\nmean AP" + " " * 36 + "".join(
        f"{np.mean([results[sp][c]['ap'] for c in classes]):>16.3f}" for sp in targets))
    if len(targets) == 2:
        a, b = targets
        deltas = [results[b][c]["ap"] - results[a][c]["ap"] for c in classes]
        improved = sum(d > 0 for d in deltas)
        print(f"\n{b} vs {a}: mean AP {np.mean(deltas):+.3f}, "
              f"{improved}/{len(classes)} classes improved")
        worst = sorted(zip(classes, deltas), key=lambda t: -t[1])[:5]
        print("biggest gains: " + ", ".join(
            f"{INSTRUMENT_NAMES[c][:18]} {d:+.3f}" for c, d in worst))


if __name__ == "__main__":
    main()
