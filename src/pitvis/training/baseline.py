"""Frame-wise linear probe on frozen ResNet-50 features.

The simplest baseline: standardize features on the train split, train a single
linear layer (2048 -> 15) with cross-entropy, evaluate on the val videos with
the official challenge metric — (macro F1 + normalised edit score) / 2, scored
per video and mean-averaged. No temporal context, so expect a very low edit
score: this is the floor that temporal models must beat.

Usage: uv run pitvis-train baseline [--epochs 10] [--lr 1e-3] [--confusion]
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from pitvis.data.dataset import NUM_CLASSES, TRAIN, VAL, load_split
from pitvis.evaluation.metric import report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--confusion", action="store_true",
                    help="print the 15-way confusion matrix")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )

    train = load_split(TRAIN)
    val = load_split(VAL)
    X = np.concatenate([f for _, f, _ in train])
    y = np.concatenate([l for _, _, l in train])
    print(f"train: {len(TRAIN)} videos, {len(X)} frames | "
          f"val: {len(VAL)} videos, {sum(len(l) for _, _, l in val)} frames")

    mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
    X = torch.from_numpy((X - mean) / std)
    y = torch.from_numpy(y)

    model = nn.Linear(X.shape[1], NUM_CLASSES).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(X))
        total, correct, loss_sum = 0, 0, 0.0
        for i in range(0, len(X), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb, yb = X[idx].to(device), y[idx].to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(idx)
        print(f"epoch {epoch + 1}/{args.epochs}: "
              f"loss {loss_sum / total:.4f} acc {correct / total:.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for vid, f, l in val:
            xb = torch.from_numpy((f - mean) / std).to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            preds.append((vid, l, pred))
            print(f"video {vid:02d}: frame acc {(pred == l).mean():.4f}")
    report(preds, title="val (frame-wise linear probe)",
           show_confusion=args.confusion)


if __name__ == "__main__":
    main()
