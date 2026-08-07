"""Train SANO's PitVis task-2 model (instrument recognition) on cached features.

SANO placed joint 1st in task 2 (Das et al. Table 6, 41.6±06.3). Architecture
and the reasons for choosing it over SDS-HD are in `pitvis.models.lstm`.

    frozen ResNet-50 cache -> causal 5-frame window -> 2-layer LSTM
      -> 19 sigmoid outputs (instruments, BCE)
      -> 15 softmax outputs (steps, CE, auxiliary — training only)

Windows are sampled as a flat shuffled index over (video, t) rather than looped
per video: every position is independent given its own window, so this is honest
minibatch SGD rather than a convenience, and it avoids materialising
84,666 x 5 x 2048 floats (~3.5 GB) at once.

Usage:
    uv run pitvis-train instruments
    uv run pitvis-train instruments --no-aux-step      # ablate the step head
    uv run pitvis-train instruments --threshold 0.4    # decision threshold
    uv run pitvis-train instruments --per-class        # per-instrument F1 table
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn

from pitvis.data.dataset import TRAIN, VAL, load_split, load_split_instruments
from pitvis.evaluation.instruments import multihot, multihot_to_pairs, report
from pitvis.models.lstm import HIDDEN, LAYERS, WINDOW, SanoLSTM, causal_windows, decide
from pitvis.paths import CKPT_INSTRUMENTS


def device_of() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def gather_windows(feats: list[torch.Tensor], idx: np.ndarray, window: int,
                   dev: torch.device) -> torch.Tensor:
    """Build (B, window, D) for a batch of (video, t) index pairs.

    Left-clamped rather than zero-padded, matching `causal_windows`: position t
    sees [t-window+1 .. t], with frame 0 repeated where history runs out.
    """
    out = []
    for v, t in idx:
        f = feats[v]
        lo = max(0, t - window + 1)
        w = f[lo:t + 1]
        if len(w) < window:
            w = torch.cat([f[:1].expand(window - len(w), f.shape[1]), w])
        out.append(w)
    return torch.stack(out).to(dev)


@torch.no_grad()
def predict_video(model, feats: torch.Tensor, threshold: float,
                  chunk: int, dev: torch.device, *, return_probs: bool = False):
    """(T, D) -> (T, 2) instrument pairs. Chunked, and exact: each position is
    decoded from its own window, so chunk boundaries change nothing.

    `return_probs` also returns `(probs (T, 19) float32, keep (T, 19) int8)`.
    `keep` is worth carrying because the pairs alone are ambiguous: an all-zero
    row becomes `(-1, -2)`, which is byte-identical to the annotations' real
    out-of-patient sentinel. From `keep` the two are distinguishable, and only
    `keep` reveals when `decide` capped three positives down to two.

    `decide` recomputes the sigmoid internally. That duplication is deliberate:
    `decide` is a pure decision rule living in `models/`, and widening its
    return to avoid 19 redundant floats per row would push a display concern
    into the model layer.
    """
    x = feats.unsqueeze(0)
    w = causal_windows(x, model.window)              # a view, not a copy
    T = x.shape[1]
    keep, probs = [], []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        wc = w[:, s:e].reshape((e - s), model.window, x.shape[2]).to(dev)
        out, _ = model.lstm(wc)
        logits = model.instruments(out[:, -1])
        if return_probs:
            probs.append(torch.sigmoid(logits).cpu().numpy())
        keep.append(decide(logits, threshold).cpu().numpy())
    k = np.concatenate(keep)
    pairs = multihot_to_pairs(k)
    if return_probs:
        return pairs, np.concatenate(probs).astype(np.float32), k
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=10)      # Table 3: 10 epochs
    ap.add_argument("--lr", type=float, default=2e-4)      # Table 3: Adam 2e-4
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--hidden", type=int, default=HIDDEN)
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="sigmoid decision threshold (unspecified by the paper)")
    ap.add_argument("--aux-weight", type=float, default=0.5,
                    help="weight on the auxiliary step loss")
    ap.add_argument("--no-aux-step", dest="aux_step", action="store_false",
                    help="drop the auxiliary step head SANO trained alongside")
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-class", action="store_true",
                    help="print the per-instrument F1 table")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device_of()
    print(f"device: {dev}  window={args.window}  aux-step="
          f"{'on' if args.aux_step else 'off'}  threshold={args.threshold}")

    train = load_split_instruments(TRAIN)
    val = load_split_instruments(VAL)
    steps = {vid: l for vid, _, l in load_split(TRAIN)} if args.aux_step else {}

    X = np.concatenate([f for _, f, _ in train])
    mean, std = X.mean(0), X.std(0) + 1e-6
    del X
    CKPT_INSTRUMENTS.mkdir(parents=True, exist_ok=True)
    np.savez(CKPT_INSTRUMENTS / "standardize.npz", mean=mean, std=std)

    feats = [torch.from_numpy((f - mean) / std).float() for _, f, _ in train]
    targets = [torch.from_numpy(multihot(i)).float() for _, _, i in train]
    step_t = [torch.from_numpy(steps[v]).long() for v, _, _ in train] if args.aux_step else None

    index = np.array([(v, t) for v, f in enumerate(feats) for t in range(len(f))])
    print(f"train: {len(train)} videos, {len(index):,} windows")

    model = SanoLSTM(in_dim=feats[0].shape[1], hidden=args.hidden,
                     layers=args.layers, window=args.window,
                     dropout=args.dropout, aux_step=args.aux_step).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce, ce = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(index))
        run = tot = 0.0
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
            run += loss.item() * len(batch)
            tot += len(batch)
        print(f"  epoch {ep + 1}/{args.epochs}: loss {run / tot:.4f} "
              f"({time.time() - t0:.0f}s)")
    print(f"training done in {time.time() - t0:.0f}s")

    model.eval()
    torch.save({"model": model.state_dict(), "args": vars(args)},
               CKPT_INSTRUMENTS / "sano.pt")

    preds = []
    for vid, f, inst in val:
        x = torch.from_numpy((f - mean) / std).float()
        p = predict_video(model, x, args.threshold, args.chunk, dev)
        preds.append((vid, inst, p))
        exact = (multihot(inst) == multihot(p)).all(axis=1).mean()
        print(f"video {vid:02d}: exact-set match {exact:.4f}")

    title = (f"val (SANO, window={args.window}, "
             f"aux-step={'on' if args.aux_step else 'off'}, "
             f"threshold={args.threshold})")
    m = report(preds, title=title, show_per_class=args.per_class)
    (CKPT_INSTRUMENTS / "result.json").write_text(json.dumps(
        {"mean": m["mean"], "std": m["std"], "args": vars(args)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
