"""Train CITI's PitVis-2023 task-1 model (ARST) on the cached feature space.

Three stages, each frozen before the next trains (ARST §3.3):

    1. spatial    2048-d cached ResNet-50 -> Linear -> 512-d Z_t
                  trained frame-wise, shuffled, cross-entropy
    2. TeCNO      Z_1:T -> two-stage causal TCN -> 512-d F_t
                  trained per video, CE on both stages' logits
    3. ARST       F_1:T + shifted labels -> 15-way logits
                  trained per video with teacher forcing

Inference is a sequential auto-regressive rollout with Consistency Constraint
Inference (ARST §2.3, Algorithm 1), then the official challenge metric.

Usage:
    uv run pitvis-train-arst                      # full three-stage run
    uv run pitvis-train-arst --no-cci             # ablate CCI
    uv run pitvis-train-arst --width 0            # ablate the banded mask
    uv run pitvis-train-arst --mask-excluded      # drop 0/11/13 from argmax
"""

import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pitvis.data.dataset import NUM_CLASSES, TRAIN, VAL, load_split
from pitvis.evaluation.metric import report
from pitvis.models.arst import ARST, BAND_WIDTH, CCI_N, SpatialEmbedding, TeCNO
from pitvis.paths import CKPT

EXCLUDED = [0, 11, 13]      # encoded; scored classes are the other 12


def device_of() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# stage 1 — spatial embedding
# --------------------------------------------------------------------------

def train_spatial(train, mean, std, args, dev):
    """Frame-wise training of the 2048->512 projection (ARST §2.1).

    ARST fine-tunes ResNet-50 here; we only train the projection because the
    cache holds features, not pixels. See notes/citi-baseline.md.
    """
    X = np.concatenate([f for _, f, _ in train])
    y = np.concatenate([l for _, _, l in train])
    X = torch.from_numpy((X - mean) / std)
    y = torch.from_numpy(y)

    model = SpatialEmbedding(X.shape[1], num_classes=NUM_CLASSES).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_spatial, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(args.epochs_spatial):
        model.train()
        perm = torch.randperm(len(X))
        tot = corr = 0
        run = 0.0
        for i in range(0, len(X), args.batch_frames):
            idx = perm[i:i + args.batch_frames]
            xb, yb = X[idx].to(dev), y[idx].to(dev)
            _, logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
            corr += (logits.argmax(1) == yb).sum().item()
            tot += len(idx)
        print(f"  [spatial] epoch {ep + 1}/{args.epochs_spatial}: "
              f"loss {run / tot:.4f} acc {corr / tot:.4f}")
    return model.eval()


@torch.no_grad()
def embed(spatial, split, mean, std, dev):
    """Frozen 512-d Z_t per video."""
    out = []
    for vid, f, l in split:
        x = torch.from_numpy((f - mean) / std).to(dev)
        z, _ = spatial(x)
        out.append((vid, z.unsqueeze(0), torch.from_numpy(l).long().to(dev).unsqueeze(0)))
    return out


# --------------------------------------------------------------------------
# stage 2 — TeCNO
# --------------------------------------------------------------------------

def train_tecno(zs, args, dev):
    """Per-video training on full sequences. TeCNO is convolutional, so a whole
    video fits — no chunking needed and the ~1000-frame causal receptive field
    is preserved intact."""
    model = TeCNO(num_classes=NUM_CLASSES, dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_tecno)
    loss_fn = nn.CrossEntropyLoss()

    order = np.arange(len(zs))
    for ep in range(args.epochs_tecno):
        model.train()
        np.random.shuffle(order)
        run = corr = tot = 0.0
        for i in order:
            _, z, y = zs[i]
            stages, _ = model(z)
            # TeCNO supervises every stage
            loss = sum(loss_fn(s.squeeze(0), y.squeeze(0)) for s in stages)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            corr += (stages[-1].squeeze(0).argmax(1) == y.squeeze(0)).sum().item()
            tot += y.numel()
        print(f"  [tecno]   epoch {ep + 1}/{args.epochs_tecno}: "
              f"loss {run / len(zs):.4f} acc {corr / tot:.4f}")
    return model.eval()


@torch.no_grad()
def temporal(tecno, zs):
    """Frozen 512-d F_t per video (TeCNO stage-2 hidden)."""
    return [(vid, tecno(z)[1], y) for vid, z, y in zs]


# --------------------------------------------------------------------------
# stage 3 — ARST
# --------------------------------------------------------------------------

def train_arst(fs, args, dev):
    """Teacher-forced training (ARST §2.2).

    ARST uses one whole video per iteration. We chunk to `--chunk` frames
    because full-video self-attention is O(T^2) and T reaches 8,645 here —
    an 8645x8645x8-head attention matrix is ~2.4 GB. With a banded mask of
    width W only the first W queries of each chunk lose any context, so the
    approximation touches ~1% of positions at the default chunk size.
    """
    model = ARST(num_classes=NUM_CLASSES, width=args.width, dropout=args.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_arst)
    loss_fn = nn.CrossEntropyLoss()

    windows = []
    for _, f, y in fs:
        T = f.size(1)
        for s in range(0, T, args.chunk):
            windows.append((f[:, s:s + args.chunk], y[:, s:s + args.chunk], s))

    for ep in range(args.epochs_arst):
        model.train()
        np.random.shuffle(windows)
        run = corr = tot = 0.0
        for f, y, off in windows:
            logits = model(f, model.shift(y), offset=off)
            loss = loss_fn(logits.squeeze(0), y.squeeze(0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item() * y.numel()
            corr += (logits.squeeze(0).argmax(1) == y.squeeze(0)).sum().item()
            tot += y.numel()
        print(f"  [arst]    epoch {ep + 1}/{args.epochs_arst}: "
              f"loss {run / tot:.4f} acc(teacher-forced) {corr / tot:.4f}")
    return model.eval()


@torch.no_grad()
def encode_memory(model, f, chunk, dev):
    """Encoder memory for a whole video, chunked with `width` overlap so the
    result is identical to encoding the full sequence at once."""
    T, W = f.size(1), model.width
    mem = torch.zeros(1, T, f.size(2), device=dev)
    s = 0
    while s < T:
        lo = max(0, s - W)                       # overlap carries the band
        hi = min(T, s + chunk)
        out = model.encode(f[:, lo:hi], offset=lo)
        mem[:, s:hi] = out[:, s - lo:]
        s = hi
    return mem


@torch.no_grad()
def cci_decode(model, f, args, dev):
    """Auto-regressive rollout with Consistency Constraint Inference.

    ARST Algorithm 1. On a predicted transition at t, keep feeding the OLD
    phase and look ahead n frames; accept the transition only if all n
    lookahead predictions agree with the new phase, otherwise revert.

    Note this makes the system fixed-LAG rather than strictly causal: frame t's
    final label is emitted after observing t+n. The challenge permitted it
    (TSO-NCT's threshold smoothing has the same property), but it is not the
    same thing as a purely causal model. `--no-cci` gives the strict version.

    Exactness: with a banded mask of width W, position t attends only to
    [t-W, t], so a rolling window of W+1 gives bit-identical results to
    decoding the full prefix — that is what makes this tractable at T~8,600.
    """
    T = f.size(1)
    W = model.width
    mem = encode_memory(model, f, args.chunk, dev)
    sos = model.phase.sos

    prev = torch.full((1, T + CCI_N + 1), sos, dtype=torch.long, device=dev)
    preds = np.zeros(T, dtype=np.int64)
    banned = torch.tensor(EXCLUDED, device=dev) if args.mask_excluded else None

    def step(t, prev_seq):
        """logits at absolute position t given decoder inputs prev_seq."""
        lo = max(0, t - W)
        logits = model.decode(mem[:, lo:t + 1], prev_seq[:, lo:t + 1], offset=lo)[0, -1]
        if banned is not None:
            logits = logits.index_fill(0, banned, float("-inf"))
        return logits

    for t in range(T):
        p = int(step(t, prev).argmax())
        if args.cci and t > 0 and p != preds[t - 1]:
            probe = prev.clone()
            accept = True
            for j in range(1, CCI_N + 1):
                if t + j >= T:
                    break
                probe[0, t + j] = preds[t - 1]          # keep asserting the OLD phase
                if int(step(t + j, probe).argmax()) != p:
                    accept = False
                    break
            if not accept:
                p = int(preds[t - 1])
        preds[t] = p
        if t + 1 < prev.size(1):
            prev[0, t + 1] = p                          # shifted input for next step
    return preds


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cci", dest="cci", action="store_false")
    ap.add_argument("--mask-excluded", action="store_true",
                    help="remove classes 0/11/13 from the argmax at inference")
    ap.add_argument("--confusion", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = device_of()
    print(f"device: {dev}  band width W={args.width}  CCI={'on' if args.cci else 'off'}")

    train, val = load_split(TRAIN), load_split(VAL)
    X = np.concatenate([f for _, f, _ in train])
    mean, std = X.mean(0), X.std(0) + 1e-6
    del X

    CKPT.mkdir(parents=True, exist_ok=True)
    np.savez(CKPT / "standardize.npz", mean=mean, std=std)   # roadmap 1.3

    t0 = time.time()
    spatial = train_spatial(train, mean, std, args, dev)
    z_train, z_val = embed(spatial, train, mean, std, dev), embed(spatial, val, mean, std, dev)

    tecno = train_tecno(z_train, args, dev)
    f_train, f_val = temporal(tecno, z_train), temporal(tecno, z_val)

    arst = train_arst(f_train, args, dev)
    print(f"training done in {time.time() - t0:.0f}s")

    torch.save({"spatial": spatial.state_dict(), "tecno": tecno.state_dict(),
                "arst": arst.state_dict(), "args": vars(args)}, CKPT / "citi.pt")

    preds = []
    for (vid, f, y) in f_val:
        t1 = time.time()
        p = cci_decode(arst, f, args, dev)
        truth = y.squeeze(0).cpu().numpy()
        preds.append((vid, truth, p))
        print(f"video {vid:02d}: frame acc {(p == truth).mean():.4f}  "
              f"({time.time() - t1:.0f}s)")

    title = f"val (CITI / ARST, W={args.width}, CCI={'on' if args.cci else 'off'}"
    title += ", masked)" if args.mask_excluded else ")"
    m = report(preds, title=title, show_confusion=args.confusion)
    (CKPT / "result.json").write_text(json.dumps(
        {"mean": m["mean"], "std": m["std"], "args": vars(args)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
