"""Trace a tensor through the CITI cascade and print every shape.

`models/` holds architecture definitions, not a pipeline — there is nothing
here to "train once". What is useful instead is the executable form of
notes/citi-dataflow.md: push one real video's cached features through all
three stages and print the shape and parameter count at each hop.

That makes it a smoke test as well as a reference. Every stage is instantiated
and actually run, so a shape bug, a wrong channel count or a broken mask shows
up here in about a second — without waiting on a 112-second training run to
fail at stage three.

Shapes come from the real cache when it exists; otherwise a synthetic tensor
of the same width is used so the trace still works on a fresh clone.

Usage:
    uv run pitvis-models                  # trace video 01
    uv run pitvis-models --video 20       # the longest video (8,645 frames)
    uv run pitvis-models --width 0        # ablated banded mask
    uv run pitvis-models --no-mask        # skip the attention-mask diagram
"""

import argparse

import numpy as np
import torch

from pitvis.models.arst import (
    ARST,
    BAND_WIDTH,
    CCI_N,
    D_MODEL,
    N_HEADS,
    TCN_LAYERS,
    SpatialEmbedding,
    TeCNO,
    banded_causal_mask,
)
from pitvis.paths import FEATURES

NUM_CLASSES = 15


def shape(t) -> str:
    return "(" + ", ".join(f"{d:,}" for d in tuple(t.shape)) + ")"


def params(m) -> int:
    return sum(p.numel() for p in m.parameters())


def load(video: int, fallback_len: int) -> tuple[np.ndarray, np.ndarray | None, str]:
    d = FEATURES / f"video_{video:02d}"
    if (d / "features.npy").exists():
        f = np.load(d / "features.npy")
        lp = d / "labels.npy"
        return f, (np.load(lp) if lp.exists() else None), f"cache: {d}"
    rng = np.random.default_rng(0)
    f = rng.standard_normal((fallback_len, 2048), dtype=np.float32)
    lab = rng.integers(0, NUM_CLASSES, fallback_len)
    return f, lab, f"SYNTHETIC — no cache at {d}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", type=int, default=1, help="video to trace (default: 1)")
    ap.add_argument("--width", type=int, default=BAND_WIDTH,
                    help=f"ARST banded-mask width (default: {BAND_WIDTH})")
    ap.add_argument("--chunk", type=int, default=1024,
                    help="ARST chunk length (default: 1024)")
    ap.add_argument("--synthetic-len", type=int, default=4096,
                    help="length to use when the cache is absent")
    ap.add_argument("--no-mask", dest="mask", action="store_false",
                    help="skip the attention-mask diagram")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    feats, labels, source = load(args.video, args.synthetic_len)
    T, D = feats.shape
    if labels is None:                       # video 19 has features but no labels
        labels = np.zeros(T, dtype=np.int64)
        source += "  (no labels — using zeros for the trace)"

    print(f"=== cache ===\n  {source}")
    print(f"  features  {shape(torch.empty(*feats.shape))}  {feats.dtype}"
          f"  {feats.nbytes / 1e6:,.1f} MB")
    print(f"  labels    ({T:,},)  range [{labels.min()}, {labels.max()}]")

    x = torch.from_numpy(feats)
    y = torch.from_numpy(labels).long().unsqueeze(0)

    print(f"\n=== stage 1 — SpatialEmbedding ===")
    spatial = SpatialEmbedding(D, num_classes=NUM_CLASSES).eval()
    with torch.no_grad():
        z, logits1 = spatial(x)
    print(f"  in   x       {shape(x)}")
    print(f"  out  z       {shape(z)}      Linear({D:,}, {D_MODEL})")
    print(f"  out  logits  {shape(logits1)}         Linear({D_MODEL}, {NUM_CLASSES})"
          f"  [training head, discarded]")
    print(f"  parameters   {params(spatial):,}")

    print(f"\n=== stage 2 — TeCNO ===")
    tecno = TeCNO(num_classes=NUM_CLASSES).eval()
    zb = z.unsqueeze(0)
    with torch.no_grad():
        stages, ft = tecno(zb)
    rf = 1 + 2 * sum(2 ** i for i in range(TCN_LAYERS))
    print(f"  in   z       {shape(zb)}   (B, T, C)")
    print(f"       -> conv {shape(zb.transpose(1, 2))}   (B, C, T) for Conv1d")
    print(f"  stage1 out   {shape(stages[0])}")
    print(f"  stage2 out   {shape(stages[1])}     (stage 2 consumes stage 1's"
          f" {NUM_CLASSES}-way posteriors, not features)")
    print(f"  out  F_t     {shape(ft)}   <- feeds ARST")
    print(f"  parameters   {params(tecno):,}")
    print(f"  receptive field  {rf:,} frames/stage, {2 * rf - 1:,} cascaded"
          f"  = {(2 * rf - 1) / 60:.1f} min @ 1 fps")

    print(f"\n=== stage 3 — ARST (W={args.width}) ===")
    model = ARST(num_classes=NUM_CLASSES, width=args.width).eval()
    shifted = model.shift(y)
    chunk = min(args.chunk, T)
    with torch.no_grad():
        mem = model.encode(ft[:, :chunk])
        out = model.decode(mem, shifted[:, :chunk])
    print(f"  labels y     {shape(y)}")
    print(f"  shifted prev {shape(shifted)}   first 5 = {shifted[0, :5].tolist()}"
          f"  (SOS = {model.phase.sos})")
    print(f"  phase table  {shape(model.phase.table)}   segment width"
          f" {model.phase.seg}, {D_MODEL - NUM_CLASSES * model.phase.seg} tail dims"
          f" always 0")
    print(f"  encoder mem  {shape(mem)}   (chunk = {chunk:,})")
    print(f"  decoder out  {shape(out)}      Linear({D_MODEL}, {NUM_CLASSES})")
    print(f"  parameters   {params(model):,}   ({N_HEADS} heads"
          f" x {D_MODEL // N_HEADS}-d, CCI n={CCI_N})")

    if args.mask:
        n = 8
        m = banded_causal_mask(n, args.width, torch.device("cpu"))
        print(f"\n=== banded causal mask (first {n} positions, 1 = attends) ===")
        print("      key " + " ".join(f"{i}" for i in range(n)))
        for i, row in enumerate((m == 0).int().tolist()):
            print(f"  q {i}     " + " ".join(str(v) for v in row))
        print(f"  upper triangle 0 = causality; lower-left 0 = the band"
              f" (t-{args.width}..t) sliding forward")

    total = params(spatial) + params(tecno) + params(model)
    print(f"\n=== total trainable: {total:,} ===")
    for name, m_ in [("SpatialEmbedding", spatial), ("TeCNO", tecno), ("ARST", model)]:
        print(f"  {name:<18} {params(m_):>12,}   {100 * params(m_) / total:5.1f}%")
    print("  (the frozen ResNet-50 that produced the cache is not counted —"
          " it is never loaded at training time)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
