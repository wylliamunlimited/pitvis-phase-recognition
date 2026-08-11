"""Export the step cascade to ONNX, so a non-Python runtime can serve it.

WHY THIS SHAPE. ARST decodes auto-regressively over its own past labels, and
the consistency constraint re-runs the decoder on speculative inputs before
accepting a transition. Neither is a graph — they are control flow over a graph.
So the cascade is split at exactly the seam the decoder already has:

    front.onnx    (T, D_in) standardised features -> (1, T, 512) memory
                  spatial -> TeCNO -> ARST.encode, computed once per video
    decode.onnx   (1, L, 512) memory window + (1, L, 512) decoder input
                  -> (15,) logits at the last position, L <= width + 1

and the loop, the CCI probes and the argmax masking are written in the host
language. `cci_decode` in training/arst.py is the reference implementation.

THE DECODER INPUT IS BUILT OUTSIDE THE GRAPH. `ARST.decode` adds a sinusoidal
positional slice to a phase embedding, and both tables are FIXED buffers — not
learned. Passing `y_win` in rather than `(prev, offset)` keeps a dynamic slice
out of the graph, and a table lookup plus an add is trivial in any host
language. Both tables are written to tables.npz.

DYNAMO, NOT THE LEGACY EXPORTER. `torch.onnx.export(..., dynamo=False)` bakes
the trace-time sequence length into `MultiheadAttention`'s internal reshape, so
the graph runs at T=64 and fails on everything else:
"Input shape:{4337,1,512}, requested shape:{64,8,64}". The dynamo path traces
through `torch.export` and keeps T symbolic.

FIDELITY. `--verify` rolls the exported graphs through the full CCI decode and
compares against the torch path second by second. On video_25 the memory agrees
to 4.6e-06 and **the predictions agree exactly, 4337 of 4337 seconds**. That is
the bar: float noise in the activations is expected, a single differing second
is not, because a prediction feeds a surface that reads as clinical.

Usage:
    uv run pitvis-export --steps-model arst-v2:best --verify 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pitvis.inference import checkpoints
from pitvis.paths import DATA, video_dir

OUT_DEFAULT = DATA / "onnx"


class Front(nn.Module):
    """Everything before the auto-regressive loop, run once per video."""

    def __init__(self, spatial, tecno, arst):
        super().__init__()
        self.spatial, self.tecno, self.arst = spatial, tecno, arst

    def forward(self, x):                      # (T, D_in) standardised
        z, _ = self.spatial(x)
        _, ft = self.tecno(z.unsqueeze(0))
        return self.arst.encode(ft, 0)         # (1, T, 512)


class DecodeStep(nn.Module):
    """One decoder step over a trailing window -> logits at the last position.

    Takes `y_win` already assembled rather than (prev, offset): see the module
    docstring — it keeps a dynamic slice out of the graph and costs the caller
    one gather and one add.
    """

    def __init__(self, arst):
        super().__init__()
        self.arst = arst

    def forward(self, mem_win, y_win):
        from pitvis.models.arst import banded_causal_mask
        m = banded_causal_mask(mem_win.size(1), self.arst.width, mem_win.device)
        return self.arst.head(self.arst.decoder(y_win, mem_win, m, m))[0, -1]


def export(spec: str, out: Path) -> dict:
    """Write front.onnx, decode.onnx, tables.npz and meta.json. Returns meta."""
    from pitvis.inference.predict import load_checkpoint

    ck = checkpoints.resolve(spec)
    # The standardisation stats carry the true feature dimension, and they are
    # resolved WITH the weights precisely so the two cannot disagree.
    mean = np.load(ck.stats)["mean"]
    spatial, tecno, arst, mean, std, trained, width, meta = load_checkpoint(
        ck.path, ck.stats, feature_dim=len(mean), device=torch.device("cpu"))

    front, step = Front(spatial, tecno, arst).eval(), DecodeStep(arst).eval()
    out.mkdir(parents=True, exist_ok=True)

    x = torch.zeros(64, len(mean))
    T = torch.export.Dim("T", min=8, max=16384)
    L = torch.export.Dim("L", min=1, max=width + 1)
    torch.onnx.export(front, (x,), dynamo=True,
                      dynamic_shapes={"x": {0: T}}).save(str(out / "front.onnx"))
    mw = torch.zeros(1, width + 1, 512)
    torch.onnx.export(step, (mw, mw.clone()), dynamo=True,
                      dynamic_shapes={"mem_win": {1: L}, "y_win": {1: L}}
                      ).save(str(out / "decode.onnx"))

    np.savez(out / "tables.npz",
             pe=arst.pos.pe.numpy(), phase=arst.phase.table.numpy(),
             mean=mean, std=std)

    from pitvis.training.arst import CCI_N, EXCLUDED
    info = {"spec": spec, "space": meta["space"], "feature_dim": int(len(mean)),
            "width": int(width), "mask_excluded": bool(meta["mask_excluded"]),
            "excluded": list(EXCLUDED), "cci_n": int(CCI_N),
            "sos": int(arst.phase.sos), "num_classes": 15, "d_model": 512}
    (out / "meta.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def rollout(s_dec, mem, tables, info) -> np.ndarray:
    """The reference host-language decode: CCI + masking over decode.onnx.

    Mirrors `training/arst.cci_decode`. This is the function a Rust port must
    reproduce, and `--verify` is what proves it did.
    """
    pe, phase = tables["pe"], tables["phase"]
    W, T = info["width"], mem.shape[1]
    preds = np.zeros(T, np.int64)
    prev = np.full(T + info["cci_n"] + 1, info["sos"], np.int64)
    banned = np.array(info["excluded"]) if info["mask_excluded"] else None

    def logits(t, pv):
        lo = max(0, t - W)
        y = phase[pv[lo:t + 1]] + pe[lo:t + 1]
        lg = s_dec.run(None, {"mem_win": mem[:, lo:t + 1],
                              "y_win": y[None].astype(np.float32)})[0]
        if banned is not None:
            lg = lg.copy()
            lg[banned] = -np.inf
        return lg

    for t in range(T):
        p = int(logits(t, prev).argmax())
        if t > 0 and p != preds[t - 1]:
            probe, accept = prev.copy(), True
            for j in range(1, info["cci_n"] + 1):
                if t + j >= T:
                    break
                probe[t + j] = preds[t - 1]
                if int(logits(t + j, probe).argmax()) != p:
                    accept = False
                    break
            if not accept:
                p = int(preds[t - 1])
        preds[t] = p
        if t + 1 < len(prev):
            prev[t + 1] = p
    return preds


def verify(out: Path, vid: int, spec: str) -> bool:
    """Roll the exported graphs out and compare with the torch path, per second."""
    import onnxruntime as ort
    from pitvis.inference.predict import load_checkpoint, predict

    info = json.loads((out / "meta.json").read_text())
    tables = np.load(out / "tables.npz")
    mean, std = tables["mean"], tables["std"]
    feats = np.load(video_dir(info["space"], vid) / "features.npy")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s_front = ort.InferenceSession(str(out / "front.onnx"), so)
    s_dec = ort.InferenceSession(str(out / "decode.onnx"), so)

    x = ((feats - mean) / std).astype(np.float32)
    mem = s_front.run(None, {"x": x})[0]
    p_onnx = rollout(s_dec, mem, tables, info)

    ck = checkpoints.resolve(spec)
    spatial, tecno, arst, m2, s2, trained, w, meta = load_checkpoint(
        ck.path, ck.stats, feature_dim=len(mean), device=torch.device("cpu"))
    with torch.no_grad():
        ref_mem = Front(spatial, tecno, arst).eval()(torch.from_numpy(x)).numpy()
    p_torch = predict(feats, spatial, tecno, arst, m2, s2, torch.device("cpu"),
                      chunk=trained["chunk"], cci=True,
                      mask_excluded=meta["mask_excluded"])

    differ = int((p_onnx != p_torch).sum())
    print(f"video_{vid:02d}: memory max|Δ| {np.abs(mem - ref_mem).max():.3e}")
    print(f"video_{vid:02d}: {len(p_torch) - differ}/{len(p_torch)} seconds agree "
          f"({100 * (1 - differ / len(p_torch)):.4f}%)")
    if differ:
        print(f"  FAIL — {differ} second(s) differ. A prediction feeds a surface "
              f"that reads as clinical; float drift in the activations is fine, "
              f"a changed decision is not.")
    return differ == 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps-model", default=checkpoints.default("steps"),
                    metavar="SPEC", help="checkpoint to export")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--verify", type=int, nargs="?", const=25, metavar="VIDEO",
                    help="after exporting, roll out and compare against torch "
                         "on this cached video (default 25)")
    args = ap.parse_args(argv)

    info = export(args.steps_model, args.out)
    print(f"exported {args.steps_model} -> {args.out}")
    print(f"  space={info['space']} dim={info['feature_dim']} "
          f"width={info['width']} mask_excluded={info['mask_excluded']}")
    for f in sorted(args.out.iterdir()):
        print(f"  {f.name:14s} {f.stat().st_size / 1e6:8.2f} MB")

    if args.verify is not None:
        if not verify(args.out, args.verify, args.steps_model):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
