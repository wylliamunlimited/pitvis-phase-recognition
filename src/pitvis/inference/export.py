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


def _bins(out: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write float32 C-order .bin sidecars; shapes are recorded in pipeline.json."""
    (out / "bin").mkdir(parents=True, exist_ok=True)
    for name, a in arrays.items():
        np.ascontiguousarray(a, dtype=np.float32).tofile(out / "bin" / f"{name}.bin")


def _shapes(out: Path) -> dict:
    import numpy as _np
    got = {}
    for f in sorted((out / "bin").glob("*.bin")):
        got[f.stem] = int(f.stat().st_size // 4)
    return got


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
    # Raw little-endian f32 sidecars alongside the .npz. The whole point of
    # this bundle is to be consumed WITHOUT Python, and a non-Python runtime
    # should not have to implement a .npy parser to read four constant tables.
    # Shapes travel in pipeline.json.
    _bins(out, {"pe": arst.pos.pe.numpy(), "phase": arst.phase.table.numpy(),
                "steps_mean": mean, "steps_std": std})

    from pitvis.training.arst import CCI_N, EXCLUDED
    info = {"spec": spec, "space": meta["space"], "feature_dim": int(len(mean)),
            "width": int(width), "mask_excluded": bool(meta["mask_excluded"]),
            "excluded": list(EXCLUDED), "cci_n": int(CCI_N),
            "sos": int(arst.phase.sos), "num_classes": 15, "d_model": 512}
    (out / "meta.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


class Windows(nn.Module):
    """SANO over pre-built causal windows: (N, W, D) -> (N, 19) logits.

    The windowing itself is a slice-and-stack over the feature array, not a
    learned op, so it stays in the host language — same reasoning as the phase
    table. Exporting it would bake the window length into the graph.
    """

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, win):
        h = self.m.drop(self.m.lstm(win)[0][:, -1])
        return self.m.instruments(h)


def export_instruments(spec: str, out: Path) -> dict:
    """Write instruments.onnx + instruments.npz (stats, thresholds)."""
    from pitvis.inference.predict import load_instrument_checkpoint

    ck = checkpoints.resolve(spec)
    mean = np.load(ck.stats)["mean"]
    model, mean, std, trained, meta = load_instrument_checkpoint(
        ck.path, ck.stats, feature_dim=len(mean), device=torch.device("cpu"))
    mod = Windows(model).eval()
    out.mkdir(parents=True, exist_ok=True)

    win = torch.zeros(4, model.window, len(mean))
    N = torch.export.Dim("N", min=1, max=65536)
    torch.onnx.export(mod, (win,), dynamo=True,
                      dynamic_shapes={"win": {0: N}}
                      ).save(str(out / "instruments.onnx"))

    th = meta.get("thresholds")
    tau = (np.array(th, np.float32) if th is not None
           else np.full(19, np.nan, np.float32))
    np.savez(out / "instruments.npz", mean=mean, std=std, thresholds=tau)
    _bins(out, {"inst_mean": mean, "inst_std": std, "inst_tau": tau})
    return {"spec": spec, "space": meta["space"], "feature_dim": int(len(mean)),
            "window": int(model.window), "num_instruments": 19,
            "per_class_thresholds": th is not None,
            "threshold": float(trained.get("threshold", 0.5))}


def export_backbone(space_name: str, out: Path) -> dict:
    """Write backbone.onnx: (N, 3, 224, 224) preprocessed pixels -> (N, D)."""
    from pitvis.data import spaces
    from pitvis.data.extract_features import build_model

    space = spaces.get(space_name)
    model, _tf, sd = build_model(torch.device("cpu"), space)
    out.mkdir(parents=True, exist_ok=True)
    px = sd["transform"]["input_size"][1]
    dummy = torch.zeros(2, 3, px, px)
    N = torch.export.Dim("N", min=1, max=4096)
    torch.onnx.export(model, (dummy,), dynamo=True,
                      dynamic_shapes={"x": {0: N}}
                      ).save(str(out / "backbone.onnx"))
    return {"space": space_name, "backbone": space.backbone,
            "feature_dim": int(sd["feature_dim"]), "transform": sd["transform"]}


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
    print(f"video_{vid:02d} steps:       memory max|Δ| {np.abs(mem - ref_mem).max():.3e}, "
          f"{len(p_torch) - differ}/{len(p_torch)} seconds agree "
          f"({100 * (1 - differ / len(p_torch)):.4f}%)")

    ok_inst = _verify_instruments(out, vid, feats)
    if differ:
        print(f"  FAIL — {differ} step second(s) differ. A prediction feeds a "
              f"surface that reads as clinical; float drift in the activations "
              f"is fine, a changed decision is not.")
    return differ == 0 and ok_inst


def _verify_instruments(out: Path, vid: int, feats: np.ndarray) -> bool:
    """Same gate for task 2: pairs must agree second for second."""
    import onnxruntime as ort
    from pitvis.evaluation.instruments import multihot_to_pairs
    from pitvis.inference.predict import (load_instrument_checkpoint,
                                          predict_instruments)
    from pitvis.models.lstm import decide, decide_per_class

    info = json.loads((out / "pipeline.json").read_text())["instruments"]
    z = np.load(out / "instruments.npz")
    mean, std = z["mean"], z["std"]
    th = None if np.isnan(z["thresholds"]).all() else z["thresholds"]

    x = ((feats - mean) / std).astype(np.float32)
    W = info["window"]
    # Left-clamped causal windows, exactly as models.lstm.causal_windows does:
    # frame 0 sees itself W times rather than zeros, so the model is never fed
    # a context it was not trained on.
    pad = np.concatenate([np.repeat(x[:1], W - 1, 0), x])
    win = np.stack([pad[i:i + W] for i in range(len(x))]).astype(np.float32)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(out / "instruments.onnx"), so)
    lg = np.concatenate([sess.run(None, {"win": win[s:s + 2048]})[0]
                         for s in range(0, len(win), 2048)])
    t = torch.from_numpy(lg)
    keep = (decide_per_class(t, torch.from_numpy(th)) if th is not None
            else decide(t, info["threshold"])).numpy()
    p_onnx = multihot_to_pairs(keep)

    ck = checkpoints.resolve(info["spec"])
    model, m2, s2, trained, meta = load_instrument_checkpoint(
        ck.path, ck.stats, feature_dim=len(mean), device=torch.device("cpu"))
    tt = meta.get("thresholds")
    p_torch = predict_instruments(
        feats, model, m2, s2, torch.device("cpu"), threshold=info["threshold"],
        chunk=trained["chunk"],
        thresholds=np.array(tt, np.float32) if tt is not None else None)

    d = int((p_onnx != p_torch).any(1).sum())
    print(f"video_{vid:02d} instruments: "
          f"{len(p_torch) - d}/{len(p_torch)} seconds agree "
          f"({100 * (1 - d / len(p_torch)):.4f}%)")
    if d:
        print(f"  FAIL — {d} instrument second(s) differ.")
    return d == 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps-model", default=checkpoints.default("steps").name,
                    metavar="SPEC", help="checkpoint to export")
    ap.add_argument("--instruments-model",
                    default=checkpoints.default("instruments").name, metavar="SPEC")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--verify", type=int, nargs="?", const=25, metavar="VIDEO",
                    help="after exporting, roll out and compare against torch "
                         "on this cached video (default 25)")
    args = ap.parse_args(argv)

    info = export(args.steps_model, args.out)
    inst = export_instruments(args.instruments_model, args.out)
    back = export_backbone(info["space"], args.out)
    (args.out / "pipeline.json").write_text(json.dumps(
        {"steps": info, "instruments": inst, "backbone": back,
         "bin_floats": _shapes(args.out)}, indent=2) + "\n")
    print(f"exported -> {args.out}")
    print(f"  backbone    {back['backbone']} -> {back['feature_dim']}-d "
          f"@{back['transform']['input_size'][1]}px")
    print(f"  steps       {args.steps_model} width={info['width']} "
          f"mask_excluded={info['mask_excluded']}")
    print(f"  instruments {args.instruments_model} window={inst['window']} "
          f"per-class-tau={inst['per_class_thresholds']}")
    if inst["space"] != info["space"]:
        raise SystemExit(
            f"space mismatch: steps reads {info['space']}, instruments "
            f"{inst['space']}. One pipeline embeds each frame ONCE, so both "
            f"heads must read the same space.")
    for f in sorted(args.out.iterdir()):
        print(f"  {f.name:14s} {f.stat().st_size / 1e6:8.2f} MB")

    if args.verify is not None:
        if not verify(args.out, args.verify, args.steps_model):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
