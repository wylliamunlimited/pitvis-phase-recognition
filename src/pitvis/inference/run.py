"""Point the trained models at a video: `uv run pitvis-predict --video case.mp4`.

Runs **both** challenge tasks off a single feature pass — decoding is the
expensive part, so there is no reason to pay for it twice:

    task 1  spatial -> TeCNO -> ARST + CCI    -> one step per second
    task 2  causal window -> LSTM -> sigmoid  -> up to two instruments per second

    uv run pitvis-predict --video 26531686/video_19.mp4
    uv run pitvis-predict --video case.mp4 --out results/case
    uv run pitvis-predict --video 26531686/video_25.mp4 \
                          --labels 26531686/annotations_25.csv
    uv run pitvis-predict --video case.mp4 --no-instruments   # task 1 only

Writes, in the challenge's own encodings:

    predictions.csv   int_time,int_step                    task 1
    segments.csv      start_s,end_s,int_step,duration_s    task 1, human-readable
    instruments.csv   int_time,int_instrument1,int_instrument2   task 2
    summary.json      what ran, with scores if labels were given

`--labels` pointed at an `annotations_NN.csv` scores **both** tasks, since that
file carries `int_step` and `int_instrument1/2` together. A `.npy` of step
labels scores task 1 only and says so.

Task 2 is skipped with a message rather than an error when its checkpoint is
absent, so this still works on a machine where only task 1 has been trained.

Uses the feature cache when the manifest records this exact video in the
current feature space, so re-running on one of the 25 challenge videos takes
seconds instead of re-decoding. `--no-cache` forces a full decode — use it to
verify the end-to-end path really works from pixels.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pitvis.data import spaces
from pitvis.data.dataset import step_name
from pitvis.evaluation.instruments import INSTRUMENT_NAMES
from pitvis.evaluation.instruments import report as ireport
from pitvis.evaluation.metric import decode, report
from pitvis.inference import predict as P
from pitvis.paths import CKPT, CKPT_INSTRUMENTS, PREDICTIONS


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", required=True, type=Path, help="path to a video file")
    ap.add_argument("--out", type=Path,
                    help="output directory (default: predictions/<video stem>/)")
    ap.add_argument("--labels", type=Path,
                    help="ground truth (.npy or annotations CSV) — enables scoring")
    ap.add_argument("--ckpt", type=Path, default=CKPT / "citi.pt")
    ap.add_argument("--standardize", type=Path, default=CKPT / "standardize.npz")
    ap.add_argument("--instrument-ckpt", type=Path, default=None,
                    help="task-2 checkpoint (default: the best variant if "
                         "trained, else the SANO reproduction)")
    ap.add_argument("--instrument-standardize", type=Path, default=None,
                    help="task-2 standardisation stats (default: beside the "
                         "checkpoint)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="instrument sigmoid decision threshold")
    ap.add_argument("--no-steps", dest="steps", action="store_false",
                    help="skip task 1 (steps)")
    ap.add_argument("--no-instruments", dest="instruments", action="store_false",
                    help="skip task 2 (instruments)")
    ap.add_argument("--probs", action="store_true",
                    help="also write step_probs.npy / instrument_probs.npy "
                         "(per-second class distributions; pitvis-app needs these)")
    ap.add_argument("--no-cache", dest="cache", action="store_false",
                    help="always decode from the video, never reuse the feature cache")
    ap.add_argument("--no-cci", dest="cci", action="store_false",
                    help="disable the consistency constraint (strictly causal)")
    ap.add_argument("--mask-excluded", action="store_true",
                    help="remove classes 0/11/13 from the argmax")
    ap.add_argument("--width", type=int, help="override the checkpoint's band width")
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"))
    ap.add_argument("--confusion", action="store_true",
                    help="with --labels, also print the 15-way confusion matrix")
    args = ap.parse_args(argv)

    # The winner ships as the default when it exists; sano.pt is the fallback
    # so a machine that only ran the reproduction still works unchanged.
    best = CKPT_INSTRUMENTS / "v2" / "best"
    if args.instrument_ckpt is None:
        args.instrument_ckpt = (best / "model.pt" if (best / "model.pt").exists()
                                else CKPT_INSTRUMENTS / "sano.pt")
    if args.instrument_standardize is None:
        args.instrument_standardize = args.instrument_ckpt.parent / "standardize.npz"

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")
    if not args.steps and not args.instruments:
        raise SystemExit("--no-steps and --no-instruments leaves nothing to predict")

    import torch
    from pitvis.training.arst import device_of
    dev = torch.device(args.device) if args.device else device_of()

    # PREDICTIONS, not Path("predictions") — the old form was CWD-relative, so
    # running this from a subdirectory quietly wrote somewhere nothing else
    # looks. Anything reading the output (pitvis-app) resolves it from paths.
    out_dir = args.out or PREDICTIONS / args.video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"video   {args.video}")
    print(f"device  {dev}")

    # ONE PASS PER FEATURE SPACE, not one per task. The two tasks used to
    # share a single decode; a task-2 model trained on a different backbone
    # breaks that, so features are resolved per space and memoised. When both
    # tasks want the same space -- the SANO default -- this is exactly the old
    # behaviour and costs nothing.
    _feats: dict[str, np.ndarray] = {}

    def features_for(space: str) -> np.ndarray:
        if space in _feats:
            return _feats[space]
        t = time.time()
        f = P.cached_features(args.video, space) if args.cache else None
        if f is not None:
            print(f"features  [{space}] cache hit — {f.shape} "
                  f"({time.time() - t:.1f}s)")
        else:
            print(f"features  [{space}] decoding at 1 fps"
                  + ("" if args.cache else " (--no-cache)"))
            f = P.embed(args.video, dev, space)
            print(f"          [{space}] {f.shape} in {time.time() - t:.0f}s")
        _feats[space] = f
        return f

    step_space = spaces.DEFAULT
    inst_space = (P.instrument_space(args.instrument_ckpt)
                  if args.instruments else step_space)
    if args.instruments and inst_space != step_space and args.steps:
        print(f"note    the two tasks use different feature spaces "
              f"({step_space} for steps, {inst_space} for instruments), "
              f"so this video is embedded twice")

    features = features_for(step_space if args.steps else inst_space)
    summary = {"video": str(args.video), "tasks": []}
    n_frames = len(features)

    # ---- task 1: steps -----------------------------------------------------
    if args.steps:
        spatial, tecno, arst, mean, std, trained, width = P.load_checkpoint(
            args.ckpt, args.standardize, features.shape[1], dev, args.width
        )
        print(f"\ntask 1  {args.ckpt.name}  (trained W={trained['width']}, "
              f"seed={trained['seed']})")
        print(f"        W={width}, CCI={'on' if args.cci else 'off'}, "
              f"mask-excluded={'on' if args.mask_excluded else 'off'}")

        t1 = time.time()
        out = P.predict(features, spatial, tecno, arst, mean, std, dev,
                        args.chunk, args.cci, args.mask_excluded,
                        return_probs=args.probs)
        preds, sprobs = out if args.probs else (out, None)
        print(f"        {len(preds)} seconds in {time.time() - t1:.0f}s")

        # per-second, in the challenge's own encoding (background is -1)
        raw = decode(preds)
        pd.DataFrame({"int_time": np.arange(len(raw)), "int_step": raw}).to_csv(
            out_dir / "predictions.csv", index=False)

        segments = P.to_segments(preds)
        segments.to_csv(out_dir / "segments.csv", index=False)

        print(f"\n{len(segments)} step segments over {len(preds)} s "
              f"({len(preds) / 60:.1f} min)")
        print(f"  {'start':>7} {'end':>7} {'dur':>6}  step")
        for r in segments.itertuples():
            if r.duration_s < 5:            # keep the console readable
                continue
            name = step_name(r.int_step, raw=True)
            print(f"  {r.start_s:>7} {r.end_s:>7} {r.duration_s:>6}  "
                  f"{r.int_step:>3}  {name}")
        short = (segments.duration_s < 5).sum()
        if short:
            print(f"  ... plus {short} segment(s) under 5 s, omitted here "
                  f"but present in segments.csv")

        summary["tasks"].append("steps")
        summary["steps"] = {
            "frames": int(len(preds)), "segments": int(len(segments)),
            "width": width, "cci": args.cci,
            "mask_excluded": args.mask_excluded, "checkpoint": str(args.ckpt),
        }

        if sprobs is not None:
            np.save(out_dir / "step_probs.npy", sprobs)
            # held = the seconds where CCI overruled the decoder's own argmax.
            # Recorded because it is the honest caveat on every confidence
            # number downstream: there, probs.argmax() != the emitted label.
            held = int((sprobs.argmax(1) != preds).sum())
            summary["steps"]["probs"] = {
                "path": "step_probs.npy", "stage": "pre_cci",
                "encoding": "encoded_0_14", "held": held,
                "held_frac": round(held / len(preds), 5),
            }
            print(f"        probs {sprobs.shape} — CCI overruled the argmax on "
                  f"{held} of {len(preds)} s ({100 * held / len(preds):.1f}%)")

        if args.labels:
            labels = P.load_labels(args.labels, len(preds))
            m = report([(args.video.stem, labels, preds)],
                       title=f"steps — {args.video.name} vs {args.labels.name}",
                       show_confusion=args.confusion)
            summary["steps"]["metric"] = m["mean"]
            summary["steps"]["frame_accuracy"] = float((preds == labels).mean())

    # ---- task 2: instruments ----------------------------------------------
    if args.instruments:
        ifeatures = features_for(inst_space)
        loaded = P.load_instrument_checkpoint(
            args.instrument_ckpt, args.instrument_standardize,
            ifeatures.shape[1], dev,
        )
        if loaded is None:
            print(f"\ntask 2  SKIPPED — no checkpoint at {args.instrument_ckpt}\n"
                  f"        train one with `uv run pitvis-train instruments`")
        else:
            imodel, imean, istd, iargs, imeta = loaded
            n_frames = len(ifeatures)
            print(f"\ntask 2  {args.instrument_ckpt.name}  "
                  f"(variant={imeta['variant']}, space={imeta['space']}, "
                  f"window={iargs['window']}, seed={iargs['seed']})")
            if imeta["thresholds"] is not None:
                lo, hi = imeta["thresholds"].min(), imeta["thresholds"].max()
                print(f"        per-class thresholds {lo:.2f}-{hi:.2f}, "
                      f"capped by margin")
            print(f"        threshold={args.threshold}")

            t2 = time.time()
            iout = P.predict_instruments(ifeatures, imodel, imean, istd, dev,
                                         args.threshold, args.chunk,
                                         return_probs=args.probs,
                                         thresholds=imeta["thresholds"])
            inst, iprobs, ikeep = iout if args.probs else (iout, None, None)
            print(f"        {len(inst)} seconds in {time.time() - t2:.0f}s")

            pd.DataFrame({
                "int_time": np.arange(len(inst)),
                "int_instrument1": inst[:, 0],
                "int_instrument2": inst[:, 1],
            }).to_csv(out_dir / "instruments.csv", index=False)

            counts = np.bincount(inst[inst >= 0].ravel(), minlength=19)
            present = [(k, int(c)) for k, c in enumerate(counts) if c]
            present.sort(key=lambda kv: -kv[1])
            print(f"\n{len(present)} instrument class(es) predicted:")
            for k, c in present:
                print(f"  {k:>3}  {INSTRUMENT_NAMES[k]:<34} {c:>7} s "
                      f"({100 * c / len(inst):5.1f}%)")

            summary["tasks"].append("instruments")
            summary["instruments"] = {
                "frames": int(len(inst)), "threshold": args.threshold,
                "classes_predicted": len(present),
                "checkpoint": str(args.instrument_ckpt),
                "variant": imeta["variant"], "space": imeta["space"],
                "per_class_thresholds": (None if imeta["thresholds"] is None
                                         else [round(float(t), 3)
                                               for t in imeta["thresholds"]]),
            }

            if iprobs is not None:
                np.save(out_dir / "instrument_probs.npy", iprobs)
                # An all-zero row means nothing cleared the threshold. It is
                # written to instruments.csv as (-1, -2) — byte-identical to
                # the annotations' out-of-patient sentinel, which SANO has no
                # class for. Counting it here so consumers can tell them apart.
                empty = int((ikeep.sum(1) == 0).sum())
                summary["instruments"]["probs"] = {
                    "path": "instrument_probs.npy", "threshold": args.threshold,
                    "below_threshold": empty,
                    "below_threshold_frac": round(empty / len(inst), 5),
                }
                print(f"        probs {iprobs.shape} — nothing above threshold "
                      f"on {empty} of {len(inst)} s "
                      f"({100 * empty / len(inst):.1f}%)")

            if args.labels:
                truth = P.load_instrument_labels(args.labels, len(inst))
                if truth is None:
                    print("\nnote: --labels carries no instrument columns, so task 2 "
                          "is unscored.\n      Pass an annotations_NN.csv to score both.")
                else:
                    im = ireport([(args.video.stem, truth, inst)],
                                 title=f"instruments — {args.video.name}")
                    summary["instruments"]["metric"] = im["mean"]

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    written = sorted(p.name for p in out_dir.iterdir() if p.is_file())
    print(f"\nwrote {out_dir}/ -> {', '.join(written)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
