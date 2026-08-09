"""Disk -> one self-contained JSON document describing a case.

This is the whole data model. The frontend consumes this and nothing else, and
`parseCase()` in `api.js` is the only code on that side that touches raw keys —
so the schema has exactly two ends and a change breaks both visibly.

Three rules the shape follows, each for a reason:

**Self-contained.** Everything needed to render a case is in one document, so
comparing cases later is N fetches and no new endpoint. The single exception is
the full 19-way instrument distribution (~500 KB), which almost nothing needs;
it lives behind `/instrument_probs`.

**Per-second data as arrays of numbers, never arrays of objects.** `{"step":
[-1,-1,1,...]}` is a fifth the size of `[{"t":0,"step":-1},...]` and indexes
directly by second. For a 4,337 s case the whole document is ~170 KB, ~25 KB
gzipped, which is why there is no binary format here.

**Ambiguity is resolved here, not downstream.** The wire format carries
`state: "none" | "one" | "two" | "out_of_patient"` rather than the raw sentinel
pair, because `(-1, -2)` means two different things depending on where it came
from — see `_instrument_state`.

Two index conventions that must not drift:

- Row `i` is wall-clock second `i`, for all 25 videos including the 25 fps
  `video_24`. Nothing here divides by fps.
- `segments[].end_s` is INCLUSIVE (`inference/predict.py` emits `t + n - 1`),
  so timeline geometry uses `end_s + 1` as the right edge.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from pitvis.app import media, names
from pitvis.app.catalogue import CaseRef
from pitvis.data.dataset import BACKGROUND
from pitvis.paths import PREDICTIONS

SCHEMA_VERSION = 1

PRE_CCI_CAVEAT = (
    "Confidence is the probability the ARST decoder assigned to the step that "
    "was actually emitted, read at the moment of decision — BEFORE the "
    "consistency constraint may override it. Where cci_held is 1 the emitted "
    "step is the previous one, not this distribution's argmax, so confidence "
    "reads low. That is the signal, not a defect: it marks exactly where the "
    "constraint is holding a phase the current frame does not support."
)

NO_OUT_OF_PATIENT_CLASS = (
    "SANO's head is 19 sigmoids and has no out-of-patient class, so a "
    "predicted state of 'none' means nothing cleared the threshold. It is NOT "
    "the scope leaving the patient, even though instruments.csv writes both as "
    "(-1, -2). Class 0 ('no visible instrument / occluded') is a third, "
    "distinct thing and is a real scored class."
)


def _r(a, nd: int = 3) -> list:
    """Round to `nd` places and hand back plain Python floats for json."""
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# --------------------------------------------------------------------------


def build_case(ref: CaseRef) -> dict:
    """The case document. Raises FileNotFoundError when nothing is predicted."""
    d = PREDICTIONS / ref.case_id
    pred_csv = d / "predictions.csv"
    if not pred_csv.exists():
        raise FileNotFoundError(
            f"no predictions for {ref.case_id} — nothing has been run on it yet"
        )

    raw_steps = pd.read_csv(pred_csv)["int_step"].to_numpy().astype(np.int64)
    seconds = int(len(raw_steps))
    summary = _read_json(d / "summary.json")
    probe = media.probe(ref.video)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "case_id": ref.case_id,
        "generated_at": _now(),
        # Carried on the document, not left to the catalogue, because it is the
        # single most important caveat on every score below: a training video
        # scores roughly twice as well as a held-out one.
        "split": ref.split,
        "video": {
            "url": f"/api/cases/{ref.case_id}/video",
            "seconds": seconds,
            **probe,
        },
        "labels": names.payload(),
        "prediction": _prediction(d, ref, raw_steps, summary),
        "instruments": _instruments(d, seconds, summary),
        "truth": _truth(ref, raw_steps, d, seconds),
        # Seams, present and empty from v1 so the renderer never has to branch
        # on their absence — only on their contents.
        "corrections": {"available": False, "edits": []},
        "explanations": {"available": False, "segments": []},
        "live": None,
    }
    doc["scores"] = _scores(ref, doc, raw_steps, d, seconds)
    return doc


# -- task 1 -----------------------------------------------------------------


def _prediction(d: Path, ref: CaseRef, raw_steps: np.ndarray,
                summary: dict) -> dict:
    steps = summary.get("steps", {})
    encoded = raw_steps.copy()
    encoded[encoded == -1] = BACKGROUND

    probs = _load(d / "step_probs.npy")
    per_second: dict[str, list] = {"step": raw_steps.tolist()}
    meta = {"available": False, "caveat": PRE_CCI_CAVEAT}

    if probs is not None and len(probs) == len(raw_steps):
        top1 = probs.argmax(1)
        ordered = np.sort(probs, axis=1)
        conf = probs[np.arange(len(probs)), encoded]
        held = top1 != encoded
        per_second |= {
            "confidence": _r(conf),
            # encoded -> raw: class 0 is background, which the wire calls -1
            "top1_step": [-1 if k == BACKGROUND else int(k) for k in top1],
            "top1_prob": _r(probs.max(1)),
            "margin": _r(ordered[:, -1] - ordered[:, -2]),
            "cci_held": held.astype(int).tolist(),
        }
        meta |= {
            "available": True,
            "stage": "pre_cci",
            "held": int(held.sum()),
            "held_frac": round(float(held.mean()), 5),
        }

    conf = per_second.get("confidence")
    return {
        "available": True,
        "stale": bool(ref.prediction.get("stale")),
        "computed_at": ref.prediction.get("computed_at"),
        "model": {
            "task1": {
                "checkpoint": Path(steps.get("checkpoint", "?")).name,
                "width": steps.get("width"),
                "cci": steps.get("cci"),
                "mask_excluded": steps.get("mask_excluded"),
            },
        },
        "segments": _segments(raw_steps, conf, per_second.get("cci_held")),
        "per_second": per_second,
        "confidence_meta": meta,
    }


def _segments(raw: np.ndarray, conf: list | None,
              held: list | None) -> list[dict]:
    """Contiguous runs of one step, with confidence aggregated over each run.

    Rebuilt here rather than read from segments.csv so the aggregates line up
    with the per-second arrays by construction. The boundaries are identical —
    both are a run-length encoding of the same column.
    """
    out = []
    edges = np.flatnonzero(np.diff(raw)) + 1
    for i, (s, e) in enumerate(zip([0, *edges], [*edges, len(raw)])):
        seg = {
            "i": i, "start_s": int(s), "end_s": int(e - 1),      # end INCLUSIVE
            "duration_s": int(e - s), "step": int(raw[s]),
            "source": "model",
        }
        if conf is not None:
            span = conf[s:e]
            seg["confidence"] = {
                "mean": round(float(np.mean(span)), 3),
                "min": round(float(np.min(span)), 3),
                "held_frac": round(float(np.mean(held[s:e])), 3) if held else 0.0,
            }
        out.append(seg)
    return out


# -- task 2 -----------------------------------------------------------------


def _instrument_state(slot1: int, slot2: int, *, truth: bool) -> str:
    """The one place the `(-1, -2)` collision is resolved.

    The same pair on the wire means different things by source. In ground truth
    it is the scope out of the patient. In a prediction it can only mean
    nothing cleared the sigmoid threshold, because SANO has no class for
    out-of-patient at all — `multihot_to_pairs` reuses that pair as padding for
    an all-zero row. Conflating them would tell a viewer the scope had left the
    patient for 26% of video_19.
    """
    if slot1 == -1:
        return "out_of_patient" if truth else "none"
    return "two" if slot2 != -2 else "one"


def _instruments(d: Path, seconds: int, summary: dict) -> dict:
    csv = d / "instruments.csv"
    if not csv.exists():
        return {"available": False,
                "reason": "task 2 was not run for this case"}

    df = pd.read_csv(csv)
    s1 = df["int_instrument1"].to_numpy().astype(np.int64)
    s2 = df["int_instrument2"].to_numpy().astype(np.int64)
    meta = summary.get("instruments", {})
    # A model with per-class thresholds has no single bar, and rendering 0.50
    # beside a class whose actual bar is 0.05 would be a plain falsehood on a
    # surface whose whole point is not flattering the model. `taus` is the
    # 19-vector when the checkpoint carries one, None otherwise; `threshold`
    # stays the scalar the older single-bar models use.
    taus = meta.get("per_class_thresholds")
    threshold = float(meta.get("threshold", 0.5))

    state = [_instrument_state(int(a), int(b), truth=False)
             for a, b in zip(s1, s2)]
    per_second: dict[str, list] = {
        "state": state,
        "slot1": [None if v < 0 else int(v) for v in s1],
        "slot2": [None if v < 0 else int(v) for v in s2],
    }

    probs = _load(d / "instrument_probs.npy")
    if probs is not None and len(probs) == seconds:
        idx = np.arange(seconds)
        per_second |= {
            "conf1": [None if v < 0 else round(float(probs[i, v]), 3)
                      for i, v in zip(idx, s1)],
            "conf2": [None if v < 0 else round(float(probs[i, v]), 3)
                      for i, v in zip(idx, s2)],
            # The best of all 19 even when nothing cleared the bar — this is
            # what lets a 'none' second show its runner-up instead of a blank.
            "max_prob": _r(probs.max(1)),
            "max_class": probs.argmax(1).astype(int).tolist(),
            # `decide` keeps only the top 2 when more clear the threshold, and
            # silently drops the rest. Surfacing that rather than hiding it.
            "capped": ((probs >= (np.asarray(taus, dtype=np.float32)
                                  if taus else threshold)).sum(1) > 2)
                      .astype(int).tolist(),
        }

    return {
        "available": True,
        "threshold": threshold,
        "per_class_thresholds": taus,
        "checkpoint": Path(meta.get("checkpoint", "?")).name,
        "note": NO_OUT_OF_PATIENT_CLASS,
        "lanes": _lanes(s1, s2),
        "per_second": per_second,
    }


def _lanes(s1: np.ndarray, s2: np.ndarray) -> list[dict]:
    """Per-class presence as [start, end_inclusive] runs — for the timeline."""
    present = np.zeros((len(s1), 19), dtype=bool)
    for slot in (s1, s2):
        ok = slot >= 0
        present[np.flatnonzero(ok), slot[ok]] = True

    lanes = []
    for k in range(19):
        col = present[:, k]
        if not col.any():
            continue
        pad = np.r_[False, col, False]
        edges = np.flatnonzero(np.diff(pad.astype(np.int8)))
        runs = [[int(a), int(b - 1)] for a, b in zip(edges[::2], edges[1::2])]
        lanes.append({"id": k, "name": names.INSTRUMENT_NAMES[k],
                      "seconds": int(col.sum()), "intervals": runs})
    lanes.sort(key=lambda x: -x["seconds"])
    return lanes


# -- ground truth -----------------------------------------------------------


def _truth(ref: CaseRef, raw_steps: np.ndarray, d: Path, seconds: int) -> dict:
    if ref.truth is None:
        return {
            "available": False,
            "reason": (f"no annotations file for {ref.case_id}. For video_19 "
                       f"this is a gap in the PitVis download, not an "
                       f"exclusion — the video is intact but unlabelled."),
        }

    from pitvis.inference.predict import load_instrument_labels, load_labels

    encoded = load_labels(ref.truth, seconds)          # background -> 0
    raw = encoded.copy()
    raw[raw == BACKGROUND] = -1
    pairs = load_instrument_labels(ref.truth, seconds)

    out = {
        "available": True,
        "source": str(ref.truth.name),
        "segments": _segments(raw, None, None),
        "per_second": {"step": raw.tolist()},
        "agreement": {
            "step": (raw == raw_steps).astype(int).tolist(),
            "frame_accuracy": round(float((raw == raw_steps).mean()), 4),
        },
    }
    if pairs is not None:
        out["per_second"] |= {
            "inst_state": [_instrument_state(int(a), int(b), truth=True)
                           for a, b in pairs],
            "inst_slot1": [None if v < 0 else int(v) for v in pairs[:, 0]],
            "inst_slot2": [None if v < 0 else int(v) for v in pairs[:, 1]],
        }
    return out


# -- scores -----------------------------------------------------------------


def _scores(ref: CaseRef, doc: dict, raw_steps: np.ndarray,
            d: Path, seconds: int) -> dict:
    """Official metrics for this ONE video.

    `evaluate_video` rather than `evaluate`: with a single video the latter's
    `mean` is just the value and its `std` is 0, and presenting that as mean±std
    would misrepresent the sample size. It also returns numpy arrays under
    `pooled`, which json.dumps cannot serialise.
    """
    if not doc["truth"]["available"]:
        return {"available": False,
                "reason": "no ground truth, so nothing to score against"}

    from pitvis.evaluation import instruments as I
    from pitvis.evaluation import metric as M
    from pitvis.inference.predict import load_instrument_labels, load_labels

    out = {"available": True,
           "scope": ("this video alone — NOT the 5-video mean±std the paper "
                     "and the README quote")}

    encoded_true = load_labels(ref.truth, seconds)
    encoded_pred = raw_steps.copy()
    encoded_pred[encoded_pred == -1] = BACKGROUND
    try:
        out["steps"] = M.evaluate_video(encoded_true, encoded_pred)
        out["steps"]["frame_accuracy"] = doc["truth"]["agreement"]["frame_accuracy"]
    except ValueError as exc:
        out["steps"] = {"error": str(exc)}

    ipred = _instrument_pairs(d, seconds)
    itrue = load_instrument_labels(ref.truth, seconds)
    if ipred is not None and itrue is not None:
        try:
            out["instruments"] = I.evaluate_video(itrue, ipred)
            out["instruments"]["note"] = (
                "'metric' is the challenge's own number, including the "
                "column-ordering defect in the vendored script. 'weighted' is "
                "the same F1 with columns aligned by name. See "
                "notes/instruments.md."
            )
        except ValueError as exc:
            out["instruments"] = {"error": str(exc)}
    return out


def _instrument_pairs(d: Path, seconds: int) -> np.ndarray | None:
    csv = d / "instruments.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return df[["int_instrument1", "int_instrument2"]].to_numpy().astype(np.int64)


# -- the lazily-fetched extra ----------------------------------------------


def instrument_probs(ref: CaseRef) -> list | None:
    """(T, 19) sigmoid outputs, rounded. ~500 KB — kept out of the case doc."""
    probs = _load(PREDICTIONS / ref.case_id / "instrument_probs.npy")
    return None if probs is None else _r(probs)


def _load(path: Path) -> np.ndarray | None:
    return np.load(path) if path.exists() else None


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
