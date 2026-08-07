"""The official PitVis task-2 (instrument) metric, per video, plus diagnostics.

Counterpart to `metric.py`, which does the same job for task 1. The two are
deliberately separate files: task 2 is a different problem (multi-label, not
multi-class) scored by a different vendored script with different conventions,
and folding them together would obscure exactly the differences that matter.

`official_instruments.calculate_insts_evaluation_metric` is the authority for
the headline number, the same way `official.py` is for steps. This module
replicates its internals so the score can be broken down, then **asserts** the
replication reproduces the vendored one-shot value — the guard `metric.py` uses.

Three numbers are reported per video, because the official one is contested:

- `metric`     the vendored function verbatim: weighted F1, columns compared
               POSITIONALLY. This is the challenge's number, defect included.
- `weighted`   the same weighted F1 with columns aligned BY NAME.
- `macro_f1`   macro F1, aligned by name — what Das et al. Table 6 says task 2
               was scored with.

Why three, and not one:

1. The paper (§3.4.3, and Table 6's own column header) says *macro*-F1. The
   shipped code computes *weighted*. Nothing in either source reveals which
   produced the published 41.7 / 41.6, so claiming leaderboard comparability
   from one of them would be a fabrication.
2. `hot_encode_insts` fits a separate MultiLabelBinarizer on trues and on preds
   and then appends missing columns, so whenever the two observe different class
   sets the column ORDERS diverge — and `f1_score` on DataFrames compares by
   position. Measured on a three-frame example, that is 0.333 positional against
   0.600 aligned. It fires for essentially any real prediction.

`metric` stays the headline for the same reason `metric.py` preserves the
official quirks: the point of vendoring is that our number is the challenge's
number by construction. But a defect that silently halves a score has to be
visible, so `column_order_diverged` is reported per video and `report()` says so
loudly.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from pitvis.evaluation.official_instruments import (
    calculate_insts_evaluation_metric,
    hot_encode_insts,
)

# 19 scored classes, ids 0..18 — what survives after the vendored code pops the
# -1 and -2 columns. Class 0 is SCORED, not a sentinel.
NUM_INSTRUMENTS = 19
INSTRUMENT_IDS = list(range(NUM_INSTRUMENTS))

# Sentinels, all three distinct. See notes/data-dictionary.md §4.
OUT_OF_PATIENT = -1     # slot 1 only: scope outside the patient
NO_SECONDARY = -2       # slot 2 only: this column is unused

INSTRUMENT_NAMES = {
    0: "no visible instrument / occluded", 1: "bipolar forceps", 2: "cottle",
    3: "cup forceps", 4: "dural scissors", 5: "freer elevator",
    6: "haemostatic foam", 7: "irrigation syringe", 8: "kerrisons",
    9: "micro doppler", 10: "nasal cutting forceps", 11: "pituitary rongeurs",
    12: "retractable knife", 13: "ring curette", 14: "spatula dissector",
    15: "stealth pointer", 16: "suction", 17: "surgical drill",
    18: "tissue glue",
}

METRICS = ("metric", "weighted", "macro_f1")


def pairs_to_lists(y: np.ndarray) -> list[list[int]]:
    """(T, 2) int array -> the list-of-2-lists the vendored code demands.

    Sentinels are passed through untouched: the vendored `hot_encode_insts`
    needs to see -1 and -2 so it can pop those columns.
    """
    y = np.asarray(y)
    if y.ndim != 2 or y.shape[1] != 2:
        raise ValueError(f"expected (T, 2) instrument pairs, got {y.shape}")
    return [[int(a), int(b)] for a, b in y]


def multihot(y: np.ndarray) -> np.ndarray:
    """(T, 2) pairs -> (T, 19) binary matrix over ids 0..18.

    Negative sentinels drop out, so an out-of-patient frame becomes an all-zero
    row — matching the vendored encoder, which pops the -1/-2 columns but keeps
    the row (its `remove_background_insts` call is commented out).
    """
    y = np.asarray(y)
    M = np.zeros((len(y), NUM_INSTRUMENTS), dtype=np.int8)
    for col in range(y.shape[1]):
        v = y[:, col]
        keep = v >= 0
        M[np.arange(len(y))[keep], v[keep]] = 1
    return M


def multihot_to_pairs(M: np.ndarray) -> np.ndarray:
    """(T, 19) binary -> (T, 2) pairs, ascending, padded with -2.

    Two is a structural maximum (the label is a pair of columns), so more than
    two positives is a caller bug, not something to silently truncate.
    """
    M = np.asarray(M)
    out = np.full((len(M), 2), NO_SECONDARY, dtype=np.int64)
    for i, row in enumerate(M):
        ids = np.flatnonzero(row)
        if len(ids) > 2:
            raise ValueError(
                f"frame {i}: {len(ids)} instruments predicted, but the label is a "
                f"pair of columns so 2 is a structural maximum"
            )
        if len(ids) == 0:
            out[i] = (OUT_OF_PATIENT, NO_SECONDARY)   # the vendored padding rule
        else:
            out[i, :len(ids)] = ids
    return out


def evaluate_video(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Official task-2 metric for one video, plus the two contested variants."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} trues vs {len(y_pred)} preds")
    if len(y_true) == 0:
        raise ValueError("no frames to evaluate")

    trues, preds = pairs_to_lists(y_true), pairs_to_lists(y_pred)

    # The vendored encoder, used exactly as the challenge used it.
    dt, dp = hot_encode_insts(ls_trues=trues, ls_preds=preds)
    diverged = list(dt.columns) != list(dp.columns)

    # Positional comparison — what f1_score does to those two DataFrames, and
    # therefore what the challenge script actually computed.
    positional = f1_score(dt, dp, average="weighted", zero_division=1)
    official = calculate_insts_evaluation_metric(ls_trues=trues, ls_preds=preds)
    assert abs(positional - official) < 1e-12, (
        f"replication drifted from the vendored function: "
        f"{positional!r} vs {official!r}"
    )

    # Name-aligned comparison, on our own fixed column order.
    Mt, Mp = multihot(y_true), multihot(y_pred)
    aligned_w = f1_score(Mt, Mp, average="weighted", zero_division=1)
    aligned_m = f1_score(Mt, Mp, average="macro", zero_division=1)

    return {
        "frames": int(len(y_true)),
        "metric": float(official),
        "weighted": float(aligned_w),
        "macro_f1": float(aligned_m),
        "column_order_diverged": bool(diverged),
        "exact": int((Mt == Mp).all(axis=1).sum()),
    }


def evaluate(per_video: list[tuple]) -> dict:
    """Score a split. `per_video` is [(vid, y_true, y_pred), ...], (T, 2) pairs.

    Per video then mean±std across videos, never pooled — Das et al. report
    task 2 the same way as task 1 ("calculated across the 8-testing-videos
    (mean±std)", Table 6 caption). Population std, ddof=0, as in `metric.py`.
    """
    if not per_video:
        raise ValueError("no videos to evaluate")

    videos = {vid: evaluate_video(t, p) for vid, t, p in per_video}
    scores = {k: np.array([v[k] for v in videos.values()]) for k in METRICS}

    Mt = np.concatenate([multihot(t) for _, t, _ in per_video])
    Mp = np.concatenate([multihot(p) for _, _, p in per_video])

    return {
        "videos": videos,
        "mean": {k: float(v.mean()) for k, v in scores.items()},
        "std": {k: float(v.std()) for k, v in scores.items()},
        "pooled": {
            "support": Mt.sum(axis=0),
            "predicted": Mp.sum(axis=0),
            "per_class_f1": f1_score(Mt, Mp, average=None, zero_division=0),
        },
    }


def report(per_video: list[tuple], title: str = "instruments",
           show_per_class: bool = False) -> dict:
    """Print a task-2 report and return the dict from `evaluate`."""
    m = evaluate(per_video)

    print(f"\n== {title} ==")
    print("official task-2 metric — per video, mean-averaged "
          f"(n={len(m['videos'])})\n")
    print(f"{'video':>5} {'frames':>8} {'official':>9} {'aligned-w':>10} "
          f"{'macro':>7} {'exact':>8}")
    for vid, v in m["videos"].items():
        print(f"{vid:>5} {v['frames']:>8} {v['metric']:>9.4f} "
              f"{v['weighted']:>10.4f} {v['macro_f1']:>7.4f} {v['exact']:>8}")
    print(f"{'mean':>5} {'':>8} {m['mean']['metric']:>9.4f} "
          f"{m['mean']['weighted']:>10.4f} {m['mean']['macro_f1']:>7.4f}")
    print(f"{'std':>5} {'':>8} {m['std']['metric']:>9.4f} "
          f"{m['std']['weighted']:>10.4f} {m['std']['macro_f1']:>7.4f}")

    print(f"\nCHALLENGE METRIC (task 2): {m['mean']['metric']:.4f} "
          f"± {m['std']['metric']:.4f}")

    diverged = [v for v in m["videos"].values() if v["column_order_diverged"]]
    if diverged:
        print(f"\nnote: on {len(diverged)}/{len(m['videos'])} video(s) the vendored "
              "encoder produced DIFFERENT column\n      orders for truths and "
              "predictions, and f1_score compares them positionally.\n      The "
              "'official' column above therefore includes that defect — which is "
              "what\n      the challenge script does. 'aligned-w' is the same "
              "metric compared by name.")

    print("\nnote: Das et al. Table 6 labels task 2 'Macro-F1' but the shipped script\n"
          "      computes WEIGHTED F1. Which produced the published 41.7/41.6 is not\n"
          "      recoverable from either source, so all three are printed above.\n"
          "      Validation scores are NOT comparable to the leaderboard: the paper\n"
          "      reports a -47% val->test drop for instruments (§6.5).")

    if show_per_class:
        print("\n-- per class (pooled across videos, NOT the official metric) --")
        print(f"{'id':>3}  {'name':<34} {'support':>8} {'predicted':>10} {'F1':>7}")
        for k in INSTRUMENT_IDS:
            sup = int(m["pooled"]["support"][k])
            pred = int(m["pooled"]["predicted"][k])
            f1 = m["pooled"]["per_class_f1"][k]
            f1s = f"{f1:.3f}" if sup or pred else "—"
            print(f"{k:>3}  {INSTRUMENT_NAMES[k]:<34} {sup:>8} {pred:>10} {f1s:>7}")

    return m
