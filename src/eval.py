"""Evaluation for PitVis step recognition, aligned to the challenge standard.

The headline number is the official one, computed by the vendored organisers'
code in src/official_metric.py:

    metric = (macro F1 + normalised edit score) / 2

scored PER VIDEO and then mean-averaged across videos, reported as mean±std.
Das et al. 2024 (arXiv 2409.01184): scores are "mean-averaged across the
8-testing-videos" and reported as mean±std, "not pooled frame-wise". Pooling
videos into one array would also corrupt the edit score, since concatenation
manufactures segment boundaries that do not exist.

Three properties of the official metric are easy to get wrong, so they are
asserted or surfaced here rather than assumed:

- Rows are excluded by GROUND TRUTH only (raw -1, 11, 13). A model that
  *predicts* one of those classes on a retained row is not filtered out; the
  class enters the macro average at F1 = 0 and lowers the score. `report`
  prints how often that happened as `leaked`, because it is a free win: at
  inference, masking classes 0/11/13 out of the argmax can only help.
- `f1_score` is called with no `labels=` and `zero_division=1`. We replicate
  that call to recover the F1/edit split, then assert our two halves recombine
  to exactly what the vendored one-shot function returns.
- The edit score is computed after exclusion, so removed rows splice the
  sequence and neighbouring segments merge.

Diagnostics (per-class recall/F1, confusion matrix) are pooled across videos
and use a fixed 12-class label set, so they are stable and readable. They are
NOT the official metric and are labelled as such in the output.

Label encoding: inputs to this module are 15-way encoded (0 = background,
k = step k). `decode` maps back to the raw space the official code expects.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from dataset import BACKGROUND, NUM_CLASSES
from official_metric import calculate_edit_score, calculate_steps_evaluation_metric, clean_steps

EXCLUDED = [0, 11, 13]      # encoded; raw [-1, 11, 13]
EXCLUDED_RAW = [-1, 11, 13]
SCORED = [k for k in range(NUM_CLASSES) if k not in EXCLUDED]

STEP_NAMES = {
    0: "background", 1: "nasal corridor creation", 2: "anterior sphenoidotomy",
    3: "septum displacement", 4: "sphenoid sinus clearance", 5: "sellotomy",
    6: "durotomy", 7: "tumour excision", 8: "haemostasis",
    9: "synthetic graft placement", 10: "fat graft placement",
    11: "gasket seal construct", 12: "dural sealant", 13: "nasal packing",
    14: "debris clearance",
}

METRICS = ("macro_f1", "edit_score", "metric")


def decode(y: np.ndarray) -> list[int]:
    """15-way encoded labels -> raw step labels (background 0 -> -1) as ints.

    The vendored official code takes plain Python ints and compares against the
    literal list [-1, 11, 13], so hand it exactly that.
    """
    raw = np.asarray(y).astype(np.int64).copy()
    raw[raw == BACKGROUND] = -1
    return raw.tolist()


def evaluate_video(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Official metric for ONE video. Inputs are 15-way encoded arrays.

    Returns the F1/edit split plus the combined metric. The split is recovered
    by replicating the vendored function's two internal calls, then checked
    against the vendored function itself.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} trues vs {len(y_pred)} preds")

    trues_raw, preds_raw = decode(y_true), decode(y_pred)
    trues_clean, preds_clean = clean_steps(ls_trues=trues_raw, ls_preds=preds_raw)
    if not trues_clean:
        raise ValueError(
            "every ground-truth row is an excluded class (-1/11/13); the official "
            "metric is undefined here (it would divide by zero)"
        )

    # Replicates official_metric.calculate_steps_evaluation_metric exactly: no
    # `labels=` (so predicted-only classes join the average) and zero_division=1.
    macro_f1 = f1_score(
        y_true=trues_clean, y_pred=preds_clean, average="macro", zero_division=1
    )
    edit_score = calculate_edit_score(
        ls_trues=trues_clean, ls_preds=preds_clean, bl_norm=True
    )
    metric = (macro_f1 + edit_score) / 2

    official = calculate_steps_evaluation_metric(ls_trues=trues_raw, ls_preds=preds_raw)
    assert abs(metric - official) < 1e-12, \
        f"split ({metric}) disagrees with vendored official metric ({official})"

    return {
        "frames": len(y_true),
        "scored_frames": len(trues_clean),
        "macro_f1": float(macro_f1),
        "edit_score": float(edit_score),
        "metric": float(metric),
        # predictions of an excluded class that survived onto a scored row
        "leaked": int(np.isin(preds_clean, EXCLUDED_RAW).sum()),
    }


def evaluate(per_video: list[tuple[int, np.ndarray, np.ndarray]]) -> dict:
    """Score a split. `per_video` is [(vid, y_true, y_pred), ...], 15-way encoded.

    Returns per-video official scores, their mean/std across videos (std is
    numpy's default population std, ddof=0), and pooled diagnostics.
    """
    if not per_video:
        raise ValueError("no videos to evaluate")

    videos = {vid: evaluate_video(t, p) for vid, t, p in per_video}
    scores = {k: np.array([v[k] for v in videos.values()]) for k in METRICS}

    y_true = np.concatenate([t for _, t, _ in per_video])
    y_pred = np.concatenate([p for _, _, p in per_video])
    keep = ~np.isin(y_true, EXCLUDED)
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    support = cm.sum(axis=1)
    with np.errstate(invalid="ignore"):
        recall = np.diag(cm) / support

    return {
        "videos": videos,
        "mean": {k: float(v.mean()) for k, v in scores.items()},
        "std": {k: float(v.std()) for k, v in scores.items()},
        "pooled": {
            "support": support,
            "recall": recall,
            "confusion_matrix": cm,
            "per_class_f1": f1_score(
                y_true[keep], y_pred[keep], labels=SCORED,
                average=None, zero_division=0,
            ),
        },
    }


def report(
    per_video: list[tuple[int, np.ndarray, np.ndarray]],
    title: str = "eval",
    show_confusion: bool = False,
) -> dict:
    """Print a report and return the metrics dict from `evaluate`."""
    m = evaluate(per_video)

    print(f"\n== {title} ==")
    print("official challenge metric — per video, mean-averaged "
          f"(n={len(m['videos'])})\n")
    print(f"{'video':>5} {'frames':>8} {'scored':>8} {'macro F1':>9} "
          f"{'edit':>7} {'metric':>7} {'leaked':>7}")
    for vid, v in m["videos"].items():
        print(f"{vid:>5} {v['frames']:>8} {v['scored_frames']:>8} "
              f"{v['macro_f1']:>9.4f} {v['edit_score']:>7.4f} "
              f"{v['metric']:>7.4f} {v['leaked']:>7}")
    print(f"{'mean':>5} {'':>8} {'':>8} {m['mean']['macro_f1']:>9.4f} "
          f"{m['mean']['edit_score']:>7.4f} {m['mean']['metric']:>7.4f}")
    print(f"{'std':>5} {'':>8} {'':>8} {m['std']['macro_f1']:>9.4f} "
          f"{m['std']['edit_score']:>7.4f} {m['std']['metric']:>7.4f}")
    print(f"\nCHALLENGE METRIC: {m['mean']['metric']:.4f} "
          f"± {m['std']['metric']:.4f}")

    leaked = sum(v["leaked"] for v in m["videos"].values())
    if leaked:
        print(f"\nnote: {leaked} predictions of an excluded class (0/11/13) landed on "
              "scored rows.\n      Each such class joins the macro average at F1 = 0. "
              "Masking 0/11/13\n      out of the argmax at inference can only raise "
              "this metric.")

    pooled = m["pooled"]
    print("\n-- diagnostics (pooled across videos, NOT the official metric) --")
    print(f"{'cls':>3}  {'name':<28} {'support':>8} {'recall':>7} {'F1':>7}")
    f1_by_cls = dict(zip(SCORED, pooled["per_class_f1"]))
    for k in range(NUM_CLASSES):
        rec = f"{pooled['recall'][k]:.3f}" if pooled["support"][k] else "—"
        f1 = f"{f1_by_cls[k]:.3f}" if k in f1_by_cls else "excl"
        print(f"{k:>3}  {STEP_NAMES[k]:<28} {pooled['support'][k]:>8} "
              f"{rec:>7} {f1:>7}")

    if show_confusion:
        print("\nconfusion matrix (rows = true, cols = pred, 15-way)")
        print("     " + "".join(f"{k:>7}" for k in range(NUM_CLASSES)))
        for k, row in enumerate(pooled["confusion_matrix"]):
            print(f"{k:>3}  " + "".join(f"{c:>7}" for c in row))

    return m
