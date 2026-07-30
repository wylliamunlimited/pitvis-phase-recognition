"""Evaluation for PitVis step recognition.

Matches the challenge's rarity exclusion: classes [-1, 11, 13] (encoded
[0, 11, 13]) are removed before scoring — background, gasket seal construct
(2 videos), nasal packing (1 video). Reports per-class accuracy (recall),
macro F1 over the 12 scored classes, and the full 15-way confusion matrix,
so macro F1 stays comparable to the paper.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from dataset import NUM_CLASSES

EXCLUDED = [0, 11, 13]  # raw [-1, 11, 13]
SCORED = [k for k in range(NUM_CLASSES) if k not in EXCLUDED]

STEP_NAMES = {
    0: "background", 1: "nasal corridor creation", 2: "anterior sphenoidotomy",
    3: "septum displacement", 4: "sphenoid sinus clearance", 5: "sellotomy",
    6: "durotomy", 7: "tumour excision", 8: "haemostasis",
    9: "synthetic graft placement", 10: "fat graft placement",
    11: "gasket seal construct", 12: "dural sealant", 13: "nasal packing",
    14: "debris clearance",
}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Score predictions. Inputs are 15-way encoded label arrays."""
    keep = ~np.isin(y_true, EXCLUDED)
    macro_f1 = f1_score(
        y_true[keep], y_pred[keep], labels=SCORED, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    with np.errstate(invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
    return {
        "macro_f1": macro_f1,
        "per_class_acc": per_class_acc,
        "confusion_matrix": cm,
    }


def report(y_true: np.ndarray, y_pred: np.ndarray, title: str = "eval") -> dict:
    """Print a human-readable report and return the metrics dict."""
    m = evaluate(y_true, y_pred)
    print(f"\n== {title} ==")
    print(f"macro F1 (excl. bg/11/13): {m['macro_f1']:.4f}")
    print(f"{'cls':>3}  {'name':<28} {'support':>8} {'acc':>7}")
    support = m["confusion_matrix"].sum(axis=1)
    for k in range(NUM_CLASSES):
        acc = m["per_class_acc"][k]
        acc_str = f"{acc:.3f}" if support[k] else "    —"
        tag = "  (excluded)" if k in EXCLUDED else ""
        print(f"{k:>3}  {STEP_NAMES[k]:<28} {support[k]:>8} {acc_str:>7}{tag}")
    return m
