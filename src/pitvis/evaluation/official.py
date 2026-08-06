"""VENDORED — the official PitVis step metric. Do not edit.

Copied verbatim from the challenge organisers' repository:

    https://github.com/dreets/pitvis
    helper_scripts/evaluation_steps.py
    commit b1cb307629d2cdf566893f0752d475629427f44d (2023-09-25)
    sha256 e41469697b865053cac396c958b91e5fda14243a2dd520f3852fb7999b263902

The ONLY modification is this docstring. Everything below is unmodified, so the
file stays diffable against upstream and the headline number is the challenge's
number by construction rather than by our reinterpretation of it. Naming style
(`ls_`/`flt_`/`int_` prefixes) is theirs; do not "fix" it.

Consumed by `pitvis.evaluation.metric`. Behaviours worth knowing before you
read it — all are faithful to upstream and deliberately preserved:

- `remove_background_steps` filters rows by GROUND TRUTH only (-1, 11, 13).
  Predictions of those classes survive on retained rows.
- `metrics.f1_score` is called with no `labels=`, so the macro average is taken
  over the labels sklearn infers from the union of cleaned trues AND preds. A
  class the model predicts but that is never true still enters the average, at
  F1 = 0, and drags the score down.
- `zero_division=1`.
- `calculate_edit_score` runs on the sequence AFTER background removal, so
  excluded rows are spliced out and the segments either side of a gap merge.
- Inputs are RAW step labels (background is -1, not 0). See `metric.decode`.
- Scores ONE video. The paper mean-averages per-video scores across videos;
  never concatenate videos before calling this.

Edge case upstream does not handle: if every ground-truth row is excluded, the
cleaned lists are empty and `calculate_edit_score` divides by zero.
`pitvis.evaluation.metric` guards against this rather than patching it here.
"""

# global imports
from itertools import groupby
from numpy import zeros
from sklearn import metrics

# strongly typed
from numpy import ndarray
from typing import List
from typing import Tuple


def main():
    # ls_trues: List[int] = []  # ground-truth steps per frame (1 integer value)
    # ls_preds: List[int] = []  # prediction steps per frame (1 integer value)
    # flt_evaluation_metric: float = calculate_steps_evaluation_metric(ls_trues=ls_trues, ls_preds=ls_preds)
    pass


def calculate_steps_evaluation_metric(ls_trues: List[int], ls_preds: List[int]) -> float:
    """Calculate the step evaluation metric from a ground-truth list and prediction list."""
    ls_trues_clean, ls_preds_clean = clean_steps(ls_trues=ls_trues, ls_preds=ls_preds)
    flt_f1_score: float = metrics.f1_score(
        y_true=ls_trues_clean,
        y_pred=ls_preds_clean,
        average="macro",
        zero_division=1,
    )
    flt_edit_score: float = calculate_edit_score(ls_trues=ls_trues_clean, ls_preds=ls_preds_clean, bl_norm=True)
    flt_metric: float = (flt_f1_score + flt_edit_score) / 2
    return flt_metric


def clean_steps(ls_trues: List[int], ls_preds: List[int]) -> Tuple[List[int], List[int]]:
    """Ensure input data is compatible with evaluation metric calculation."""
    ls_trues, ls_preds = check_steps_lists_are_compatible(ls_trues=ls_trues, ls_preds=ls_preds)
    ls_trues_cln, ls_preds_cln = remove_background_steps(ls_trues=ls_trues, ls_preds=ls_preds)
    return ls_trues_cln, ls_preds_cln


def check_steps_lists_are_compatible(ls_trues: List[int], ls_preds: List[int]) -> Tuple[List[int], List[int]]:
    """Ensure truths and predictions are compatible."""
    if len(ls_trues) != len(ls_preds):
        print(f"Lengths of truths ({len(ls_trues)}) and preds ({len(ls_preds)}) are not equal.")
        raise SystemExit
    return ls_trues, ls_preds


def remove_background_steps(ls_trues: List[int], ls_preds: List[int]) -> Tuple[List[int], List[int]]:
    """Remove classes that are not to be used in the the calculation of the evaluation metric (-1, 11, 13)."""
    ls_trues_clean: List[int] = []
    ls_preds_clean: List[int] = []
    for int_index, int_truth in enumerate(ls_trues):
        if int_truth not in [-1, 11, 13]:
            ls_trues_clean.append(int_truth)
            ls_preds_clean.append(ls_preds[int_index])
    return ls_trues_clean, ls_preds_clean


def calculate_edit_score(ls_trues: List[int], ls_preds: List[int], bl_norm: bool = True) -> float:
    """The distance between two sequences, as defined in https://en.wikipedia.org/wiki/Levenshtein_distance.
    bl_norm=True normalises the edit score for each sequence by the max number of segments in either the ground-truth or
    the prediction.
    This code is based on: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py.
    """
    ls_trues_sequence: List[int] = [key for key, _group in groupby(ls_trues)]
    ls_preds_sequence: List[int] = [key for key, _group in groupby(ls_preds)]

    int_row: int = len(ls_preds_sequence)
    int_col: int = len(ls_trues_sequence)
    d_matrix: ndarray = zeros([int_row + 1, int_col + 1], float)
    for int_i in range(int_row + 1):
        d_matrix[int_i, 0] = int_i
    for int_i in range(int_col + 1):
        d_matrix[0, int_i] = int_i
    for int_j in range(1, int_col + 1):
        for int_i in range(1, int_row + 1):
            if ls_trues_sequence[int_j - 1] == ls_preds_sequence[int_i - 1]:
                d_matrix[int_i, int_j] = d_matrix[int_i - 1, int_j - 1]
            else:
                d_matrix[int_i, int_j] = 1 + min(
                    d_matrix[int_i - 1, int_j],
                    d_matrix[int_i, int_j - 1],
                    d_matrix[int_i - 1, int_j - 1]
                )
    if bl_norm:
        flt_score: float = (1 - d_matrix[-1, -1] / max(int_row, int_col))
    else:
        flt_score: float = d_matrix[-1, -1]
    return flt_score


if __name__ == "__main__":
    main()
