"""Tests pinning pitvis/evaluation/instruments.py to the official task-2 metric.

Same contract as `test_eval.py`: expected numbers are derived by hand in the
comments so a change in sklearn, or a well-meant edit to the metric, fails
loudly instead of drifting.

Task 2 differs from task 1 in ways that are easy to "fix" by accident, so each
gets a test: three distinct sentinels, background rows kept rather than dropped,
weighted rather than macro averaging, and a hard maximum of two instruments.

The last section is the important one — agreement with the vendored function.
That is the whole reason for vendoring rather than reimplementing.
"""

import numpy as np
import pytest

from pitvis.evaluation.instruments import (
    evaluate,
    evaluate_video,
    multihot,
    multihot_to_pairs,
)
from pitvis.evaluation.official_instruments import calculate_insts_evaluation_metric


def pairs(*rows: tuple[int, int]) -> np.ndarray:
    """(T, 2) int64 instrument pairs."""
    return np.array(rows, dtype=np.int64)


# -- the three kinds of nothing ----------------------------------------------

def test_sentinels_stay_distinct():
    """-1 (out of patient), 0 (nothing visible) and -2 (unused slot) are three
    different states. Collapsing any pair of them corrupts the target."""
    y = pairs((-1, -2), (0, -2), (16, -2))
    M = multihot(y)
    assert M[0].sum() == 0        # out of patient -> all-zero row
    assert M[1].sum() == 1 and M[1][0] == 1   # class 0 is a REAL class
    assert M[2].sum() == 1 and M[2][16] == 1


def test_multihot_round_trips():
    y = pairs((13, 16), (8, -2), (-1, -2), (0, -2))
    assert np.array_equal(multihot_to_pairs(multihot(y)), y)


def test_out_of_patient_rows_are_scored_not_dropped():
    """The steps metric drops background rows; the instrument metric keeps them,
    because `remove_background_insts` is commented out upstream. An all-zero row
    predicted correctly still counts toward the frame total."""
    y = pairs((-1, -2), (-1, -2), (16, -2))
    assert evaluate_video(y, y.copy())["frames"] == 3


# -- perfect and hand-computed ------------------------------------------------

def test_perfect_prediction_scores_one():
    y = pairs((13, 16), (8, -2), (0, -2))
    r = evaluate_video(y, y.copy())
    assert r["metric"] == pytest.approx(1.0)
    assert r["weighted"] == pytest.approx(1.0)
    assert r["macro_f1"] == pytest.approx(1.0)
    assert r["exact"] == 3


def test_aligned_weighted_by_hand():
    """4 frames, truth is suction(16) throughout; 2 also have curette(13).
    Prediction gets suction right and misses curette entirely.

      class 16: TP=4 FP=0 FN=0 -> F1 = 1.0, support 4
      class 13: TP=0 FP=0 FN=2 -> precision 0/0 -> zero_division=1,
                recall 0 -> F1 = 0.0, support 2
      weighted = (4*1.0 + 2*0.0) / 6 = 2/3
    """
    y = pairs((13, 16), (13, 16), (16, -2), (16, -2))
    p = pairs((16, -2), (16, -2), (16, -2), (16, -2))
    assert evaluate_video(y, p)["weighted"] == pytest.approx(2 / 3)


def test_exact_set_match_counts_whole_frames():
    y = pairs((13, 16), (8, -2))
    p = pairs((13, 16), (16, -2))
    assert evaluate_video(y, p)["exact"] == 1


# -- multi-label, so order carries no information -----------------------------

def test_order_invariance():
    """The metric multi-hots the pair, so (8, 16) and (16, 8) are the same set."""
    y = pairs((8, 16), (8, 16))
    swapped = pairs((16, 8), (16, 8))
    assert evaluate_video(y, swapped)["metric"] == pytest.approx(1.0)
    assert np.array_equal(multihot(y), multihot(swapped))


# -- two is a structural maximum ----------------------------------------------

def test_three_instruments_is_rejected():
    """The label is a pair of columns, so no frame can carry three. Silently
    truncating would hide a caller bug."""
    M = np.zeros((1, 19), dtype=np.int8)
    M[0, [3, 8, 16]] = 1
    with pytest.raises(ValueError, match="structural maximum"):
        multihot_to_pairs(M)


def test_empty_prediction_becomes_the_padded_sentinel():
    """Upstream pads a zero-length prediction to [-1, -2]."""
    M = np.zeros((1, 19), dtype=np.int8)
    assert multihot_to_pairs(M).tolist() == [[-1, -2]]


# -- agreement with the vendored function -------------------------------------

@pytest.mark.parametrize("true_rows,pred_rows", [
    (((13, 16), (8, -2)), ((13, 16), (8, -2))),          # perfect
    (((13, 16), (8, -2)), ((16, -2), (16, -2))),         # partial
    (((-1, -2), (0, -2)), ((0, -2), (-1, -2))),          # sentinels swapped
    (((16, -2),) * 5, ((13, -2),) * 5),                  # wholly wrong
    (((3, 16), (8, 16), (0, -2)), ((3, 16), (16, -2), (0, -2))),
])
def test_matches_the_vendored_one_shot(true_rows, pred_rows):
    """Our replication must reproduce the vendored function exactly.

    `evaluate_video` asserts this internally; this test makes the check explicit
    and runs it over sentinel, partial and total-miss cases.
    """
    y, p = pairs(*true_rows), pairs(*pred_rows)
    official = calculate_insts_evaluation_metric(
        ls_trues=[[int(a), int(b)] for a, b in y],
        ls_preds=[[int(a), int(b)] for a, b in p],
    )
    assert evaluate_video(y, p)["metric"] == pytest.approx(official, abs=1e-12)


# -- the upstream column-order defect -----------------------------------------

def test_column_order_defect_is_detected_not_inherited_silently():
    """`hot_encode_insts` fits separate MultiLabelBinarizers on trues and preds,
    so when the two observe different class sets the column orders diverge and
    f1_score compares them positionally.

    We keep the vendored behaviour as the headline number, but it must never be
    silent — the flag is what `report()` warns on.
    """
    y = pairs((13, 16), (8, 16), (0, -2))
    p = pairs((16, -2), (16, -2), (0, -2))     # preds observe fewer classes
    r = evaluate_video(y, p)
    assert r["column_order_diverged"] is True
    # and the defect materially changes the score
    assert r["metric"] != pytest.approx(r["weighted"])


def test_no_divergence_when_both_sides_see_the_same_classes():
    y = pairs((13, 16), (13, 16))
    assert evaluate_video(y, y.copy())["column_order_diverged"] is False


# -- per video, then mean ------------------------------------------------------

def test_evaluate_means_the_per_video_scores():
    a = (1, pairs((16, -2), (16, -2)), pairs((16, -2), (16, -2)))   # perfect
    b = (2, pairs((13, -2), (13, -2)), pairs((16, -2), (16, -2)))   # all wrong
    m = evaluate([a, b])
    assert set(m["videos"]) == {1, 2}
    assert m["mean"]["metric"] == pytest.approx(
        (m["videos"][1]["metric"] + m["videos"][2]["metric"]) / 2
    )


def test_std_is_population_std():
    a = (1, pairs((16, -2), (16, -2)), pairs((16, -2), (16, -2)))
    b = (2, pairs((13, -2), (13, -2)), pairs((16, -2), (16, -2)))
    m = evaluate([a, b])
    scores = [m["videos"][1]["metric"], m["videos"][2]["metric"]]
    assert m["std"]["metric"] == pytest.approx(np.std(scores))       # ddof=0


# -- guards --------------------------------------------------------------------

def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_video(pairs((16, -2), (16, -2)), pairs((16, -2)))


def test_empty_split_raises():
    with pytest.raises(ValueError, match="no videos"):
        evaluate([])


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match=r"\(T, 2\)"):
        evaluate_video(np.zeros((4,), np.int64), np.zeros((4,), np.int64))
