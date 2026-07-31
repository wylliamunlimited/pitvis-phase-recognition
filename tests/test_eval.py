"""Tests pinning src/eval.py to the official PitVis metric.

These exist because the official metric has three behaviours that a reasonable
reimplementation would "fix" and thereby silently diverge from the challenge:
predicted-only classes joining the macro average, exclusion splicing the
sequence before the edit score, and per-video rather than pooled scoring. Each
gets a test, with the expected numbers derived by hand so a change in sklearn or
a well-meant edit to eval.py shows up as a failure rather than a drift.

Run: uv run pytest
"""

import numpy as np
import pytest

from eval import decode, evaluate, evaluate_video
from official_metric import calculate_steps_evaluation_metric


def enc(*labels: int) -> np.ndarray:
    """15-way encoded label array (0 = background)."""
    return np.array(labels, dtype=np.int64)


# -- encoding ----------------------------------------------------------------

def test_decode_maps_background_to_minus_one():
    assert decode(enc(0, 1, 7, 14, 0)) == [-1, 1, 7, 14, -1]


def test_decode_returns_plain_ints():
    # the vendored code does `int_truth not in [-1, 11, 13]` and groupby(), so
    # it must receive real ints, not numpy scalars
    assert all(type(v) is int for v in decode(enc(0, 3)))


# -- perfect and simple cases ------------------------------------------------

def test_perfect_prediction_scores_one():
    y = enc(1, 1, 2, 2, 3)
    r = evaluate_video(y, y.copy())
    assert r["macro_f1"] == pytest.approx(1.0)
    assert r["edit_score"] == pytest.approx(1.0)
    assert r["metric"] == pytest.approx(1.0)
    assert r["leaked"] == 0


def test_known_values_by_hand():
    # true segments [1, 2]; pred segments [1]
    #   edit: Levenshtein([1], [1,2]) = 1, normalised 1 - 1/max(1,2) = 0.5
    #   F1 over inferred labels {1, 2}:
    #     class 1: TP=2 FP=2 FN=0 -> p=0.5 r=1.0 -> F1 = 2/3
    #     class 2: TP=0 FP=0 FN=2 -> p=0/0 -> zero_division=1, r=0 -> F1 = 0
    #   macro F1 = 1/3
    r = evaluate_video(enc(1, 1, 2, 2), enc(1, 1, 1, 1))
    assert r["edit_score"] == pytest.approx(0.5)
    assert r["macro_f1"] == pytest.approx(1 / 3)
    assert r["metric"] == pytest.approx((1 / 3 + 0.5) / 2)


def test_frames_and_scored_frames_counted_separately():
    r = evaluate_video(enc(0, 1, 1, 11, 13, 2), enc(0, 1, 1, 11, 13, 2))
    assert r["frames"] == 6
    assert r["scored_frames"] == 3  # rows with true in {0, 11, 13} dropped


# -- quirk 1: exclusion is by ground truth, predicted-only classes still count

def test_predicting_an_excluded_class_is_not_filtered_out():
    r = evaluate_video(enc(1, 1, 2, 2, 3, 3), enc(1, 1, 2, 2, 3, 11))
    assert r["leaked"] == 1
    # inferred labels {1, 2, 3, 11}: F1 = 1, 1, 2/3, 0 -> macro = 2/3
    assert r["macro_f1"] == pytest.approx(2 / 3)


def test_leaking_costs_more_than_an_equally_wrong_scored_prediction():
    """One wrong frame either way, but leaking into class 11 scores worse.

    This is the whole reason `report` prints `leaked`: masking 0/11/13 out of
    the argmax at inference is free score.
    """
    y_true = enc(1, 1, 2, 2, 3, 3)
    leaked = evaluate_video(y_true, enc(1, 1, 2, 2, 3, 11))
    scored = evaluate_video(y_true, enc(1, 1, 2, 2, 3, 2))
    assert leaked["leaked"] == 1 and scored["leaked"] == 0
    assert leaked["macro_f1"] == pytest.approx(2 / 3)
    # inferred labels {1, 2, 3}: F1 = 1, 0.8, 2/3 -> macro = 0.8222
    assert scored["macro_f1"] == pytest.approx((1 + 0.8 + 2 / 3) / 3)
    assert leaked["macro_f1"] < scored["macro_f1"]


def test_background_predictions_also_leak():
    r = evaluate_video(enc(1, 1, 2, 2), enc(1, 1, 0, 0))
    assert r["leaked"] == 2


# -- quirk 2: exclusion splices the sequence before the edit score -----------

def test_excluded_rows_merge_the_segments_around_them():
    """True [1,1,bg,bg,1,1] cleans to [1,1,1,1] -> ONE segment, not three.

    A naive implementation that computed segments before exclusion would see
    true segments [1, bg, 1] and score this below 1.0.
    """
    r = evaluate_video(enc(1, 1, 0, 0, 1, 1), enc(1, 1, 1, 1, 1, 1))
    assert r["edit_score"] == pytest.approx(1.0)
    assert r["macro_f1"] == pytest.approx(1.0)


def test_edit_score_ignores_segment_duration():
    # both collapse to segments [1, 2]; only the order/identity matters
    assert evaluate_video(enc(1, 2, 2, 2), enc(1, 1, 1, 2))["edit_score"] == \
        pytest.approx(1.0)


def test_edit_score_punishes_flicker():
    """Frame-wise accuracy 0.5 but the prediction oscillates -> edit score sags.

    This is the term a frame-wise model loses on, and the reason the linear
    probe is expected to score poorly even where its accuracy looks fine.
    """
    y_true = enc(*([1] * 4 + [2] * 4))
    flicker = evaluate_video(y_true, enc(1, 2, 1, 2, 1, 2, 1, 2))
    clean = evaluate_video(y_true, enc(1, 1, 1, 1, 2, 2, 2, 2))
    assert clean["edit_score"] == pytest.approx(1.0)
    assert flicker["edit_score"] < 0.5


# -- agreement with the vendored one-shot function ---------------------------

@pytest.mark.parametrize("true_seq,pred_seq", [
    ((1, 1, 2, 2, 3), (1, 1, 2, 2, 3)),
    ((1, 1, 2, 2), (1, 1, 1, 1)),
    ((0, 1, 1, 11, 2, 13, 2), (1, 1, 2, 0, 2, 2, 11)),
    ((7,) * 10, (8,) * 10),
    ((1, 2, 3, 4, 5, 6, 7), (7, 6, 5, 4, 3, 2, 1)),
])
def test_split_recombines_to_the_official_number(true_seq, pred_seq):
    """Our F1/edit split must reproduce the vendored function's single number.

    evaluate_video asserts this internally; this test makes the check explicit
    and runs it over sequences that exercise exclusion, leakage and reordering.
    """
    y_true, y_pred = enc(*true_seq), enc(*pred_seq)
    r = evaluate_video(y_true, y_pred)
    official = calculate_steps_evaluation_metric(
        ls_trues=decode(y_true), ls_preds=decode(y_pred)
    )
    assert r["metric"] == pytest.approx(official, abs=1e-12)


# -- quirk 3: per-video scoring, mean-averaged ------------------------------

def test_evaluate_means_the_per_video_scores():
    a = (1, enc(1, 1, 2, 2), enc(1, 1, 2, 2))          # perfect -> 1.0
    b = (2, enc(1, 1, 2, 2), enc(1, 1, 1, 1))          # the hand-checked case
    m = evaluate([a, b])
    expected = (1.0 + (1 / 3 + 0.5) / 2) / 2
    assert m["mean"]["metric"] == pytest.approx(expected)
    assert m["mean"]["edit_score"] == pytest.approx((1.0 + 0.5) / 2)
    assert set(m["videos"]) == {1, 2}


def test_std_is_population_std_across_videos():
    a = (1, enc(1, 1, 2, 2), enc(1, 1, 2, 2))
    b = (2, enc(1, 1, 2, 2), enc(1, 1, 1, 1))
    m = evaluate([a, b])
    scores = [m["videos"][1]["metric"], m["videos"][2]["metric"]]
    assert m["std"]["metric"] == pytest.approx(np.std(scores))  # ddof=0


def test_pooling_videos_flatters_the_score():
    """Why we never concatenate: pooling both merges and cancels errors.

    Two videos, each collapsing a two-step sequence into one predicted step
    (edit 0.5, macro F1 1/3 apiece). Concatenated, the opposite mistakes cancel
    in the frame-wise F1 and the merged segment sequence looks closer to the
    truth, so the pooled number is much higher than the honest per-video mean.
    """
    v1_true, v1_pred = enc(1, 1, 2, 2), enc(1, 1, 1, 1)
    v2_true, v2_pred = enc(2, 2, 1, 1), enc(2, 2, 2, 2)
    per_video = evaluate([(1, v1_true, v1_pred), (2, v2_true, v2_pred)])
    pooled = evaluate_video(
        np.concatenate([v1_true, v2_true]), np.concatenate([v1_pred, v2_pred])
    )
    assert per_video["mean"]["metric"] == pytest.approx((1 / 3 + 0.5) / 2)  # 0.4167
    assert pooled["metric"] == pytest.approx((0.5 + 2 / 3) / 2)             # 0.5833
    assert pooled["metric"] > per_video["mean"]["metric"]


# -- diagnostics -------------------------------------------------------------

def test_pooled_diagnostics_cover_all_15_classes():
    m = evaluate([(1, enc(0, 1, 2, 11, 13), enc(0, 1, 2, 11, 13))])
    assert len(m["pooled"]["support"]) == 15
    assert m["pooled"]["confusion_matrix"].shape == (15, 15)
    assert len(m["pooled"]["per_class_f1"]) == 12  # scored classes only


# -- guards ------------------------------------------------------------------

def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate_video(enc(1, 2, 3), enc(1, 2))


def test_all_excluded_truth_raises_instead_of_dividing_by_zero():
    with pytest.raises(ValueError, match="excluded class"):
        evaluate_video(enc(0, 0, 11, 13), enc(1, 2, 3, 4))


def test_empty_split_raises():
    with pytest.raises(ValueError, match="no videos"):
        evaluate([])
