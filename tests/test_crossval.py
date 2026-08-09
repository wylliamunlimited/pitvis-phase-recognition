"""Pins the cross-validation harness — the thing that decides which variant wins.

Two properties matter more than anything the harness computes:

1. **Every training video is scored exactly once, by a model that never saw
   it.** If a fold's fit ever received a held-out video, the ranking would be
   measuring memorisation and would look *better* the more it leaked.
2. **Aggregation is per-video-then-mean, never pooled frame-wise.** Pooling
   inflates scores by letting opposite per-video errors cancel; `CLAUDE.md`
   records 0.583 pooled against 0.417 honest on a two-video toy case, and the
   challenge's own convention is mean-over-cases.

No data and no torch here. `fit` is a stub that returns a constant prediction,
so the tests run in milliseconds and fail for exactly one reason.
"""

import numpy as np
import pytest

from pitvis.data.dataset import TRAIN
from pitvis.data.folds import folds as fold_ids


def pairs(n: int, a: int = 16, b: int = -2) -> np.ndarray:
    """`n` frames all carrying the same instrument pair."""
    return np.tile(np.array([[a, b]], dtype=np.int64), (n, 1))


# -- the leakage property ----------------------------------------------------

def test_no_fold_ever_trains_on_a_video_it_scores():
    """The property that makes the whole ranking meaningful."""
    for i, held in enumerate(fold_ids(5)):
        train = [v for v in TRAIN if v not in set(held)]
        assert set(train).isdisjoint(held), f"fold {i} trains on what it scores"


def test_the_union_of_held_out_videos_is_every_training_video_once():
    seen = [v for f in fold_ids(5) for v in f]
    assert sorted(seen) == sorted(TRAIN)
    assert len(seen) == len(set(seen)) == 19


# -- aggregation -------------------------------------------------------------

def test_aggregation_is_per_video_then_mean_not_pooled():
    """Two videos, one perfect and one wholly wrong.

    Per-video: (1.0 + 0.0) / 2 = 0.5 exactly, whatever the frame counts are.
    Pooled would weight by length and land somewhere else entirely — here the
    long video would dominate and drag the mean toward its own score.
    """
    from pitvis.evaluation.instruments import evaluate

    good = (1, pairs(100, 16), pairs(100, 16))          # 100 frames, perfect
    bad = (2, pairs(900, 8), pairs(900, 13))            # 900 frames, all wrong
    m = evaluate([good, bad])["mean"]["weighted"]

    per_video = (
        evaluate([good])["mean"]["weighted"] + evaluate([bad])["mean"]["weighted"]
    ) / 2
    assert m == pytest.approx(per_video)
    assert m == pytest.approx(0.5, abs=1e-9)


def test_a_long_wrong_video_cannot_be_hidden_by_a_short_right_one():
    """The failure mode pooling would introduce, stated as a test."""
    from pitvis.evaluation.instruments import evaluate

    m = evaluate([(1, pairs(10, 16), pairs(10, 16)),
                  (2, pairs(5000, 8), pairs(5000, 13))])["mean"]["weighted"]
    assert m == pytest.approx(0.5, abs=1e-9), \
        "frame counts leaked into a per-video mean"


# -- the dead-class counter --------------------------------------------------

def test_never_predicted_counts_classes_the_model_never_emits():
    """The number this whole exercise is about: 9 of 19 at the start."""
    from pitvis.evaluation.instruments import evaluate

    # truth carries ids 8 and 16; the model only ever says 16.
    truth = np.concatenate([pairs(50, 8), pairs(50, 16)])
    pred = pairs(100, 16)
    pooled = evaluate([(1, truth, pred)])["pooled"]
    never = int((np.asarray(pooled["predicted"]) == 0).sum())
    assert pooled["predicted"][8] == 0
    assert pooled["predicted"][16] == 100
    assert never == 18, "every class except 16 should be counted as never predicted"


# -- the leaderboard ---------------------------------------------------------

def test_summarise_ranks_by_macro_and_flags_a_metric_regression():
    from pitvis.training.crossval import summarise

    def entry(name, macro, metric):
        return {"variant": name, "space": "resnet50",
                "mean": {"macro_f1": macro, "metric": metric, "weighted": 0.6},
                "std": {"macro_f1": 0.05, "metric": 0.04, "weighted": 0.04},
                "dead_classes": 9, "never_predicted": 9, "seconds": 60}

    text = summarise([
        entry("control", 0.25, 0.23),
        entry("winner", 0.40, 0.23),        # macro up, metric flat -> PASS
        entry("cheater", 0.45, 0.10),       # macro up, metric collapses -> FAIL
    ])
    lines = [ln for ln in text.splitlines() if ln.strip()]
    # ranked by macro: cheater (0.45) then winner (0.40) then control
    order = [ln.split()[0] for ln in lines[2:5]]
    assert order == ["cheater", "winner", "control"]
    assert "FAIL" in text and "PASS" in text


def test_summarise_says_so_when_nothing_has_run():
    from pitvis.training.crossval import summarise
    assert "no cross-validation results" in summarise([])
