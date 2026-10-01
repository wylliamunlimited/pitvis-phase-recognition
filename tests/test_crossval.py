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


# -- the leak guard ----------------------------------------------------------
#
# `check_no_leak` is the only thing standing between this project and the
# failure it was written after: an encoder fine-tuned on all 19 TRAIN videos,
# then cross-validated over folds drawn from that same set, scoring 0.917 macro
# against an honest 0.504. That gap was memorisation.
#
# It used to pass whenever it could not prove anything. `encoder_saw` returned
# None both for "this space is frozen" and for "the manifest is missing", and
# the guard read None as the former. A feature directory copied to another
# machine without its manifest therefore looked exactly like a frozen space.


def _manifest(tmp_path, monkeypatch, trained_on, *, exists=True):
    """Point `manifest_path` at a manifest recording `trained_on`."""
    import json

    import pitvis.paths

    target = tmp_path / "manifest.json"
    if exists:
        payload = {"space": {}}
        if trained_on is not None:
            payload["space"]["_trained_on"] = trained_on
        target.write_text(json.dumps(payload))
    monkeypatch.setattr(pitvis.paths, "manifest_path", lambda space: target)


def test_a_frozen_space_passes_without_consulting_any_cache(tmp_path, monkeypatch):
    """Frozen-ness is DECLARED in the registry, so no manifest is needed.

    Pointed at a manifest that does not exist, on purpose: a frozen space must
    not depend on the cache to be cleared.
    """
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, None, exists=False)
    check_no_leak("resnet50", 5)          # must not raise
    check_no_leak("dinov2_vitb14", 5)


def test_a_fine_tuned_space_with_no_manifest_is_REFUSED(tmp_path, monkeypatch):
    """The bug this section exists for. Absence of evidence is not evidence."""
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, None, exists=False)
    with pytest.raises(SystemExit, match="CANNOT VERIFY"):
        check_no_leak("dinov2_ft", 5)


def test_a_fine_tuned_space_whose_manifest_omits_trained_on_is_REFUSED(
        tmp_path, monkeypatch):
    """A manifest that exists but records no provenance is the same hole."""
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, None)
    with pytest.raises(SystemExit, match="CANNOT VERIFY"):
        check_no_leak("resnet50_ft", 5)


def test_a_fine_tuned_encoder_that_saw_held_out_videos_is_refused(
        tmp_path, monkeypatch):
    """The original failure: trained on all of TRAIN, cross-validated on it."""
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, list(TRAIN))
    with pytest.raises(SystemExit, match="LEAK"):
        check_no_leak("dinov2_ft", 5)


def test_one_overlapping_video_is_enough_to_refuse(tmp_path, monkeypatch):
    """Not a majority rule — a single held-out video the encoder saw is a leak."""
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, [fold_ids(5)[0][0]])
    with pytest.raises(SystemExit, match="LEAK"):
        check_no_leak("dinov2_ft", 5)


def test_a_fine_tuned_encoder_that_saw_nothing_held_out_passes(
        tmp_path, monkeypatch):
    """The honest configuration: provenance recorded, and it does not overlap.

    This is what a per-fold encoder looks like from here, and it is the case
    that must keep working — refusing every fine-tuned space would make the
    guard useless in the other direction.
    """
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, [])
    check_no_leak("dinov2_ft", 5)         # must not raise


def test_the_refusal_names_the_space_and_what_to_run(tmp_path, monkeypatch):
    """A guard that blocks a 23-hour job has to say what to do next."""
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, None, exists=False)
    with pytest.raises(SystemExit) as e:
        check_no_leak("dinov2_ft", 5)
    msg = str(e.value)
    assert "dinov2_ft" in msg
    assert "pitvis-extract" in msg
    assert "VAL" in msg


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
