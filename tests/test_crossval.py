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


# -- the guard's SCOPE, once there is one encoder per fold -------------------
#
# A per-fold encoder legitimately saw the OTHER folds' videos — they are its
# training set. So the question "did this encoder see something it is about to
# score" has to be asked per fold, and `fold=i` is what asks it. Asking the
# old, unscoped question of a per-fold cache would refuse every honest
# configuration; asking the scoped question of one cache serving all five folds
# would wave through the leak the guard exists for.


def _saw_everything_except(fold: int) -> list[int]:
    """What fold `fold`'s encoder was fine-tuned on: TRAIN minus its own four."""
    held = set(fold_ids(5)[fold])
    return [v for v in TRAIN if v not in held]


def test_a_per_fold_encoder_passes_for_its_own_fold(tmp_path, monkeypatch):
    """The configuration the whole exercise exists to make possible."""
    from pitvis.training.crossval import check_no_leak

    for i in range(5):
        _manifest(tmp_path, monkeypatch, _saw_everything_except(i))
        check_no_leak("dinov2_ft@fold%d" % i, 5, fold=i)   # must not raise


def test_the_same_encoder_is_refused_for_every_OTHER_fold(tmp_path, monkeypatch):
    """How an off-by-one in the fold->encoder mapping is caught.

    Hand fold 0 the encoder built for fold 1 and the overlap is fold 0's own
    held-out videos, because that encoder trained on them. No separate check is
    needed — the same comparison catches it.
    """
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, _saw_everything_except(1))
    check_no_leak("dinov2_ft@fold1", 5, fold=1)
    for i in (0, 2, 3, 4):
        with pytest.raises(SystemExit, match="LEAK"):
            check_no_leak("dinov2_ft@fold1", 5, fold=i)


def test_a_per_fold_encoder_is_still_refused_unscoped(tmp_path, monkeypatch):
    """`fold=None` asks about all 19 held-out videos, and must keep doing so.

    One cache cannot serve five folds honestly if its encoder is fine-tuned,
    whichever fold you look at — which is why the scoped question is only asked
    when the harness is reading five caches.
    """
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, _saw_everything_except(2))
    with pytest.raises(SystemExit, match="LEAK"):
        check_no_leak("dinov2_ft@fold2", 5)


def test_the_scoped_refusal_names_the_fold(tmp_path, monkeypatch):
    from pitvis.training.crossval import check_no_leak

    _manifest(tmp_path, monkeypatch, _saw_everything_except(1))
    with pytest.raises(SystemExit) as e:
        check_no_leak("dinov2_ft@fold1", 5, fold=3)
    assert "fold 3 holds out" in str(e.value)


# -- which cache each fold reads ---------------------------------------------

def test_a_frozen_space_resolves_to_one_cache_per_fold():
    from pitvis.training.crossval import fold_spaces

    assert fold_spaces("resnet50", 5, False) == ["resnet50"] * 5


def test_per_fold_encoders_resolve_to_five_different_caches():
    from pitvis.training.crossval import fold_spaces

    assert fold_spaces("dinov2_ft", 5, True) == [
        f"dinov2_ft@fold{i}" for i in range(5)]


def test_per_fold_encoders_on_a_frozen_space_is_refused_not_ignored():
    """Reading the same cache five times and calling it honest is the failure."""
    from pitvis.training.crossval import fold_spaces

    with pytest.raises(SystemExit, match="no per-fold encoders"):
        fold_spaces("resnet50", 5, True)


# -- the harness, end to end over a stub ------------------------------------
#
# `fit` is a constant predictor and the loader hands back zeros, so these are
# about plumbing only: which space each fold reads, how many times the cache is
# loaded, and whether the guard runs before any fitting happens.


def _manifests(tmp_path, monkeypatch, trained_on_for):
    """A manifest per space, so a five-cache CV can be guarded in a test."""
    import json

    import pitvis.paths

    def path_for(space):
        p = tmp_path / f"{space}.json"
        if not p.exists():
            seen = trained_on_for(space)
            p.write_text(json.dumps({"space": {"_trained_on": seen}}))
        return p

    monkeypatch.setattr(pitvis.paths, "manifest_path", path_for)


def _stub_task(tmp_path, loaded):
    """A Task whose loader records the space it was asked for."""
    from pitvis.training.crossval import Task

    def loader(videos, space):
        loaded.append(space)
        return [(v, np.zeros((3, 4), np.float32), np.zeros(3, np.int64))
                for v in videos]

    def evaluate(per_video):
        return {
            "videos": {v: {"macro_f1": 1.0, "metric": 1.0} for v, _, _ in per_video},
            "mean": {"macro_f1": 1.0, "metric": 1.0},
            "std": {"macro_f1": 0.0, "metric": 0.0},
            "pooled": {"per_class_f1": [1.0], "support": [3], "predicted": [3]},
        }

    return Task(name="stub", loader=loader, evaluate=evaluate,
                metrics=("macro_f1", "metric"), primary="macro_f1",
                guard="metric", cv_dir=tmp_path / "cv")


def _run(task, *, space, per_fold, fitted=None):
    import argparse

    import torch

    from pitvis.training.crossval import cross_validate

    def fit(train, args, dev):
        if fitted is not None:
            fitted.append(sorted(v for v, _, _ in train))
        return lambda feats: np.zeros(len(feats), np.int64)

    return cross_validate(fit, argparse.Namespace(), torch.device("cpu"),
                          variant="stub", space=space, task=task, quiet=True,
                          per_fold_encoders=per_fold)


def test_a_frozen_cv_still_loads_the_cache_exactly_once(tmp_path):
    """19 videos at 2048-d is ~693 MB — reloading per fold is disk-bound.

    The per-fold change had to keep this: the loop now asks per fold, and the
    frozen answer is the same name five times, which must not become five
    loads.
    """
    loaded = []
    entry = _run(_stub_task(tmp_path, loaded), space="resnet50", per_fold=False)

    assert loaded == ["resnet50"]
    assert entry["fold_spaces"] == ["resnet50"] * 5


def test_each_fold_reads_its_own_encoder_in_order(tmp_path, monkeypatch):
    """The property the harness change exists for."""
    loaded, fitted = [], []
    _manifests(tmp_path, monkeypatch,
               lambda sp: _saw_everything_except(int(sp[-1])))
    entry = _run(_stub_task(tmp_path, loaded), space="dinov2_ft",
                 per_fold=True, fitted=fitted)

    assert loaded == [f"dinov2_ft@fold{i}" for i in range(5)]
    # One load per fold, not one per video and not five caches resident.
    assert len(loaded) == 5
    # And the fold structure is untouched: every training video scored once,
    # by a model that never saw it.
    assert sorted(int(v) for v in entry["videos"]) == sorted(TRAIN)
    for i, train in enumerate(fitted):
        assert set(train).isdisjoint(fold_ids(5)[i])


def test_the_per_fold_rows_record_which_encoder_produced_them(tmp_path, monkeypatch):
    loaded = []
    _manifests(tmp_path, monkeypatch,
               lambda sp: _saw_everything_except(int(sp[-1])))
    entry = _run(_stub_task(tmp_path, loaded), space="dinov2_ft", per_fold=True)

    assert [row["space"] for row in entry["per_fold"]] == \
        [f"dinov2_ft@fold{i}" for i in range(5)]
    assert entry["space"] == "dinov2_ft", "the declared space stays the base"


def test_the_leak_guard_runs_before_anything_is_fitted(tmp_path, monkeypatch):
    """A guard that fires after the leak has been computed has not guarded.

    Fold 3's cache is given fold 0's provenance, so the mapping is wrong and
    the run must stop with no fit at all — not after three folds of training.
    """
    loaded, fitted = [], []
    _manifests(tmp_path, monkeypatch,
               lambda sp: _saw_everything_except(0 if sp.endswith("3")
                                                 else int(sp[-1])))
    with pytest.raises(SystemExit, match="LEAK"):
        _run(_stub_task(tmp_path, loaded), space="dinov2_ft", per_fold=True,
             fitted=fitted)
    assert fitted == [] and loaded == []


def test_a_per_fold_result_is_labelled_as_one_on_the_leaderboard(tmp_path):
    """`dinov2_ft@folds` — the expensive honest version of a refused row."""
    from pitvis.training.crossval import summarise

    base = {"mean": {"macro_f1": 0.4, "metric": 0.3, "weighted": 0.6},
            "std": {"macro_f1": 0.05, "metric": 0.04, "weighted": 0.04},
            "dead_classes": 0, "never_predicted": 0, "seconds": 1}
    text = summarise([
        {"variant": "frozen", "space": "dinov2_vitb14",
         "fold_spaces": ["dinov2_vitb14"] * 5, **base},
        {"variant": "honest", "space": "dinov2_ft",
         "fold_spaces": [f"dinov2_ft@fold{i}" for i in range(5)], **base},
    ])
    assert "dinov2_ft@folds" in text
    assert "dinov2_ft@fold0" not in text, "the column names the family, not fold 0"
    assert "dinov2_vitb14 " in text, "a one-cache row is unchanged"


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
