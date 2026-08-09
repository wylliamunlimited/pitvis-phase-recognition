"""Pins the frozen cross-validation folds and the four properties they satisfy.

The folds exist because VAL is five videos and the paper measures a -47 point
val->test collapse for instruments. Ranking variants on VAL would rank noise
and would silently turn VAL into a selection set; ranking on out-of-fold TRAIN
scores does not.

The ranking is only meaningful if every variant sees the *identical* partition,
which is why the folds are a frozen literal rather than something a generator
produces at run time. These tests are what stops a well-meaning edit — a
reshuffle, a "cleaner" seed, a sixth fold — from invalidating every comparison
already recorded in notes/instrument-variants.md.

Class coverage below is measured on the 19 TRAIN videos, not assumed:
class 17 (surgical drill) is in exactly 3, class 1 (bipolar forceps) in 8.
No data is loaded here — the presence sets are the pinned constants.
"""

import pytest

from pitvis.data import folds as F
from pitvis.data.dataset import TRAIN


def all_held_out() -> list[int]:
    return [v for f in F.FOLDS_5 for v in f]


# -- constraint 1: a partition ----------------------------------------------

def test_folds_are_disjoint_and_cover_train_exactly():
    held = all_held_out()
    assert len(held) == len(set(held)), "a video is held out by two folds"
    assert sorted(held) == sorted(TRAIN)


def test_every_video_is_held_out_exactly_once():
    """This is what makes the 19 out-of-fold scores a clean per-video set."""
    for vid in TRAIN:
        assert sum(vid in f for f in F.FOLDS_5) == 1


def test_train_portion_is_the_complement():
    for i, fold in enumerate(F.FOLDS_5):
        train = F.train_videos(i)
        assert set(train) & set(fold) == set()
        assert sorted(train + fold) == sorted(TRAIN)


# -- constraint 2: no class is structurally unlearnable in any fold ----------

def test_every_scarce_class_survives_in_every_training_portion():
    """A class absent from a fold's training portion scores 0.000 for reasons
    that have nothing to do with the model, and the fold would then measure the
    partition instead."""
    for i in range(len(F.FOLDS_5)):
        train = set(F.train_videos(i))
        for cls, videos in F.SCARCE.items():
            assert train & set(videos), \
                f"fold {i} leaves class {cls} with no training video"


# -- constraints 3 and 4: the two scarce classes ----------------------------

def test_class_17_three_videos_land_in_three_different_folds():
    """Surgical drill occurs in only videos 10, 14 and 17. Concentrating them
    would leave some training portion with a single example video."""
    seats = {next(i for i, f in enumerate(F.FOLDS_5) if v in f)
             for v in F.SCARCE[17]}
    assert len(seats) == 3


def test_every_fold_holds_out_at_least_one_class_1_video():
    """Otherwise that fold's score says nothing about the second-rarest class."""
    for i, fold in enumerate(F.FOLDS_5):
        assert set(fold) & set(F.SCARCE[1]), \
            f"fold {i} holds out no bipolar-forceps video"


# -- guards ------------------------------------------------------------------

def test_only_five_fold_is_frozen():
    with pytest.raises(SystemExit, match="only 5-fold is frozen"):
        F.folds(4)


def test_folds_returns_a_copy_so_callers_cannot_mutate_the_constant():
    got = F.folds()
    got[0].append(999)
    assert 999 not in F.FOLDS_5[0]


def test_fold_sizes_are_balanced():
    """19 videos as 4/4/4/4/3 — no fold carries double another's weight."""
    sizes = sorted(len(f) for f in F.FOLDS_5)
    assert sizes == [3, 4, 4, 4, 4]
