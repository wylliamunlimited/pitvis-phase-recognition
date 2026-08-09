"""Fixed cross-validation folds over the 19 training videos.

WHY THIS EXISTS. The five validation videos are the only honest held-out set we
have, and Das et al. measure a **-47 point** val->test collapse for instrument
recognition (SDS-HD: 89 on validation, 41.7 on test) against -7 for steps. With
five cases and a per-video std around 0.04-0.05, ranking model variants on VAL
would mostly rank noise, and every comparison would quietly turn VAL into a
selection set. So variants are ranked by cross-validation *inside* TRAIN, and
VAL is scored exactly once, for the winner.

FROZEN AS A LITERAL, generator kept beside it — the same rule `dataset.py`
applies to TRAIN/VAL ("do not derive the split by arithmetic"). Every variant
must see the identical partition or the ranking compares partitions rather than
models; a generator that reshuffles on a library upgrade would destroy that
silently.

Each video is held out exactly once, so the union of the five folds' held-out
predictions is 19 videos, every one scored by a model that never saw it.

The four constraints, all pinned in tests/test_folds.py:

1. The folds are disjoint and their union is exactly TRAIN.
2. Every one of the 19 instrument classes appears in the *training* portion of
   every fold. Otherwise a fold guarantees F1 0.000 for that class for purely
   structural reasons and the metric measures the partition, not the model.
3. Class 17 (surgical drill) occurs in only **three** training videos
   (10, 14, 17). They land in three different folds, so no training portion is
   ever left with fewer than two.
4. Class 1 (bipolar forceps, 8 videos) is spread so every fold holds out at
   least one — otherwise a fold's score says nothing about the second-rarest
   class.

Chosen by seeded search over 200,000 shuffles, minimising the spread in
held-out frame count subject to all four: the best found is **350 seconds**
across folds of 16,748-17,098 s, so no fold is meaningfully easier by size.

A CAVEAT TO REPORT, NOT TO FIX. `evaluate_video` uses `zero_division=1`, so a
class absent from both truth and prediction scores 1.0. Folds 0 and 1 hold out
no class-17 video, so their macro is inflated exactly as the VAL headline
already is (video 21 is missing 5 of 19 classes in truth). Because the folds
are frozen this is a constant offset across variants, so the *ranking* stays
valid — but the pooled per-class table is the honest read of absolute
competence.
"""

import random

from pitvis.data.dataset import TRAIN

# Held-out video ids per fold. 19 videos as 4/4/4/4/3.
FOLDS_5: list[list[int]] = [
    [3, 4, 11, 15],
    [8, 13, 22, 23],
    [5, 6, 10, 18],
    [2, 7, 16, 17],
    [9, 14, 20],
]

_SIZES = [4, 4, 4, 4, 3]

# The two classes scarce enough to constrain the search, measured on TRAIN.
SCARCE = {
    17: [10, 14, 17],                        # surgical drill — 3 videos, 404 positives
    1: [2, 5, 6, 11, 13, 15, 16, 20],        # bipolar forceps — 8 videos, 184 positives
}


def folds(k: int = 5) -> list[list[int]]:
    """Held-out video ids per fold. Only k=5 is frozen."""
    if k != 5:
        raise SystemExit(
            f"only 5-fold is frozen (got k={k}). Re-run assign_folds and freeze "
            f"the result rather than generating folds at run time — variants "
            f"must all see the identical partition."
        )
    return [list(f) for f in FOLDS_5]


def train_videos(fold: int, k: int = 5) -> list[int]:
    """The training portion for one fold: TRAIN minus that fold's held-out ids."""
    held = set(folds(k)[fold])
    return [v for v in TRAIN if v not in held]


def assign_folds(presence: dict[int, set[int]], frames: dict[int, int],
                 seed: int = 0, trials: int = 200_000) -> list[list[int]]:
    """Search for an assignment satisfying the four constraints.

    Kept beside the frozen literal so the constraints are reproducible and the
    next person can re-derive them, not so it runs in the training path — it
    never does. `presence` maps class id -> the set of TRAIN videos containing
    it; `frames` maps video id -> length in seconds.
    """
    def ok(cand: list[list[int]]) -> bool:
        for f in cand:
            if any(not (presence[c] - set(f)) for c in presence):
                return False
            if not (presence[1] & set(f)):
                return False
        seats = {next(i for i, f in enumerate(cand) if v in f) for v in SCARCE[17]}
        return len(seats) == 3

    rng, best = random.Random(seed), None
    for _ in range(trials):
        order = list(TRAIN)
        rng.shuffle(order)
        cand, i = [], 0
        for s in _SIZES:
            cand.append(sorted(order[i:i + s]))
            i += s
        if not ok(cand):
            continue
        totals = [sum(frames[v] for v in f) for f in cand]
        spread = max(totals) - min(totals)
        if best is None or spread < best[0]:
            best = (spread, cand)
    if best is None:
        raise SystemExit("no fold assignment satisfied the constraints")
    return best[1]
