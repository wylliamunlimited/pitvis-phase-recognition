"""Pins the checkpoint registry — how a model NAME becomes a set of weights.

`training/registry.py` answers "what can I train". This answers "what have I
trained, and where does it live", which is a different question with a
different answer on any given machine. Without it `pitvis-predict` could only
be pointed at raw paths, so using a model meant knowing the on-disk layout by
heart — and layouts drift.

These tests are path arithmetic and grammar only: no torch, no checkpoint
files, nothing that depends on what happens to be trained here.
"""

import pytest

from pitvis.inference import checkpoints as C
from pitvis.paths import CKPT, CKPT_INSTRUMENTS


# -- the spec grammar --------------------------------------------------------

def test_a_reproduction_resolves_to_its_bare_file():
    c = C.resolve("arst")
    assert c.path == CKPT / "citi.pt"
    assert c.stats == CKPT / "standardize.npz"
    assert c.task == C.STEPS


def test_a_v2_family_defaults_to_the_variant_the_leaderboard_picked():
    """`instruments-v2` alone means `instruments-v2:best` — naming the family
    should give you the model that won, not an error."""
    assert C.resolve("instruments-v2").path == C.resolve("instruments-v2:best").path


def test_an_explicit_variant_selects_its_own_directory():
    c = C.resolve("instruments-v2:weighted")
    assert c.path == CKPT_INSTRUMENTS / "v2" / "weighted" / "model.pt"
    assert c.stats == CKPT_INSTRUMENTS / "v2" / "weighted" / "standardize.npz"


def test_stats_always_sit_beside_the_weights():
    """Standardisation statistics are part of the model — applying the wrong
    ones silently shifts every feature. They are never resolved separately."""
    for spec in ("arst", "arst-v2:best", "instruments", "instruments-v2:x"):
        c = C.resolve(spec)
        assert c.stats.parent == c.path.parent


# -- guards ------------------------------------------------------------------

def test_unknown_model_names_the_registered_ones():
    with pytest.raises(SystemExit, match="unknown model"):
        C.resolve("arstv2")


def test_asking_a_reproduction_for_a_variant_points_at_the_v2_family():
    """The likeliest typo, and the error should carry the fix."""
    with pytest.raises(SystemExit, match=r"arst-v2:best"):
        C.resolve("arst:best")


def test_every_family_is_assigned_to_a_real_task():
    for name, (task, _, _) in C.FAMILIES.items():
        assert task in (C.STEPS, C.INSTRUMENTS), name


def test_both_tasks_have_a_reproduction_and_a_variant_family():
    """One of each per task — the reproduction is what a fresh clone trains,
    the v2 family is where the iteration lands."""
    for task in (C.STEPS, C.INSTRUMENTS):
        fams = [(n, r) for n, (t, _, r) in C.FAMILIES.items() if t == task]
        assert sum(1 for _, r in fams if r) == 1, f"{task}: one reproduction"
        assert sum(1 for _, r in fams if not r) == 1, f"{task}: one v2 family"


# -- discovery ---------------------------------------------------------------

def test_available_only_reports_checkpoints_that_exist():
    for c in C.available():
        assert c.exists


def test_available_filters_by_task():
    for c in C.available(C.STEPS):
        assert c.task == C.STEPS


def test_default_prefers_a_leaderboard_winner_over_a_reproduction():
    have = {c.name for c in C.available(C.INSTRUMENTS)}
    picked = C.default(C.INSTRUMENTS)
    if "instruments-v2:best" in have:
        assert picked.name == "instruments-v2:best"
    elif have:
        assert picked.name in have
    else:
        assert picked is None
