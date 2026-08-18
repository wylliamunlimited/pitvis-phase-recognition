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


# -- tag decoding: one owner, and it reads the legacy location ---------------

def test_tags_default_to_what_the_reproductions_were_trained_with():
    """A checkpoint with no tags at all is sano.pt or citi.pt, and those
    predate every tag. Absent must mean "the original", never "unknown"."""
    steps = C.read_tags({}, C.STEPS)
    assert steps["space"] == "resnet50"
    assert steps["arch"] == "arst"
    assert steps["mask_excluded"] is False
    assert steps["logit_adjust"] is None
    assert C.read_tags({}, C.INSTRUMENTS)["arch"] == "sano-lstm"


def test_mask_excluded_is_recovered_from_args_when_the_tag_is_missing():
    """`pitvis-train arst --mask-excluded` recorded the flag only inside
    `args`, so reading the top level alone silently un-masked a model that was
    trained masked — and masking is worth ~0.076 on the official metric."""
    assert C.read_tags({"args": {"mask_excluded": True}})["mask_excluded"] is True


def test_the_top_level_tag_beats_the_args_record():
    """`args` is a record of a command line; the tag is the decision. When both
    are present and disagree, the tag is what inference must honour."""
    ck = {"mask_excluded": True, "args": {"mask_excluded": False}}
    assert C.read_tags(ck)["mask_excluded"] is True


def test_the_logit_adjustment_round_trips_as_an_array_not_a_tau():
    """`tau * log(prior)` is computed from the training labels of the split the
    model was fitted on, so tau alone cannot reconstruct it at inference."""
    tags = C.read_tags({"prior_tau": 0.5, "logit_adjust": [-1.0, -2.0, -3.0]})
    assert tags["prior_tau"] == 0.5
    assert tags["logit_adjust"] == [-1.0, -2.0, -3.0]


# -- which checkpoint is the default ----------------------------------------

def _plant(root, variant, score=None):
    """A checkpoint on disk: weights, stats, and optionally its own scoring."""
    import json
    d = root / "v2" / variant
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.pt").write_bytes(b"")
    (d / "standardize.npz").write_bytes(b"")
    if score is not None:
        (d / "result.json").write_text(json.dumps({"mean": {C.PRIMARY_METRIC: score}}))
    return d


def test_the_default_is_the_best_scoring_checkpoint_not_the_first_alphabetically(
        tmp_path, monkeypatch):
    """THE BUG THIS EXISTS FOR. `available()` returns names sorted, so `best`
    sorted ahead of `best@dinov2_ft` and the old `endswith(":best")` rule
    resolved the default to the model the fine-tuned encoder beat by 0.0998."""
    _plant(tmp_path, "best", score=0.4420)
    _plant(tmp_path, "best@dinov2_ft", score=0.6147)
    monkeypatch.setitem(C.FAMILIES, "arst-v2", (C.STEPS, tmp_path, None))
    monkeypatch.setitem(C.FAMILIES, "arst", (C.STEPS, tmp_path / "none", "citi.pt"))
    assert C.default(C.STEPS).name == "arst-v2:best@dinov2_ft"


def test_a_worse_new_variant_does_not_become_the_default(tmp_path, monkeypatch):
    _plant(tmp_path, "best", score=0.4420)
    _plant(tmp_path, "zzz-experiment", score=0.1000)
    monkeypatch.setitem(C.FAMILIES, "arst-v2", (C.STEPS, tmp_path, None))
    monkeypatch.setitem(C.FAMILIES, "arst", (C.STEPS, tmp_path / "none", "citi.pt"))
    assert C.default(C.STEPS).name == "arst-v2:best"


def test_unscored_checkpoints_fall_back_to_the_name_convention(tmp_path, monkeypatch):
    """A machine that trained but never scored still gets a sensible answer."""
    _plant(tmp_path, "control")
    _plant(tmp_path, "best")
    monkeypatch.setitem(C.FAMILIES, "arst-v2", (C.STEPS, tmp_path, None))
    monkeypatch.setitem(C.FAMILIES, "arst", (C.STEPS, tmp_path / "none", "citi.pt"))
    assert C.default(C.STEPS).name == "arst-v2:best"


def test_ties_resolve_deterministically(tmp_path, monkeypatch):
    _plant(tmp_path, "aaa", score=0.5)
    _plant(tmp_path, "bbb", score=0.5)
    monkeypatch.setitem(C.FAMILIES, "arst-v2", (C.STEPS, tmp_path, None))
    monkeypatch.setitem(C.FAMILIES, "arst", (C.STEPS, tmp_path / "none", "citi.pt"))
    assert C.default(C.STEPS).name == "arst-v2:aaa"


def test_a_malformed_result_file_is_not_fatal(tmp_path, monkeypatch):
    """A truncated or half-written result.json must not take out `--list-models`."""
    d = _plant(tmp_path, "best")
    (d / "result.json").write_text("{ not json")
    monkeypatch.setitem(C.FAMILIES, "arst-v2", (C.STEPS, tmp_path, None))
    monkeypatch.setitem(C.FAMILIES, "arst", (C.STEPS, tmp_path / "none", "citi.pt"))
    assert C.resolve("arst-v2:best").score() is None
    assert C.default(C.STEPS).name == "arst-v2:best"


def test_the_ranking_metric_is_macro_not_the_official_number():
    """Task 2's official metric carries the vendored column-ordering defect,
    which reads the fine-tuned encoder as the WORST model tried (0.3220) where
    macro reads it as the best (0.5333). Ranking on it would pick backwards."""
    assert C.PRIMARY_METRIC == "macro_f1"


# -- pitvis-eval resolves its inputs from the checkpoint ---------------------

def test_eval_exposes_a_space_override_that_defaults_to_the_checkpoint_tag():
    """resnet50/resnet50_ft are both 2048-d and dinov2_vitb14/dinov2_ft both
    768-d, so scoring against the wrong cache loads cleanly and reports a wrong
    number. The default must come from the checkpoint, not from a constant."""
    from pitvis.evaluation import run as eval_run
    ap = _eval_parser(eval_run)
    assert ap.get_default("space") is None
    assert ap.get_default("standardize") is None


def _eval_parser(eval_run):
    """`pitvis-eval` builds its parser inside main(), so reach it the same way
    argparse does — by running main with --help and catching the exit."""
    import argparse
    import contextlib
    import io
    holder = {}
    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, *a, **kw):
        holder["ap"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = capture
    try:
        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(SystemExit):
            eval_run.main([])
    finally:
        argparse.ArgumentParser.parse_args = real_parse
    return holder["ap"]
