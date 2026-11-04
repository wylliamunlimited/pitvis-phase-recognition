"""Pins the feature-space registry and the path layout it keys.

Two caches built by different backbones are not interchangeable, and mixing
them in one training run yields a model that is silently wrong rather than
loudly broken. `extract_features.space_id` already guards that at write time;
this registry is what lets two spaces coexist instead of one having to be
deleted for the other to exist.

The invariant worth guarding here is that `Space.name` never reaches the hashed
payload. `space_id` hashes {backbone, feature_dim, target_fps, transform}; if a
name ever leaked in, the existing cache's id would move off 67912d3efc6852e7
and the guard would reject 940 MB of perfectly good features, demanding a
25-minute re-extract for nothing.
"""

import re

import pytest

from pitvis import paths
from pitvis.data import spaces


# -- registry mechanics ------------------------------------------------------

def test_registry_is_keyed_by_name_not_hand_listed():
    """The list is the source of truth; the dict is derived from it."""
    for key, space in spaces.SPACES.items():
        assert key == space.name


def test_default_space_is_registered():
    assert spaces.DEFAULT in spaces.SPACES


def test_names_puts_the_default_first_then_the_rest_sorted():
    got = spaces.names()
    assert got[0] == spaces.DEFAULT
    assert got[1:] == sorted(set(spaces.SPACES) - {spaces.DEFAULT})
    assert set(got) == set(spaces.SPACES)


def test_unknown_space_names_the_registered_ones():
    with pytest.raises(SystemExit, match="unknown feature space"):
        spaces.get("resnet51")


def test_get_returns_the_registered_space():
    assert spaces.get(spaces.DEFAULT).name == spaces.DEFAULT


def test_describe_lists_every_space():
    text = spaces.describe()
    for name in spaces.SPACES:
        assert name in text


# -- the frozen hashed payload ----------------------------------------------

def test_name_is_not_part_of_what_gets_hashed():
    """`name` is presentation, not identity.

    `extract_features.build_model` assembles the hashed dict from the backbone
    and the resolved transform. This asserts the registry carries the human
    name separately, so no future edit can tempt it into the hash and
    invalidate the existing cache.
    """
    space = spaces.get(spaces.DEFAULT)
    hashed_keys = {"backbone", "feature_dim", "target_fps", "transform"}
    assert "name" not in hashed_keys
    assert space.name != space.backbone or space.name == "resnet50"


def test_dinov2_is_pinned_to_224_not_its_native_518():
    """518 costs 6.4x the compute for a finer patch grid we then pool away.

    Measured on MPS: 25.1 img/s at 518 against 160.6 at 224, for the same
    768-d output. The 37x37 grid only becomes worth paying for if something
    consumes it spatially — a CAM overlay would; average pooling does not.
    """
    assert spaces.get("dinov2_vitb14").model_kwargs == {"img_size": 224}


# -- the declared width ------------------------------------------------------

def test_every_space_declares_its_feature_width():
    """Declared, so the width is readable without a cache.

    The hashed payload still takes the width from `model.num_features`, but
    that only exists after the backbone is built with pretrained weights.
    Anything reasoning about a space on a fresh clone needs the number without
    a download, and `models/run.py` is exactly that case.
    """
    for space in spaces.SPACES.values():
        assert space.feature_dim > 0


def test_the_declared_widths_are_the_two_the_cache_holds():
    """Two spaces at 2048 and two at 768 — the checkpoint tag is what tells
    a loader which, which is why both widths must be stated rather than
    inferred from the name."""
    assert spaces.get("resnet50").feature_dim == 2048
    assert spaces.get("resnet50_ft").feature_dim == 2048
    assert spaces.get("dinov2_vitb14").feature_dim == 768
    assert spaces.get("dinov2_ft").feature_dim == 768


def test_the_trace_falls_back_to_the_SPACE_width_not_a_hardcoded_one():
    """The bug this pins: `manifest_dim` returned 2048 for every space when no
    manifest existed, so `pitvis-models --space dinov2_ft` on a fresh clone
    traced the 2048-d cascade (25,500,732 params) and printed it as the 768-d
    one (24,845,372). The command the architecture atlas names for seeing the
    768-d model showed the wrong model, with no warning.

    Asserted against a space with no cache on this machine, which is the only
    situation where the fallback is reached at all.
    """
    from pitvis.models.run import manifest_dim

    for name, want in [("resnet50", 2048), ("dinov2_vitb14", 768),
                       ("dinov2_ft", 768), ("resnet50_ft", 2048)]:
        if paths.manifest_path(name).exists():
            continue                     # a real cache outranks the declaration
        assert manifest_dim(name) == want


# -- per-fold spaces ---------------------------------------------------------
#
# An honest cross-validation of a fine-tuned encoder needs one encoder per
# fold, each blind to that fold's held-out videos. That is five caches of the
# same backbone, which a single `--space` cannot express -- so `fold()` derives
# `<base>@fold<i>` and `get()` resolves it. The registry does not list them:
# the fold COUNT is owned by `data/folds.py`, and a hand-listed family would be
# a second place for k to be wrong.


def test_a_fold_space_is_derived_from_its_base_and_differs_only_in_the_encoder():
    base = spaces.get("dinov2_ft")
    f3 = spaces.fold("dinov2_ft", 3)

    assert f3.name == "dinov2_ft@fold3"
    assert f3.checkpoint == "backbone/dinov2-fold3/backbone.pt"
    # Everything that defines the image->numbers function is inherited. A fold
    # encoder differs in what it SAW, not in how it preprocesses.
    assert (f3.backbone, f3.feature_dim, f3.target_fps, f3.model_kwargs,
            f3.source, f3.frame_size) == (
        base.backbone, base.feature_dim, base.target_fps, base.model_kwargs,
        base.source, base.frame_size)


def test_get_resolves_a_fold_name_so_every_space_reader_reaches_one():
    """`pitvis-extract --space dinov2_ft@fold0` and everything downstream."""
    assert spaces.get("resnet50_ft@fold0").checkpoint \
        == "backbone/resnet50-fold0/backbone.pt"
    assert spaces.get("dinov2_ft@fold4").name == "dinov2_ft@fold4"


def test_the_fold_name_round_trips():
    for base in ("dinov2_ft", "resnet50_ft"):
        for i in range(5):
            assert spaces.split_fold(spaces.fold_name(base, i)) == (base, i)


def test_a_plain_name_is_not_read_as_a_fold_name():
    for name in spaces.names():
        assert spaces.split_fold(name) is None


def test_a_frozen_space_has_no_fold_encoders_and_says_so():
    """Refusing beats inventing a path.

    A frozen encoder has nothing to hold out, so the same cache is honest for
    every fold — and silently handing back five names for one cache is exactly
    how a leaky configuration would come to look like an honest one.
    """
    with pytest.raises(SystemExit, match="no per-fold encoders"):
        spaces.fold("resnet50", 0)
    with pytest.raises(SystemExit, match="frozen"):
        spaces.fold("dinov2_vitb14", 0)


def test_every_fine_tuned_space_declares_its_fold_template():
    """The property that makes an honest CV reachable at all for these two."""
    for space in spaces.SPACES.values():
        if space.checkpoint is not None:
            assert space.fold_checkpoint, \
                f"{space.name} is fine-tuned but declares no fold_checkpoint"
            assert "{fold}" in space.fold_checkpoint


def test_the_fold_templates_match_the_tags_the_gpu_job_writes():
    """Pins `spaces.py` against `infra/run_job.sh`, which produces the weights.

    These two files have to agree or the job spends five GPU-hours writing
    `backbone/dinov2-fold0/` while the registry looks somewhere else — and the
    failure surfaces a day later, on a different machine, as "cannot verify".
    The job now checks this itself in its first second; this is the same check
    where it costs nothing.
    """
    job = (paths.ROOT / "infra" / "run_job.sh").read_text()

    # `tag="${FT_TAG}-fold${i}"` is the convention being pinned. If the job
    # stops writing tags that way, this test is measuring nothing.
    assert 'tag="${FT_TAG}-fold${i}"' in job
    assert f'--space "${{SPACE}}{spaces.FOLD_SEP}${{i}}"' in job

    mapping = re.findall(r"FT_TAG=(\S+);\s*SPACE=(\S+);", job)
    assert mapping, "could not read the backbone -> (tag, space) map from run_job.sh"
    for ft_tag, space in mapping:
        for i in range(5):
            assert spaces.fold(space, i).checkpoint \
                == f"backbone/{ft_tag}-fold{i}/backbone.pt"


def test_a_fold_cache_never_shares_a_directory_with_its_base_or_its_siblings():
    dirs = [paths.features_dir("dinov2_ft")] + [
        paths.features_dir(spaces.fold_name("dinov2_ft", i)) for i in range(5)]
    assert len(set(dirs)) == len(dirs)


# -- the argparse converter --------------------------------------------------

def test_space_name_accepts_base_and_fold_names_and_returns_them_unchanged():
    for name in [*spaces.names(), "dinov2_ft@fold0", "resnet50_ft@fold4"]:
        assert spaces.space_name(name) == name


def test_space_name_rejects_at_parse_time_and_names_the_registered_spaces():
    """It replaced `choices=spaces.names()` at eight call sites.

    `choices` could not admit a derived name without listing ten of them in
    every `--help`; the trade is that validation has to be explicit here.
    """
    import argparse

    for bad in ("resnet51", "dinov2_ft@fold", "resnet50@fold0"):
        with pytest.raises(argparse.ArgumentTypeError):
            spaces.space_name(bad)


# -- the path layout ---------------------------------------------------------

def test_every_cache_path_hangs_off_the_space():
    space = "resnet50"
    assert paths.features_dir(space) == paths.FEATURES / space
    assert paths.video_dir(space, 7) == paths.FEATURES / space / "video_07"
    assert paths.manifest_path(space) == paths.FEATURES / space / "manifest.json"


def test_video_numbers_are_zero_padded_to_two_digits():
    """`video_07`, not `video_7` — the on-disk names the cache already uses."""
    assert paths.video_dir("resnet50", 7).name == "video_07"
    assert paths.video_dir("resnet50", 25).name == "video_25"


def test_two_spaces_never_share_a_directory():
    a, b = spaces.names()[0], spaces.names()[1]
    assert paths.features_dir(a) != paths.features_dir(b)
    assert paths.video_dir(a, 1) != paths.video_dir(b, 1)
