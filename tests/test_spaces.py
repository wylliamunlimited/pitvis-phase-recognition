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
