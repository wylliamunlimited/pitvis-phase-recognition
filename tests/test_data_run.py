"""Pins that the data workflow hands every stage the flags it was given.

`data/run.py` translates its own flags into an argv per stage. That translation
is where a workflow-wide flag can silently reach one stage and not another: the
runner used to append `--space` to the extract stage only, so
`pitvis-data --space dinov2_ft` extracted 25 videos into `dinov2_ft` and then
verified `resnet50` — reporting OK for a cache nothing had checked, or dying on
"manifest.json missing" after a 25-minute extraction that had succeeded.

The stages are replaced with recorders here, so this runs in milliseconds and
needs no feature cache, no ffmpeg and no raw video.
"""

import pytest

from pitvis.data import run as data_run


@pytest.fixture
def recorded(monkeypatch):
    """Run the workflow with every stage replaced by an argv recorder."""
    seen: dict[str, list[str]] = {}

    def record(name):
        def main(argv=None):
            seen[name] = list(argv or [])
            return 0
        return main

    monkeypatch.setattr(data_run.inventory, "main", record("inventory"))
    monkeypatch.setattr(data_run.extract_features, "main", record("extract"))
    monkeypatch.setattr(data_run.verify_cache, "main", record("verify"))
    return lambda argv: (data_run.main(argv), seen)[1]


def test_space_reaches_extract_and_verify(recorded):
    seen = recorded(["--space", "dinov2_vitb14"])
    assert "--space" in seen["extract"]
    assert seen["extract"][seen["extract"].index("--space") + 1] == "dinov2_vitb14"
    assert "--space" in seen["verify"], (
        "verify was not told which space to check — it would verify the default "
        "cache while extract wrote a different one"
    )
    assert seen["verify"][seen["verify"].index("--space") + 1] == "dinov2_vitb14"


def test_no_space_leaves_both_stages_on_their_own_default(recorded):
    seen = recorded([])
    assert "--space" not in seen["extract"]
    assert "--space" not in seen["verify"]


def test_probe_reaches_verify_only(recorded):
    seen = recorded(["--probe"])
    assert seen["verify"] == ["--probe"]
    assert "--probe" not in seen["extract"]


def test_space_and_probe_compose(recorded):
    seen = recorded(["--space", "dinov2_ft", "--probe"])
    assert seen["verify"] == ["--space", "dinov2_ft", "--probe"]


def test_videos_and_device_go_to_extract_only(recorded):
    seen = recorded(["--videos", "1", "2", "--device", "cpu"])
    assert seen["extract"] == ["1", "2", "--device", "cpu"]
    assert seen["verify"] == []


def test_an_unregistered_space_is_refused_by_the_runner(recorded):
    """The runner used to declare --space without choices, so a typo passed it
    and failed inside the stage instead."""
    with pytest.raises(SystemExit):
        recorded(["--space", "resnet51"])
