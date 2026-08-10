"""The case document, and the two model changes that feed it.

Nothing here needs the feature cache, a checkpoint or a real video: the case
fixture is synthesised in a tmp_path and the models are randomly initialised.
`data/`, `predictions/` and `26531686/` are all gitignored, so a test that
depended on them would pass here and fail on every clone.
"""

import json

import numpy as np
import pytest
import torch

from pitvis.app import case as C
from pitvis.app.catalogue import CaseRef

# --------------------------------------------------------------------------
# The (-1, -2) collision — the single highest-consequence ambiguity in the app


def test_the_same_sentinel_pair_means_different_things_by_source():
    """(-1, -2) is out-of-patient in truth and 'nothing cleared' in a prediction.

    `multihot_to_pairs` reuses the annotations' out-of-patient sentinel as
    padding for an all-zero prediction row, but SANO's head is 19 sigmoids with
    no out-of-patient class — it cannot express that state. Conflating them
    would report the scope as having left the patient for 26% of video_19.
    """
    assert C._instrument_state(-1, -2, truth=True) == "out_of_patient"
    assert C._instrument_state(-1, -2, truth=False) == "none"


def test_class_zero_is_a_real_instrument_not_a_sentinel():
    """Id 0 is 'no visible instrument / occluded' — a scored class at 31.5% of
    frames, and a different statement from either sentinel."""
    assert C._instrument_state(0, -2, truth=False) == "one"
    assert C._instrument_state(0, -2, truth=True) == "one"


def test_two_instruments():
    assert C._instrument_state(8, 16, truth=False) == "two"


# --------------------------------------------------------------------------
# Segments


def test_segments_are_a_lossless_run_length_encoding():
    raw = np.array([-1, -1, -1, 4, 4, 7, 7, 7, 7])
    segs = C._segments(raw, None, None)

    assert [s["step"] for s in segs] == [-1, 4, 7]
    assert [(s["start_s"], s["end_s"]) for s in segs] == [(0, 2), (3, 4), (5, 8)]
    # end_s is INCLUSIVE, matching inference/predict.py's `t + n - 1`. Timeline
    # geometry must use end_s + 1 as the right edge or every segment is short.
    assert all(s["duration_s"] == s["end_s"] - s["start_s"] + 1 for s in segs)
    assert sum(s["duration_s"] for s in segs) == len(raw)


def test_segment_confidence_aggregates_over_its_own_span():
    raw = np.array([1, 1, 2, 2])
    segs = C._segments(raw, [0.9, 0.5, 0.2, 0.4], [0, 0, 1, 0])
    assert segs[0]["confidence"] == {"mean": 0.7, "min": 0.5, "held_frac": 0.0}
    assert segs[1]["confidence"] == {"mean": 0.3, "min": 0.2, "held_frac": 0.5}


def test_lanes_collapse_a_class_into_inclusive_runs():
    s1 = np.array([16, 16, -1, 16, 8])
    s2 = np.array([-2, 8, -2, -2, -2])
    lanes = {v["id"]: v for v in C._lanes(s1, s2)}
    assert lanes[16]["intervals"] == [[0, 1], [3, 3]]
    assert lanes[8]["intervals"] == [[1, 1], [4, 4]]
    assert lanes[16]["seconds"] == 3


# --------------------------------------------------------------------------
# build_case, on a synthetic case


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    """A three-second case with predictions but no ground truth (video_19's shape)."""
    import pandas as pd

    from pitvis.app import media

    d = tmp_path / "predictions" / "case_x"
    d.mkdir(parents=True)
    pd.DataFrame({"int_time": [0, 1, 2], "int_step": [-1, 4, 4]}).to_csv(
        d / "predictions.csv", index=False)
    pd.DataFrame({"int_time": [0, 1, 2],
                  "int_instrument1": [-1, 16, 16],
                  "int_instrument2": [-2, -2, 8]}).to_csv(
        d / "instruments.csv", index=False)
    (d / "summary.json").write_text(json.dumps({
        "steps": {"checkpoint": "/x/citi.pt", "width": 5, "cci": True,
                  "mask_excluded": False},
        "instruments": {"checkpoint": "/x/sano.pt", "threshold": 0.5},
    }))

    probs = np.zeros((3, 15), np.float32)
    probs[0, 0] = 0.8; probs[0, 4] = 0.2       # emitted background, confident
    probs[1, 4] = 0.6; probs[1, 7] = 0.4
    probs[2, 7] = 0.7; probs[2, 4] = 0.3       # argmax says 7, CCI held 4
    np.save(d / "step_probs.npy", probs)

    iprobs = np.zeros((3, 19), np.float32)
    iprobs[0] = 0.1                             # nothing clears 0.5
    iprobs[1, 16] = 0.9
    iprobs[2, 16] = 0.9; iprobs[2, 8] = 0.7
    np.save(d / "instrument_probs.npy", iprobs)

    video = tmp_path / "case_x.mp4"
    video.write_bytes(b"\0" * 64)
    monkeypatch.setattr(C, "PREDICTIONS", tmp_path / "predictions")
    monkeypatch.setattr(media, "probe", lambda v: {
        "bytes": 64, "duration_s": 3.4, "width": 1280, "height": 720,
        "fps": 24.0, "faststart": False})

    return CaseRef(case_id="case_x", video=video, bytes=64, seconds=3,
                   features_cached=False, truth=None,
                   prediction={"available": True, "stale": False,
                               "computed_at": "2026-08-07T00:00:00Z"})


def test_case_document_is_json_serialisable(synthetic):
    """Guards the numpy leak: evaluate()'s `pooled` holds ndarrays that
    json.dumps raises on, which is why build_case uses evaluate_video."""
    doc = C.build_case(synthetic)
    assert json.loads(json.dumps(doc)) == doc


# --------------------------------------------------------------------------
# Provenance — which model produced what is on screen


def test_model_provenance_reaches_the_wire(synthetic, tmp_path):
    """There are four checkpoint families and three feature spaces, and every
    v2 checkpoint is named `model.pt` — so the filename alone cannot say what
    produced a number. summary.json has carried these tags since the variant
    work; case.py was dropping them."""
    d = tmp_path / "predictions" / "case_x"
    summary = json.loads((d / "summary.json").read_text())
    summary["steps"] |= {"model": "arst-v2", "variant": "best",
                         "space": "dinov2_vitb14"}
    summary["instruments"] |= {"variant": "weighted", "space": "dinov2_vitb14",
                               "classes_predicted": 11}
    (d / "summary.json").write_text(json.dumps(summary))

    doc = C.build_case(synthetic)
    task1 = doc["prediction"]["model"]["task1"]
    assert (task1["name"], task1["variant"], task1["space"]) \
        == ("arst-v2", "best", "dinov2_vitb14")
    assert task1["width"] == 5 and task1["cci"] is True   # the old keys survive

    inst = doc["instruments"]
    assert (inst["variant"], inst["space"], inst["classes_predicted"]) \
        == ("weighted", "dinov2_vitb14", 11)


def test_provenance_is_absent_not_invented_on_older_predictions(synthetic):
    """The common case, not the edge one: nothing predicted before the variant
    work carries these tags, so the wire must say None and let the UI report
    that rather than render a confident blank."""
    doc = C.build_case(synthetic)          # the fixture's summary has no tags
    task1 = doc["prediction"]["model"]["task1"]
    assert task1["name"] is None
    assert task1["variant"] is None
    assert task1["space"] is None
    # ...but what IS known is still reported.
    assert task1["checkpoint"] == "citi.pt"
    assert doc["instruments"]["variant"] is None
    assert doc["instruments"]["checkpoint"] == "sano.pt"


def test_every_per_second_array_has_exactly_one_entry_per_second(synthetic):
    doc = C.build_case(synthetic)
    n = doc["video"]["seconds"]
    assert n == 3
    for block in (doc["prediction"]["per_second"],
                  doc["instruments"]["per_second"]):
        assert {k: len(v) for k, v in block.items()} == {k: n for k in block}


def test_confidence_is_the_emitted_label_not_the_argmax(synthetic):
    """At t=2 the decoder preferred step 7 but CCI held step 4.

    Confidence must report p(4) = 0.3, the label actually shown — not the 0.7
    the model gave a step it was overruled on. Reading low there is the point.
    """
    ps = C.build_case(synthetic)["prediction"]["per_second"]
    assert ps["step"][2] == 4
    assert ps["top1_step"][2] == 7
    assert ps["confidence"][2] == pytest.approx(0.3)
    assert ps["top1_prob"][2] == pytest.approx(0.7)
    assert ps["cci_held"] == [0, 0, 1]


def test_background_round_trips_between_encodings(synthetic):
    """top1_step is emitted RAW, so encoded class 0 must come back as -1."""
    ps = C.build_case(synthetic)["prediction"]["per_second"]
    assert ps["step"][0] == -1 and ps["top1_step"][0] == -1


def test_a_prediction_below_threshold_is_not_out_of_patient(synthetic):
    inst = C.build_case(synthetic)["instruments"]["per_second"]
    assert inst["state"] == ["none", "one", "two"]
    assert inst["slot1"][0] is None            # never the raw -1
    assert inst["max_prob"][0] == pytest.approx(0.1)   # the runner-up survives


def test_missing_ground_truth_is_explained_not_silently_empty(synthetic):
    doc = C.build_case(synthetic)
    assert doc["truth"]["available"] is False
    assert "gap in the PitVis download" in doc["truth"]["reason"]
    assert doc["scores"]["available"] is False


def test_seams_are_present_and_empty_from_v1(synthetic):
    """The renderer branches on contents, never on a missing key."""
    doc = C.build_case(synthetic)
    assert doc["corrections"] == {"available": False, "edits": []}
    assert doc["explanations"] == {"available": False, "segments": []}
    assert doc["live"] is None


def test_unpredicted_case_raises_rather_than_returning_a_hollow_document(
        synthetic, tmp_path):
    empty = CaseRef(case_id="case_y", video=tmp_path / "y.mp4", bytes=0,
                    seconds=None, features_cached=False, truth=None)
    with pytest.raises(FileNotFoundError):
        C.build_case(empty)


# --------------------------------------------------------------------------
# The model changes: probabilities must be purely additive


def _tiny_arst():
    from pitvis.models.arst import ARST
    torch.manual_seed(0)
    m = ARST(num_classes=15, width=3)
    m.eval()
    return m


def test_step_probabilities_do_not_perturb_the_prediction():
    """cci_decode's rollout is stateful — each step feeds the next — so binding
    the logits to a variable could in principle change what is decoded. It must
    not: the artifact is additive or it is worthless."""
    from types import SimpleNamespace

    from pitvis.models.arst import D_MODEL
    from pitvis.training.arst import cci_decode

    m = _tiny_arst()
    f = torch.randn(1, 24, D_MODEL)
    opts = SimpleNamespace(chunk=8, cci=True, mask_excluded=False)
    dev = torch.device("cpu")

    plain = cci_decode(m, f, opts, dev)
    preds, probs = cci_decode(m, f, opts, dev, return_probs=True)

    assert np.array_equal(plain, preds)
    assert probs.shape == (24, 15) and probs.dtype == np.float32
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5)


def test_step_probabilities_are_pre_cci_by_construction():
    """With CCI on, the recorded argmax may disagree with the emitted label;
    with CCI off it never can. That difference IS the held signal."""
    from types import SimpleNamespace

    from pitvis.models.arst import D_MODEL
    from pitvis.training.arst import cci_decode

    m = _tiny_arst()
    f = torch.randn(1, 40, D_MODEL)
    dev = torch.device("cpu")

    preds, probs = cci_decode(
        m, f, SimpleNamespace(chunk=16, cci=False, mask_excluded=False),
        dev, return_probs=True)
    assert np.array_equal(probs.argmax(1), preds)


def test_instrument_probabilities_do_not_perturb_the_prediction():
    from pitvis.models.lstm import SanoLSTM
    from pitvis.training.instruments import predict_video

    torch.manual_seed(0)
    m = SanoLSTM(in_dim=32, hidden=16, layers=1, window=5, dropout=0.0,
                 aux_step=False)
    m.eval()
    feats = torch.randn(30, 32)
    dev = torch.device("cpu")

    plain = predict_video(m, feats, 0.5, 8, dev)
    pairs, probs, keep = predict_video(m, feats, 0.5, 8, dev, return_probs=True)

    assert np.array_equal(plain, pairs)
    assert probs.shape == (30, 19) and keep.shape == (30, 19)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_the_mask_distinguishes_below_threshold_from_out_of_patient():
    """An all-zero `keep` row is what the pairs alone cannot tell you."""
    from pitvis.models.lstm import SanoLSTM
    from pitvis.training.instruments import predict_video

    torch.manual_seed(0)
    m = SanoLSTM(in_dim=32, hidden=16, layers=1, window=5, dropout=0.0,
                 aux_step=False)
    m.eval()
    feats = torch.randn(20, 32)
    # A threshold of 1.0 is unreachable, so every row must fall through to the
    # padding pair — and `keep` must still say plainly that nothing was chosen.
    pairs, _, keep = predict_video(m, feats, 1.0, 8, torch.device("cpu"),
                                   return_probs=True)
    assert (keep.sum(1) == 0).all()
    assert (pairs == np.array([-1, -2])).all()
    assert all(C._instrument_state(a, b, truth=False) == "none"
               for a, b in pairs)


# --------------------------------------------------------------------------
# Packaging


def test_app_assets_ship_inside_the_package():
    """Resolved through importlib.resources, i.e. from wherever the package is
    actually installed.

    This is the first non-.py content in src/pitvis/, and an editable install
    hides a packaging failure completely — the same shape as the .gitignore bug
    in 5473f86, which worked locally and broke on every clone.
    """
    from importlib.resources import files

    assets = files("pitvis.app") / "assets"
    assert (assets / "index.html").is_file()
    assert (assets / "app.css").is_file()
    assert (assets / "js" / "main.js").is_file()


def test_paths_anchors_app_assets_on_the_package_not_the_repo():
    """ROOT walks out of the package and does not exist in a wheel."""
    from pitvis.paths import APP_ASSETS, PACKAGE

    assert APP_ASSETS.is_relative_to(PACKAGE)
