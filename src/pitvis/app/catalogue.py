"""What cases exist, and what state each one is in.

Deliberately cheap. This runs on every catalogue request and must not parse a
CSV, load a feature array or probe a video — the expensive per-case work lives
in `case.py` and happens only when one case is actually opened. Everything here
is a `stat()` or a lookup in the feature manifest, which is read once.

The three states a case can be in, and why the distinction matters to the UI:

- **predicted** — `predictions/<id>/` exists. Open it, seconds.
- **cached, unpredicted** — features exist, so inference is a ~45 s cache hit
  the app can offer to run for you.
- **uncached** — a full 1 fps decode of a multi-gigabyte file, 10-25 minutes.
  The app refuses and prints the command instead; silently starting that from a
  click would be a hostile surprise.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pitvis.data.dataset import TRAIN, VAL
from pitvis.data import spaces
from pitvis.paths import CKPT, CKPT_INSTRUMENTS, PREDICTIONS, RAW, manifest_path

CASE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# Extra videos registered by `pitvis-app --video`, outside the RAW download.
_extra: dict[str, Path] = {}


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    video: Path
    bytes: int
    seconds: int | None          # None until something authoritative says
    features_cached: bool
    truth: Path | None
    prediction: dict = field(default_factory=dict)

    @property
    def split(self) -> str | None:
        n = _video_number(self.case_id)
        if n is None:
            return None
        return "val" if n in VAL else "train" if n in TRAIN else None

    def to_json(self) -> dict:
        return {
            "case_id": self.case_id,
            "video": str(self.video),
            "bytes": self.bytes,
            "seconds": self.seconds,
            "features_cached": self.features_cached,
            "has_truth": self.truth is not None,
            "split": self.split,
            "prediction": self.prediction,
        }


def register(video: Path) -> str:
    """Make a video outside `26531686/` addressable. Returns its case id."""
    video = video.resolve()
    _extra[video.stem] = video
    return video.stem


def _video_number(case_id: str) -> int | None:
    m = re.fullmatch(r"video_(\d+)", case_id)
    return int(m.group(1)) if m else None


def _manifest() -> dict:
    mpath = manifest_path(spaces.DEFAULT)
    if not mpath.exists():
        return {}
    try:
        return json.loads(mpath.read_text()).get("videos", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _checkpoint_mtime() -> float:
    """Newest mtime across every task checkpoint, or 0 if none exists.

    Includes the task-2 variants under v2/. A prediction made before a better
    instrument model was trained is stale in exactly the same way it is stale
    after retraining SANO, and the staleness chip exists to say so.
    """
    candidates = [CKPT / "citi.pt", CKPT_INSTRUMENTS / "sano.pt"]
    v2 = CKPT_INSTRUMENTS / "v2"
    if v2.exists():
        candidates += sorted(v2.glob("*/model.pt"))
    return max((p.stat().st_mtime for p in candidates if p.exists()), default=0.0)


def _prediction_state(case_id: str) -> dict:
    d = PREDICTIONS / case_id
    summary = d / "summary.json"
    if not summary.exists():
        return {"available": False}
    mtime = summary.stat().st_mtime
    return {
        "available": True,
        "dir": str(d),
        "computed_at": datetime.fromtimestamp(mtime, UTC).isoformat(
            timespec="seconds").replace("+00:00", "Z"),
        "has_probs": (d / "step_probs.npy").exists(),
        # A checkpoint newer than the prediction means the numbers on screen
        # came from a model that no longer exists on disk. Not an error — but
        # the UI must offer a re-run rather than present them as current.
        "stale": _checkpoint_mtime() > mtime,
    }


def cases() -> dict[str, CaseRef]:
    """Every addressable case, keyed by id, in id order."""
    manifest = _manifest()
    videos: dict[str, Path] = {p.stem: p for p in sorted(RAW.glob("video_*.mp4"))}
    videos.update(_extra)

    out: dict[str, CaseRef] = {}
    for case_id, video in videos.items():
        if not video.exists():
            continue
        entry = manifest.get(case_id)
        cached = bool(entry) and Path(entry["source"]).resolve() == video.resolve()
        # Derived from the stem, not reformatted from the int: the files are
        # zero-padded (annotations_01.csv), so f"annotations_{n}.csv" silently
        # finds nothing for videos 1-9.
        truth = (RAW / (case_id.replace("video_", "annotations_") + ".csv")
                 if _video_number(case_id) is not None else None)
        out[case_id] = CaseRef(
            case_id=case_id,
            video=video,
            bytes=video.stat().st_size,
            seconds=entry["frames"] if cached else None,
            features_cached=cached,
            truth=truth if truth and truth.exists() else None,
            prediction=_prediction_state(case_id),
        )
    return dict(sorted(out.items()))


def get(case_id: str) -> CaseRef | None:
    """One case, or None. Membership here is the only authority for an id.

    Callers must not build a path from `case_id` — it arrives from a URL. The
    regex is a cheap first gate; the real one is that the returned `video` came
    from a directory listing, never from string concatenation.
    """
    if not CASE_ID.match(case_id):
        return None
    return cases().get(case_id)
