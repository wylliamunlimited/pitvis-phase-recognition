"""Filesystem anchors, defined once.

Every path in the project derives from `ROOT`. Previously each script computed
`Path(__file__).resolve().parent.parent` independently, which silently encoded
"I live exactly one level below the repo root" into five separate files — a
constraint that breaks the moment anything moves.

`ROOT` is resolved from this module's location: `src/pitvis/paths.py` is three
levels below the repo root.

Two anchors, not one. `ROOT` locates the *repo*; `PACKAGE` locates the
*installed package*. They coincide under the editable install we develop with
and diverge completely in a built wheel, where `ROOT` points three levels above
`site-packages/pitvis/` at something that does not exist. Anything shipped
inside the package — the app's assets — must hang off `PACKAGE`.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PACKAGE = Path(__file__).resolve().parent

# Raw PitVis download — read-only input, never tracked.
RAW = ROOT / "26531686"

# Derived artifacts — all gitignored.
DATA = ROOT / "data"
FEATURES = DATA / "features"         # one subdirectory per feature space
# Legacy single-space manifest. Superseded by `manifest_path(space)` below and
# removed once every reader takes a space; still exported so this commit
# changes no behaviour.
MANIFEST = FEATURES / "manifest.json"
CKPT = DATA / "arst"                 # CITI/ARST — task 1
CKPT_INSTRUMENTS = DATA / "instruments"   # SANO — task 2
PREDICTIONS = ROOT / "predictions"   # pitvis-predict output, one dir per video


# -- feature cache ---------------------------------------------------------
#
# The cache is keyed by feature space, so a second backbone can be extracted
# without destroying the first. These three functions are the only place the
# layout is spelled out; `video_NN` used to be formatted by hand in six modules.
#
# No default argument here on purpose. The default space lives in
# `pitvis.data.spaces`, and `paths` must not import `data` — that would make
# the import graph cyclic, since `data` imports `paths`.


def features_dir(space: str) -> Path:
    """Root of one feature space's cache."""
    return FEATURES / space


def video_dir(space: str, vid: int) -> Path:
    """Where one video's features/labels/instruments live within a space."""
    return features_dir(space) / f"video_{vid:02d}"


def manifest_path(space: str) -> Path:
    """One manifest per space, beside that space's video directories.

    Deliberately not one global manifest: the existing manifest carries exactly
    one `space` dict and `load_manifest` guards on full-dict equality against
    it. Per-space files keep that logic untouched — only the path changes.
    """
    return features_dir(space) / "manifest.json"

# Generated documentation.
NOTES = ROOT / "notes"

# Shipped inside the package, so anchored on PACKAGE and never on ROOT.
APP_ASSETS = PACKAGE / "app" / "assets"
