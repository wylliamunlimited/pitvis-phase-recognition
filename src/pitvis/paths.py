"""Filesystem anchors, defined once.

Every path in the project derives from `ROOT`. Previously each script computed
`Path(__file__).resolve().parent.parent` independently, which silently encoded
"I live exactly one level below the repo root" into five separate files — a
constraint that breaks the moment anything moves.

`ROOT` is resolved from this module's location: `src/pitvis/paths.py` is three
levels below the repo root.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# Raw PitVis download — read-only input, never tracked.
RAW = ROOT / "26531686"

# Derived artifacts — all gitignored.
DATA = ROOT / "data"
FEATURES = DATA / "features"
MANIFEST = FEATURES / "manifest.json"
CKPT = DATA / "arst"                 # CITI/ARST — task 1
CKPT_INSTRUMENTS = DATA / "instruments"   # SANO — task 2

# Generated documentation.
NOTES = ROOT / "notes"
