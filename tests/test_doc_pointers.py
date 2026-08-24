"""Every `file.py:NN` pointer in the docs must point at what the prose says.

`walkthrough.md` opens by promising "Every `file.py:NN` reference below is a
real line — follow them", and that promise decayed silently: the package
restructure renamed `train_baseline.py`, and `extract_features.py` grew from
~130 lines to 469 when multi-space support landed. Roughly fifteen pointers
ended up on blank lines, on closing brackets, or several hundred lines away
from the function they named — while still resolving to a real file at an
in-range line, which is all a link checker looks at.

So this checks the three things that actually rot, in increasing strength:

1. the file exists,
2. the span is inside it and does not start on a blank line or a bare
   delimiter (the classic symptom of a file growing underneath a pointer),
3. if the prose names a function next to the pointer — `probe()`,
   `load_annotations()` — then `def <name>` appears inside the span.

(3) is the one that catches real drift. It is deliberately scoped to
backticked `name()` forms so that ordinary prose never trips it.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# `data/inventory.py:100-105`, `training/arst.py:319`, `dataset.py:16`
POINTER = re.compile(r"`([\w/]+\.py):(\d+)(?:-(\d+))?`")

# A function named in backticks on the same line: `probe()`, `flush()`.
FUNC = re.compile(r"`([A-Za-z_]\w*)\(\)`")

# Lines that carry no identity — a pointer landing here has almost certainly
# drifted, even though the line is technically "real".
DELIMITERS = {")", "(", "}", "{", "]", "[", '"""', "'''", ",", ":"}


def docs() -> list[Path]:
    found = [p for p in (ROOT / "notes").rglob("*.md")]
    found += [ROOT / "README.md", ROOT / "CLAUDE.md"]
    found += list((ROOT / "infra").glob("*.md"))
    return sorted(p for p in found if p.exists())


def resolve(ref: str) -> Path | None:
    """Docs cite paths at several depths: `dataset.py`, `data/dataset.py`,
    `src/pitvis/data/dataset.py`. All three mean the same file."""
    for cand in (ROOT / ref, ROOT / "src" / "pitvis" / ref):
        if cand.is_file():
            return cand
    matches = [p for p in (ROOT / "src").rglob(Path(ref).name)] + [
        p for p in (ROOT / "tests").rglob(Path(ref).name)
    ]
    # Prefer a match whose tail agrees with the cited path.
    for p in matches:
        if str(p).endswith(ref):
            return p
    return matches[0] if len(matches) == 1 else None


def pointers() -> list[tuple[str, int, str, int, int | None, str]]:
    out = []
    for doc in docs():
        for lineno, line in enumerate(doc.read_text().split("\n"), 1):
            for m in POINTER.finditer(line):
                ref, start, end = m.group(1), int(m.group(2)), m.group(3)
                rel = doc.relative_to(ROOT).as_posix()
                out.append((rel, lineno, ref, start, int(end) if end else None, line))
    return out


ALL = pointers()


def test_docs_actually_contain_pointers():
    """Guards the regex: if it silently stops matching, every other test in
    this file passes vacuously."""
    assert len(ALL) > 30, f"only found {len(ALL)} pointers — regex broken?"


@pytest.mark.parametrize(
    "doc,lineno,ref,start,end,line",
    ALL,
    ids=[f"{d}:{n}->{r}:{s}" for d, n, r, s, _, _ in ALL],
)
def test_pointer(doc, lineno, ref, start, end, line):
    target = resolve(ref)
    assert target is not None, f"{doc}:{lineno} cites {ref}, which does not exist"

    src = target.read_text().split("\n")
    last = end or start
    assert start >= 1, f"{doc}:{lineno} cites {ref}:{start} — line numbers start at 1"
    assert last <= len(src), (
        f"{doc}:{lineno} cites {ref}:{start}"
        + (f"-{end}" if end else "")
        + f", but {target.relative_to(ROOT)} has only {len(src)} lines"
    )

    first = src[start - 1].strip()
    assert first, (
        f"{doc}:{lineno} cites {ref}:{start}, which is a BLANK line — "
        "the pointer has drifted"
    )
    assert first not in DELIMITERS, (
        f"{doc}:{lineno} cites {ref}:{start}, which is just {first!r} — "
        "the pointer has drifted"
    )

    span = "\n".join(src[start - 1 : last])
    for name in FUNC.findall(line):
        if f"def {name}" in "\n".join(src):  # only check names the file defines
            assert f"def {name}" in span, (
                f"{doc}:{lineno} says `{name}()` at {ref}:{start}"
                + (f"-{end}" if end else "")
                + f", but `def {name}` is at line "
                f"{next(i for i, s in enumerate(src, 1) if f'def {name}' in s)}"
            )
