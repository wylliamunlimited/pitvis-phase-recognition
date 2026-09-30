"""Every test count printed in the docs must be the count this run collected.

Companion to `test_doc_pointers.py`, and the same failure it guards against one
level up: a `file.py:NN` pointer rots when the file grows underneath it, and a
test count rots when the suite does. This one had already happened three times
in six weeks — `README.md` moved 136 -> 151 -> 203 while the architecture atlas
sat at 136, and 151 was stale within hours of being written, superseded by the
52 pointer tests added later in the same sweep.

The count's owner is collection, not prose. So nothing here hardcodes a number:
the run reports what it collected and the documents are checked against it.

Two shapes are pinned, because the docs state counts in two ways:

1. **Suite counts** — a number next to `uv run pytest`. This is the one that
   drifted.
2. **Per-file counts** — `tests/test_eval.py ... (24 tests)`, as `roadmap.md`
   writes it. Rarer, and rots the same way when a file gains a case.

A number that is neither is left alone, which matters more than it sounds:
`step-variants.md` contains the phrase "§7 tests it", and a looser regex reads
that as a claim that seven tests exist.
"""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# "uv run pytest   # 206 tests", including across the atlas's </td><td>.
SUITE = re.compile(r"uv run pytest\b[^\n]*?\b(\d[\d,]*) tests\b")

# "`tests/test_eval.py`\n(24 tests, all passing)" — the file, then its count,
# within a short window so an unrelated later number cannot be captured.
PER_FILE = re.compile(r"tests/(test_\w+)\.py.{0,160}?\((\d[\d,]*) tests", re.S)

DOCS = sorted(
    [*ROOT.glob("*.md"), *ROOT.glob("notes/**/*.md"), *ROOT.glob("infra/*.md"),
     *ROOT.glob("notes/**/*.html")]
)


def claims():
    """Every (doc, line, kind, subject, stated) the docs make about counts."""
    for doc in DOCS:
        text = doc.read_text()
        for pat, kind in ((SUITE, "suite"), (PER_FILE, "file")):
            for m in pat.finditer(text):
                groups = m.groups()
                subject, stated = (None, groups[0]) if kind == "suite" else groups
                line = text.count("\n", 0, m.start()) + 1
                yield (doc.relative_to(ROOT).as_posix(), line, kind, subject,
                       int(stated.replace(",", "")))


CLAIMS = list(claims())


def test_the_docs_state_at_least_one_count():
    """Guards the regexes themselves.

    If both patterns stopped matching — a reworded README, an HTML change —
    every assertion below would vacuously pass and the pin would be silently
    off. This is the canary for that.
    """
    assert CLAIMS, "no test count found in any doc — have the patterns drifted?"


@pytest.mark.parametrize(
    "doc,line,kind,subject,stated",
    CLAIMS,
    ids=[f"{d}:{n}->{s or 'suite'}" for d, n, _, s, _ in CLAIMS],
)
def test_stated_count_matches_what_was_collected(doc, line, kind, subject,
                                                 stated, collected):
    total, per_file = collected

    if kind == "suite":
        assert stated == total, (
            f"{doc}:{line} says {stated} tests; this run collected {total}. "
            f"The suite is the owner — update the document."
        )
        return

    assert subject in per_file, (
        f"{doc}:{line} cites tests/{subject}.py, which collected nothing "
        f"(renamed or deleted?)"
    )
    assert stated == per_file[subject], (
        f"{doc}:{line} says tests/{subject}.py has {stated} tests; it "
        f"collected {per_file[subject]}."
    )
