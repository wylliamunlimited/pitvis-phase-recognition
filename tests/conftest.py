"""What the run actually collected, made available to a test.

There is exactly one thing here, and it exists because a test count is the one
fact about this suite that no file can state without immediately being able to
go stale. `README.md` said 136, then 151, then 203, each correct for a few
hours; the architecture atlas said 136 for six weeks. The count has one real
owner — collection — and it is not a value any document can hold.

`pytest_collection_modifyitems` is the only moment it exists. The hook runs
once, after every item is known and before any of them runs, so a test can be
handed the live number rather than a copy of it.
"""

from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent


def pytest_collection_modifyitems(session, config, items):
    per_file: dict[str, int] = {}
    for item in items:
        per_file[item.path.stem] = per_file.get(item.path.stem, 0) + 1
    config.stash[COLLECTED] = (len(items), per_file)


COLLECTED = pytest.StashKey[tuple[int, dict[str, int]]]()


@pytest.fixture(scope="session")
def collected(pytestconfig):
    """(total, {module_stem: count}) for this run — or a skip if it is partial.

    Running one file is a normal thing to do, and the count is meaningless
    then. A partial run skips rather than failing: a subset is not a wrong
    document, and a test that cried wolf on `pytest tests/test_eval.py` would
    be turned off inside a week.
    """
    total, per_file = pytestconfig.stash[COLLECTED]
    on_disk = {p.stem for p in TESTS.glob("test_*.py")}
    if per_file.keys() != on_disk:
        missing = ", ".join(sorted(on_disk - per_file.keys()))
        pytest.skip(f"partial run — {missing} not collected; counts are only "
                    f"meaningful for the whole suite")
    return total, per_file
