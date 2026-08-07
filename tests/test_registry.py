"""Tests pinning the model registry's "bare pitvis-train means train all" rule.

This exists because it silently broke once: `instruments` was registered and
reachable as `pitvis-train instruments`, but a hand-maintained DEFAULT_ORDER
meant the bare command quietly skipped it. The registry's whole point is that
adding a model is ONE edit, so the default set must be derived, not listed.
"""

from pitvis.training import registry


def test_bare_train_runs_every_registered_model():
    """The regression that motivated this file."""
    assert {m.name for m in registry.resolve(None)} == set(registry.REGISTRY)


def test_order_hint_comes_first_then_the_rest_by_name():
    order = registry.default_order()
    hint = [n for n in registry.ORDER_HINT if n in registry.REGISTRY]
    assert order[:len(hint)] == hint
    assert order[len(hint):] == sorted(set(registry.REGISTRY) - set(hint))


def test_order_hint_need_not_be_exhaustive():
    """A model absent from ORDER_HINT must still run — that is the fix."""
    extra = set(registry.REGISTRY) - set(registry.ORDER_HINT)
    assert extra, "expected at least one model outside ORDER_HINT to guard"
    assert extra <= {m.name for m in registry.resolve(None)}


def test_explicit_names_are_respected_in_the_order_given():
    assert [m.name for m in registry.resolve(["arst", "baseline"])] == ["arst", "baseline"]


def test_unknown_model_names_the_registered_ones():
    import pytest
    with pytest.raises(SystemExit, match="unknown model"):
        registry.get("nope")


def test_describe_lists_every_model():
    text = registry.describe()
    for name in registry.REGISTRY:
        assert name in text
