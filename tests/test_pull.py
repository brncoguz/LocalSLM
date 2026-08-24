"""Pull progress aggregation and error wording.

Both are pure functions precisely so they can be checked without touching the
network — the aggregation bug they guard against (a bar that restarts on every
layer) is only visible on a multi-layer model.
"""

from localslm.api.routes.models import aggregate_layers, humanise_pull_error


def test_aggregate_sums_across_layers():
    layers = {"a": (100, 100), "b": (50, 200), "c": (0, 700)}
    assert aggregate_layers(layers) == (150, 1000)


def test_aggregate_is_monotonic_as_layers_appear():
    """A new layer must never make the completed count go backwards."""
    layers: dict[str, tuple[int, int]] = {}
    seen: list[int] = []
    for digest, done, total in [("a", 500, 500), ("b", 100, 900), ("b", 900, 900)]:
        layers[digest] = (done, total)
        seen.append(aggregate_layers(layers)[0])
    assert seen == sorted(seen), seen


def test_aggregate_of_nothing_is_zero():
    assert aggregate_layers({}) == (0, 0)


def test_missing_model_gets_actionable_wording():
    assert "registry" in humanise_pull_error("pull model manifest: file does not exist")


def test_network_failure_is_distinguished():
    assert "network" in humanise_pull_error("dial tcp: lookup registry.ollama.ai").lower()


def test_unknown_errors_pass_through_unchanged():
    assert humanise_pull_error("something odd") == "something odd"
