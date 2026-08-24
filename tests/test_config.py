import pytest

from localslm.config import Config, ConfigError

DATA = {
    "ollama": {"host": "http://x", "keep_alive": "5m"},
    "bench": {"repeats": 3},
    "presets": {"quick": {"bench.repeats": 1, "bench.suites": ["a"]}},
}


def test_dotted_lookup():
    cfg = Config(DATA)
    assert cfg["ollama.host"] == "http://x"
    assert cfg.get("missing.key", "fallback") == "fallback"
    with pytest.raises(ConfigError):
        _ = cfg["nope.nothing"]


def test_preset_overlays_dotted_paths():
    cfg = Config(DATA).with_preset("quick")
    assert cfg["bench.repeats"] == 1
    assert cfg["bench.suites"] == ["a"]
    # The base config is untouched — presets never mutate in place.
    assert Config(DATA)["bench.repeats"] == 3


def test_unknown_preset_lists_known_ones():
    with pytest.raises(ConfigError, match="quick"):
        Config(DATA).with_preset("nope")


def test_fingerprint_ignores_preset_catalogue():
    """Adding a preset you didn't run must not invalidate a run's config hash."""
    a = Config(DATA)
    extended = {**DATA, "presets": {**DATA["presets"], "other": {"bench.repeats": 9}}}
    assert a.fingerprint() == Config(extended).fingerprint()


def test_fingerprint_changes_with_real_config():
    a = Config(DATA)
    b = a.with_overrides({"bench.repeats": 5})
    assert a.fingerprint() != b.fingerprint()
