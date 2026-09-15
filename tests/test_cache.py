from pathlib import Path

from meta_harness.cache import CachedModel, DiskCache, cache_key


class _CountingModel:
    model = "test-model"

    def __init__(self):
        self.calls = 0

    def call(self, prompt, **kwargs):
        self.calls += 1
        return f"reply:{prompt}", {"prompt_tokens": 5, "completion_tokens": 1}

    def __call__(self, prompt, **kwargs):
        return self.call(prompt, **kwargs)[0]


def test_cache_key_is_stable_and_order_independent():
    assert cache_key("m", "p", {"a": 1, "b": 2}) == cache_key("m", "p", {"b": 2, "a": 1})
    assert cache_key("m", "p", {}) != cache_key("m", "q", {})


def test_disk_cache_round_trips(tmp_path: Path):
    cache = DiskCache(tmp_path / "cache")
    assert cache.get("k") is None
    cache.put("k", {"text": "v", "usage": {}})
    assert cache.get("k")["text"] == "v"


def test_cached_model_avoids_second_upstream_call(tmp_path: Path):
    upstream = _CountingModel()
    model = CachedModel(upstream, tmp_path / "cache")
    assert model("hello") == "reply:hello"
    assert model("hello") == "reply:hello"
    assert upstream.calls == 1
    assert (model.hits, model.misses) == (1, 1)


def test_cached_model_preserves_usage(tmp_path: Path):
    model = CachedModel(_CountingModel(), tmp_path / "cache")
    model.call("hello")
    _, usage = model.call("hello")
    assert usage["prompt_tokens"] == 5


def test_cached_model_can_be_disabled(tmp_path: Path):
    upstream = _CountingModel()
    model = CachedModel(upstream, tmp_path / "cache", enabled=False)
    model("hello")
    model("hello")
    assert upstream.calls == 2


def test_cached_model_survives_plain_callables(tmp_path: Path):
    model = CachedModel(lambda prompt, **_: "flat", tmp_path / "cache")
    text, usage = model.call("x")
    assert text == "flat"
    assert usage == {}
