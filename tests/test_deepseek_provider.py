import json

import pytest

from meta_harness import providers
from meta_harness.chat_types import ToolSpec


class _Response:
    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def _model(response, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    seen = []

    def urlopen(request, timeout):
        seen.append(request)
        return _Response(response)

    model = providers.DeepSeekModel("deepseek-flash", api_key="sk-test-deepseek")
    model.client = providers._HttpClient(urlopen=urlopen)
    return model, seen


TEXT = {"choices": [{"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107,
                  "prompt_cache_hit_tokens": 64, "prompt_cache_miss_tokens": 36}}
TOOL = {"choices": [{"message": {"role": "assistant", "content": "", "reasoning_content": "think think",
                                 "tool_calls": [{"id": "call_1", "type": "function",
                                                 "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}}]},
                     "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20,
                  "completion_tokens_details": {"reasoning_tokens": 12},
                  "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 50}}
TOOLS = [ToolSpec("bash", "run", {"type": "object", "properties": {"cmd": {"type": "string"}}})]


def test_registered_everywhere(monkeypatch):
    assert "deepseek" in providers.PROVIDERS
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert isinstance(providers.model_from_environment("deepseek", "deepseek-flash"), providers.DeepSeekModel)


def test_defaults_and_base_override(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    assert providers.DeepSeekModel("m").base_url == "https://api.deepseek.com"
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://example.test/v1/")
    assert providers.DeepSeekModel("m").base_url == "https://example.test/v1"


def test_missing_key_names_variable(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="DEEPSEEK_API_KEY"):
        providers.DeepSeekModel("m")


def test_text_chat_and_cache_aware_usage(monkeypatch):
    model, seen = _model(TEXT, monkeypatch)
    turn = model.chat("sys", [{"role": "user", "content": "hi"}], [])
    assert turn.text == "hello" and turn.tool_calls == [] and turn.stop_reason == "end"
    request = seen[0]
    assert request.full_url == "https://api.deepseek.com/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-test-deepseek"
    u = turn.usage
    assert (u.input, u.output, u.cache_read) == (100, 7, 64)
    assert u.input_includes_cache is True
    assert u.context_tokens() == 100  # hit + miss, not double counted
    price = providers.Price(input=1.0, output=2.0, cache_read=0.1)
    # uncached 36 * 1 + 7 * 2 + 64 * 0.1 = 56.4 per million
    assert providers.cost_usd(u, price) == pytest.approx(56.4 / 1_000_000)


def test_tool_call_ignores_reasoning_content_and_counts_reasoning(monkeypatch):
    model, seen = _model(TOOL, monkeypatch)
    turn = model.chat("sys", [{"role": "user", "content": "go"}], TOOLS)
    assert [(c.id, c.name, c.arguments) for c in turn.tool_calls] == [("call_1", "bash", {"cmd": "ls"})]
    assert turn.text == "" and "think" not in turn.text and turn.stop_reason == "tool_use"
    assert turn.usage.reasoning == 12
    assert turn.usage.cache_read == 0 and turn.usage.input == 50
    body = json.loads(seen[0].data)
    assert body["tools"][0]["function"]["name"] == "bash"


def test_call_returns_text_and_token_counts(monkeypatch):
    model, _ = _model(TEXT, monkeypatch)
    assert model.call("hi") == ("hello", {"prompt_tokens": 100, "completion_tokens": 7})


def test_usage_without_deepseek_fields_falls_back(monkeypatch):
    plain = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 9, "completion_tokens": 1}}
    model, _ = _model(plain, monkeypatch)
    u = model.chat("", [{"role": "user", "content": "hi"}], []).usage
    assert (u.input, u.cache_read, u.input_includes_cache) == (9, 0, True)


def test_unpriced_by_default(monkeypatch):
    monkeypatch.delenv("META_HARNESS_PRICES", raising=False)
    assert providers.price_for("deepseek-flash") is None


def test_list_models(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)

    def fake_get(url, headers, timeout):
        assert url == "https://api.deepseek.com/models"
        assert headers["Authorization"] == "Bearer k"
        return {"data": [{"id": "deepseek-v4-pro"}, {"id": "deepseek-flash"}]}

    monkeypatch.setattr(providers, "_http_get_json", fake_get)
    assert providers.list_models("deepseek", api_key="k") == ["deepseek-flash", "deepseek-v4-pro"]


def test_input_derived_from_hit_plus_miss_when_prompt_tokens_absent(monkeypatch):
    only = {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 1, "prompt_cache_hit_tokens": 30, "prompt_cache_miss_tokens": 12}}
    model, _ = _model(only, monkeypatch)
    u = model.chat("", [{"role": "user", "content": "hi"}], []).usage
    assert (u.input, u.cache_read) == (42, 30)
