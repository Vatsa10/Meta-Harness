import email.utils
import io
import json
import random
import time
import urllib.error
from pathlib import Path

import pytest

from meta_harness import embeddings, providers
from meta_harness.providers import ToolCall, ToolSpec, Usage

FIX = Path(__file__).parent / "fixtures" / "providers"
READ = ToolSpec("read", "read a file", {"type": "object", "properties": {"path": {"type": "string"}},
                                         "required": ["path"]})


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_parse_anthropic_mixed_text_and_tools():
    turn = providers.parse_anthropic(load("anthropic_tool_use.json"))
    assert turn.text == "Reading the frontier first."
    assert [c.name for c in turn.tool_calls] == ["read", "list"]
    assert turn.tool_calls[0].arguments == {"path": "frontier.json"}
    assert (turn.usage.input, turn.usage.output, turn.usage.input_includes_cache) == (812, 64, False)
    assert turn.stop_reason == "tool_use"


def test_anthropic_groups_tool_results_into_one_user_message():
    calls = [ToolCall("toolu_01", "read", {"path": "a"}), ToolCall("toolu_02", "list", {"path": "."})]
    body = providers.anthropic_request("m", "sys", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "toolu_01", "name": "read", "content": "A"},
        {"role": "tool", "tool_call_id": "toolu_02", "name": "list", "content": "B"},
    ], [READ], 1024)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in body["messages"][2]["content"]] == ["toolu_01", "toolu_02"]
    assert body["system"] == "sys" and body["tools"][0]["input_schema"]["required"] == ["path"]


def test_parse_openai_tool_calls_decodes_arguments():
    turn = providers.parse_openai(load("openai_tool_calls.json"))
    assert turn.text == ""
    assert turn.tool_calls == [ToolCall("call_1", "grep", {"pattern": "label", "path": "candidates"})]
    assert (turn.usage.input, turn.usage.output, turn.usage.input_includes_cache) == (700, 30, True)
    assert turn.stop_reason == "tool_use"


def test_parse_openai_keeps_invalid_arguments_raw():
    turn = providers.parse_openai(load("openai_bad_arguments.json"))
    assert set(turn.tool_calls[0].arguments) == {"_raw"}


def test_openai_round_trip_reencodes_arguments_as_string():
    turn = providers.parse_openai(load("openai_tool_calls.json"))
    body = providers.openai_request("m", "sys", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "name": "grep", "content": "hits"},
    ], [READ], 512)
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    sent = body["messages"][2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(sent, str) and json.loads(sent) == {"pattern": "label", "path": "candidates"}
    assert body["messages"][3]["role"] == "tool"
    assert body["tools"][0]["type"] == "function"


def test_openai_invalid_arguments_are_resent_verbatim():
    turn = providers.parse_openai(load("openai_bad_arguments.json"))
    body = providers.openai_request("m", "", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "tool_calls": turn.tool_calls},
        {"role": "tool", "tool_call_id": "call_1", "name": "grep", "content": "bad args"},
    ], [], 512)
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"pattern": '


def test_text_fixtures_have_no_tool_calls():
    a = providers.parse_anthropic(load("anthropic_text.json"))
    o = providers.parse_openai(load("openai_text.json"))
    assert a.text == o.text == "The frontier has three candidates."
    assert a.tool_calls == o.tool_calls == []
    assert a.stop_reason == o.stop_reason == "end"


@pytest.mark.parametrize("cls,fixture,url_tail", [
    (providers.AnthropicModel, "anthropic_tool_use.json", "/v1/messages"),
    (providers.OpenAICompatibleModel, "openai_tool_calls.json", "/chat/completions"),
    (providers.UnikeyModel, "openai_tool_calls.json", "/chat/completions"),
    (providers.GroqModel, "openai_tool_calls.json", "/chat/completions"),
])
def test_chat_posts_and_parses(monkeypatch, cls, fixture, url_tail):
    monkeypatch.setenv(cls.API_KEY_ENV, "k")
    model = cls("m")
    sent = []
    model.client.post = lambda url, headers, body: sent.append((url, body)) or load(fixture)
    turn = model.chat("sys", [{"role": "user", "content": "go"}], [READ], reasoning_effort="high")
    assert sent[0][0].endswith(url_tail)
    assert sent[0][1]["reasoning_effort"] == "high"
    assert turn.tool_calls


def test_missing_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="ANTHROPIC_API_KEY"):
        providers.model_from_environment("anthropic", "m")


def test_compatible_needs_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError, match="compatible needs OPENAI_BASE_URL or --base-url"):
        providers.model_from_environment("compatible", "m")
    model = providers.model_from_environment("compatible", "m", base_url="http://localhost:8000/v1/")
    assert model.base_url == "http://localhost:8000/v1"


def test_groq_defaults(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    model = providers.model_from_environment("groq", "llama")
    assert isinstance(model, providers.GroqModel)
    assert model.base_url == "https://api.groq.com/openai/v1"
    assert (providers.GroqModel.API_KEY_ENV, providers.GroqModel.BASE_URL_ENV) == ("GROQ_API_KEY", "GROQ_BASE_URL")


# --- usage, stop reasons, thinking, orphans ---------------------------------------------------

def test_cache_fields_parsed_from_both_fixtures():
    a = providers.parse_anthropic(load("anthropic_tool_use.json")).usage
    assert (a.cache_read, a.cache_write) == (2048, 128)
    o = providers.parse_openai(load("openai_tool_calls.json")).usage
    assert (o.cache_read, o.reasoning) == (512, 12)


def test_max_tokens_normalises_to_length():
    a = load("anthropic_text.json")
    a["stop_reason"] = "max_tokens"
    o = load("openai_text.json")
    o["choices"][0]["finish_reason"] = "length"
    assert providers.parse_anthropic(a).stop_reason == "length"
    assert providers.parse_openai(o).stop_reason == "length"


def test_thinking_block_round_trips_into_next_request():
    response = load("anthropic_tool_use.json")
    thinking = {"type": "thinking", "thinking": "plan", "signature": "sig=="}
    response["content"].insert(0, thinking)
    turn = providers.parse_anthropic(response)
    assert turn.provider_blocks[0] == thinking
    body = providers.anthropic_request("m", "s", [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": turn.text, "tool_calls": turn.tool_calls,
         "provider_blocks": turn.provider_blocks},
        {"role": "tool", "tool_call_id": "toolu_01", "name": "read", "content": "A"},
        {"role": "tool", "tool_call_id": "toolu_02", "name": "list", "content": "B"},
    ], [], 100)
    assert body["messages"][1]["content"][0] == thinking


def test_orphaned_calls_get_error_result_in_both_formats():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("c1", "read", {"path": "a"}),
                                                        ToolCall("c2", "read", {"path": "b"})]},
        {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "A"},
        {"role": "user", "content": "continue"},
    ]
    a = providers.anthropic_request("m", "s", messages, [], 100)
    results = a["messages"][2]["content"]
    assert [b["tool_use_id"] for b in results] == ["c1", "c2"]
    assert results[1]["content"] == "error: no result (turn ended)"
    o = providers.openai_request("m", "s", messages, [], 100)
    tools = [m for m in o["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tools] == ["c1", "c2"]
    assert tools[1]["content"] == "error: no result (turn ended)"
    assert o["messages"][-1] == {"role": "user", "content": "continue"}


def test_cost_usd_arithmetic_and_unknown_price():
    price = providers.Price(input=3.0, output=15.0, cache_read=0.3, cache_write=3.75)
    anthropic = Usage(input=1_000_000, output=100_000, cache_read=2_000_000, cache_write=100_000)
    assert providers.cost_usd(anthropic, price) == pytest.approx(3.0 + 1.5 + 0.6 + 0.375)
    openai = Usage(input=1_000_000, output=0, cache_read=400_000, input_includes_cache=True)
    assert providers.cost_usd(openai, price) == pytest.approx(0.6 * 3.0 + 0.4 * 0.3)
    assert providers.cost_usd(anthropic, None) is None


def test_price_lookup_from_environment(monkeypatch):
    monkeypatch.setenv("META_HARNESS_PRICES", json.dumps({"m": {"input": 1, "output": 2}}))
    assert providers.price_for("m") == providers.Price(1.0, 2.0, 0.0, 0.0)
    assert providers.price_for("unknown") is None
    assert providers.price_for("x", {"x": providers.Price(1, 1, 1, 1)}).cache_read == 1


# --- retries ------------------------------------------------------------------------------------

class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def _http_error(code, headers=None):
    return urllib.error.HTTPError("http://x", code, "err", headers or {}, io.BytesIO(b"nope"))


def _client(script):
    calls, sleeps = [], []

    def urlopen(request, timeout):
        calls.append(request)
        item = script[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return _Response(item)

    client = providers._HttpClient(urlopen=urlopen, sleep=sleeps.append, rng=random.Random(0))
    return client, calls, sleeps


OK = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")


def test_retries_503_then_succeeds():
    client, calls, sleeps = _client([_http_error(503), OK])
    assert client.post("http://x", {}, {})["choices"]
    assert len(calls) == 2 and len(sleeps) == 1


@pytest.mark.parametrize("code", [400, 408, 409])
def test_does_not_retry_4xx(code):
    client, calls, sleeps = _client([_http_error(code), OK])
    with pytest.raises(providers.ProviderError, match=f"HTTP {code}"):
        client.post("http://x", {}, {})
    assert len(calls) == 1 and sleeps == []


def test_retry_after_seconds_is_honoured_and_capped():
    client, _, sleeps = _client([_http_error(429, {"Retry-After": "3"}),
                                 _http_error(429, {"Retry-After": "600"}), OK])
    client.post("http://x", {}, {})
    assert sleeps == [3.0, 60.0]


def test_retry_after_http_date():
    when = email.utils.formatdate(timeval=time.time() + 5, usegmt=True)
    client, _, sleeps = _client([_http_error(503, {"Retry-After": when}), OK])
    client.post("http://x", {}, {})
    assert 3.0 <= sleeps[0] <= 6.0


def test_backoff_jitter_within_ten_percent():
    client, _, sleeps = _client([_http_error(500), _http_error(500), _http_error(500), OK])
    client.post("http://x", {}, {})
    for attempt, slept in enumerate(sleeps):
        base = min(10, 0.5 * 2 ** attempt)
        assert base * 0.9 <= slept <= base * 1.1
        assert slept != base
    assert len(sleeps) == 3


def test_empty_200_and_transport_errors_are_retried():
    empty = json.dumps({"choices": []}).encode("utf-8")
    client, calls, _ = _client([b"", empty, urllib.error.URLError("reset"), TimeoutError(), OK])
    client.retry = providers.RetryPolicy(attempts=5)
    assert client.post("http://x", {}, {})["choices"]
    assert len(calls) == 5


def test_gives_up_after_attempts():
    client, calls, _ = _client([_http_error(503)] * 4)
    with pytest.raises(providers.ProviderError, match="HTTP 503"):
        client.post("http://x", {}, {})
    assert len(calls) == 4


# --- list_models and embeddings -----------------------------------------------------------------

def test_list_models_parses_data_ids(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("GROQ_BASE_URL", raising=False)
    seen = []

    def fake_get(url, headers, timeout):
        seen.append(url)
        return {"data": [{"id": "b"}, {"id": "a"}, {}]}

    monkeypatch.setattr(providers, "_http_get_json", fake_get)
    assert providers.list_models("groq") == ["a", "b"]
    assert seen == ["https://api.groq.com/openai/v1/models"]
    with pytest.raises(ValueError):
        providers.list_models("anthropic")


def test_embed_posts_and_orders_by_index(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    sent = []

    def post(url, headers, body):
        sent.append((url, body))
        return {"object": "list", "data": [{"index": 1, "embedding": [0.0, 1.0]},
                                           {"index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 4, "total_tokens": 4}}

    vectors = embeddings.embed(["a", "b"], post=post)
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert sent[0][0] == "https://api.openai.com/v1/embeddings"
    assert sent[0][1] == {"model": "text-embedding-3-small", "input": ["a", "b"]}


def test_embed_missing_key_names_variable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(providers.ProviderError, match="GROQ_API_KEY"):
        embeddings.embed(["a"], provider="groq", post=lambda *a: {})


def test_cosine():
    assert embeddings.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert embeddings.cosine([1, 0], [0, 2]) == pytest.approx(0.0)
    assert embeddings.cosine([1, 1], [-1, -1]) == pytest.approx(-1.0)
    assert embeddings.cosine([0, 0], [1, 0]) == 0.0


# --- review fixes: redaction, tool errors, Retry-After floor -----------------------------------

def test_http_error_body_and_configured_key_are_redacted():
    key = "plain-configured-key-0123456789"
    body = b'{"error": "Incorrect API key provided: sk-proj-abc123def456. Key ' + key.encode() + b'"}'

    def urlopen(request, timeout):
        raise urllib.error.HTTPError("http://x", 401, "err", {}, io.BytesIO(body))

    client = providers._HttpClient(urlopen=urlopen, sleep=lambda s: None)
    with pytest.raises(providers.ProviderError) as info:
        client.post("http://x", {"Authorization": f"Bearer {key}"}, {})
    text = str(info.value)
    assert "HTTP 401" in text
    assert "sk-proj-abc123" not in text and key not in text


def test_unexpected_response_with_env_key_is_redacted(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key-value-9876543210")
    with pytest.raises(providers.ProviderError) as info:
        providers.parse_openai({"echo": "env-key-value-9876543210"})
    assert "env-key-value-9876543210" not in str(info.value)


def test_anthropic_tool_errors_are_flagged_openai_ignores_flag():
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "text": "", "tool_calls": [ToolCall("c1", "read", {}), ToolCall("c2", "read", {}),
                                                        ToolCall("c3", "read", {})]},
        {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "ok"},
        {"role": "tool", "tool_call_id": "c2", "name": "read", "content": "boom", "is_error": True},
    ]
    blocks = providers.anthropic_request("m", "s", messages, [], 10)["messages"][2]["content"]
    assert [b.get("is_error", False) for b in blocks] == [False, True, True]
    tools = [m for m in providers.openai_request("m", "s", messages, [], 10)["messages"] if m["role"] == "tool"]
    assert all(set(m) == {"role", "tool_call_id", "content"} for m in tools)


def test_retry_after_zero_floors_at_half_second():
    client, _, sleeps = _client([_http_error(429, {"Retry-After": "0"}), OK])
    client.post("http://x", {}, {})
    assert sleeps == [0.5]
