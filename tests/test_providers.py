from meta_harness import providers


class _FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, headers, body):
        self.calls.append((url, dict(headers), dict(body)))
        return self.response


def test_unikey_defaults_to_gateway_base_url(monkeypatch):
    monkeypatch.setenv("UNIKEY_API_KEY", "unikey-test-key")
    monkeypatch.delenv("UNIKEY_BASE_URL", raising=False)
    model = providers.UnikeyModel("gpt-5.2")
    assert model.base_url == "https://www.getunikey.ai/v1"
    assert model.api_key == "unikey-test-key"


def test_call_returns_text_and_usage(monkeypatch):
    monkeypatch.delenv("UNIKEY_BASE_URL", raising=False)
    model = providers.UnikeyModel("gpt-5.2", api_key="k")
    model.client = _FakeClient({
        "choices": [{"message": {"content": "hello"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    })
    text, usage = model.call("hi")
    assert text == "hello"
    assert usage["prompt_tokens"] == 11
    url, headers, body = model.client.calls[0]
    assert url == "https://www.getunikey.ai/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert body["model"] == "gpt-5.2"


def test_dunder_call_still_returns_plain_string():
    model = providers.UnikeyModel("gpt-5.2", api_key="k")
    model.client = _FakeClient({"choices": [{"message": {"content": "hello"}}]})
    assert model("hi") == "hello"


def test_anthropic_call_returns_usage():
    model = providers.AnthropicModel("claude-sonnet-4-6", api_key="k")
    model.client = _FakeClient({
        "content": [{"type": "text", "text": "ok"}],
        "usage": {"input_tokens": 7, "output_tokens": 2},
    })
    text, usage = model.call("hi")
    assert text == "ok"
    assert usage == {"prompt_tokens": 7, "completion_tokens": 2}


def test_model_from_environment_supports_unikey(monkeypatch):
    monkeypatch.setenv("UNIKEY_API_KEY", "k")
    assert isinstance(providers.model_from_environment("unikey", "gpt-5.2"), providers.UnikeyModel)


def test_list_unikey_models_parses_data(monkeypatch):
    monkeypatch.delenv("UNIKEY_BASE_URL", raising=False)
    payload = {"data": [{"id": "gpt-5.2"}, {"id": "claude-sonnet-4-6"}]}

    def fake_get(url, headers, timeout):
        assert url == "https://www.getunikey.ai/v1/models"
        assert headers["Authorization"] == "Bearer k"
        return payload

    monkeypatch.setattr(providers, "_http_get_json", fake_get)
    assert providers.list_unikey_models(api_key="k") == ["claude-sonnet-4-6", "gpt-5.2"]
