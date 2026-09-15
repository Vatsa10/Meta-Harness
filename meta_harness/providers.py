"""Credential-free HTTP model adapters for OpenAI-compatible and Anthropic APIs.

The adapters intentionally implement the small callable interface consumed by
the harnesses. They use urllib from the standard library, so the repository
does not force an SDK or a vendor-specific dependency on users.

`call()` returns `(text, usage)` rather than stashing usage on the instance, so a
single model object can be shared by concurrently evaluated candidates.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

UNIKEY_BASE_URL = "https://www.getunikey.ai/v1"
UNIKEY_ANTHROPIC_BASE_URL = "https://www.getunikey.ai"


class ProviderError(RuntimeError):
    pass


@dataclass
class RetryPolicy:
    attempts: int = 4
    initial_delay: float = 1.0
    maximum_delay: float = 30.0


class _HttpClient:
    def __init__(self, timeout: float = 120.0, retry: RetryPolicy | None = None):
        self.timeout = timeout
        self.retry = retry or RetryPolicy()

    def post(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retry.attempts):
            request = urllib.request.Request(url, payload, {"Content-Type": "application/json", **headers}, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                    if not isinstance(value, dict):
                        raise ProviderError("provider returned a non-object response")
                    return value
            except urllib.error.HTTPError as error:
                body_text = error.read().decode("utf-8", errors="replace")
                last_error = ProviderError(f"HTTP {error.code}: {body_text[:1000]}")
                # Retry anything transient: rate limits, conflicts, and every 5xx. Gateways in
                # front of model providers emit non-standard ones (Cloudflare 520-527).
                if not (error.code in {408, 409, 429} or error.code >= 500):
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
            if attempt + 1 < self.retry.attempts:
                time.sleep(min(self.retry.maximum_delay, self.retry.initial_delay * (2 ** attempt)))
        raise ProviderError(str(last_error or "provider request failed"))


def _http_get_json(url: str, headers: Mapping[str, str], timeout: float = 60.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise ProviderError(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')[:1000]}") from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProviderError(str(error)) from error
    if not isinstance(value, dict):
        raise ProviderError("provider returned a non-object response")
    return value


class OpenAICompatibleModel:
    """Callable adapter for OpenAI, Azure-compatible gateways, and local servers."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    API_KEY_ENV = "OPENAI_API_KEY"
    BASE_URL_ENV = "OPENAI_BASE_URL"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, retry: RetryPolicy | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV, "")
        self.base_url = (base_url or os.environ.get(self.BASE_URL_ENV) or self.DEFAULT_BASE_URL).rstrip("/")
        self.client = _HttpClient(timeout, retry)

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        messages = kwargs.pop("messages", [{"role": "user", "content": prompt}])
        body = {"model": self.model, "messages": messages, **kwargs}
        response = self.client.post(self.base_url + "/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, body)
        try:
            text = str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"unexpected chat completion response: {response}") from error
        usage = response.get("usage") or {}
        return text, {"prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                      "completion_tokens": int(usage.get("completion_tokens", 0) or 0)}

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]


class UnikeyModel(OpenAICompatibleModel):
    """Unikey gateway (https://www.getunikey.ai) - OpenAI-compatible chat completions."""

    DEFAULT_BASE_URL = UNIKEY_BASE_URL
    API_KEY_ENV = "UNIKEY_API_KEY"
    BASE_URL_ENV = "UNIKEY_BASE_URL"


class AnthropicModel:
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    API_KEY_ENV = "ANTHROPIC_API_KEY"
    BASE_URL_ENV = "ANTHROPIC_BASE_URL"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, retry: RetryPolicy | None = None, max_tokens: int = 4096):
        self.model = model
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV, "")
        self.base_url = (base_url or os.environ.get(self.BASE_URL_ENV) or self.DEFAULT_BASE_URL).rstrip("/")
        self.max_tokens = max_tokens
        self.client = _HttpClient(timeout, retry)

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        body = {"model": self.model, "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
                "messages": [{"role": "user", "content": prompt}], **kwargs}
        response = self.client.post(self.base_url + "/v1/messages",
                                    {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body)
        try:
            text = "".join(str(block.get("text", "")) for block in response["content"] if block.get("type") == "text")
        except (KeyError, TypeError) as error:
            raise ProviderError(f"unexpected messages response: {response}") from error
        usage = response.get("usage") or {}
        return text, {"prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                      "completion_tokens": int(usage.get("output_tokens", 0) or 0)}

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]


def list_unikey_models(api_key: str | None = None, base_url: str | None = None, timeout: float = 60.0) -> list[str]:
    """GET /v1/models against the Unikey gateway."""
    key = api_key or os.environ.get("UNIKEY_API_KEY", "")
    root = (base_url or os.environ.get("UNIKEY_BASE_URL") or UNIKEY_BASE_URL).rstrip("/")
    payload = _http_get_json(root + "/models", {"Authorization": f"Bearer {key}"}, timeout)
    return sorted(str(item.get("id", "")) for item in payload.get("data", []) if item.get("id"))


def model_from_environment(provider: str, model: str, **kwargs: Any) -> Any:
    provider = provider.lower()
    if provider in {"unikey", "getunikey"}:
        return UnikeyModel(model, **kwargs)
    if provider in {"openai", "compatible", "azure"}:
        return OpenAICompatibleModel(model, **kwargs)
    if provider in {"anthropic", "claude"}:
        return AnthropicModel(model, **kwargs)
    raise ValueError(f"unsupported provider: {provider}")
