"""Credential-free HTTP model adapters for OpenAI-compatible and Anthropic APIs.

The adapters intentionally implement the small callable interface consumed by
the harnesses. They use urllib from the standard library, so the repository
does not force an SDK or a vendor-specific dependency on users.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


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
                if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
            if attempt + 1 < self.retry.attempts:
                time.sleep(min(self.retry.maximum_delay, self.retry.initial_delay * (2 ** attempt)))
        raise ProviderError(str(last_error or "provider request failed"))


class OpenAICompatibleModel:
    """Callable adapter for OpenAI, Azure-compatible gateways, and local servers."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0, retry: RetryPolicy | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.client = _HttpClient(timeout, retry)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        messages = kwargs.pop("messages", [{"role": "user", "content": prompt}])
        body = {"model": self.model, "messages": messages, **kwargs}
        response = self.client.post(self.base_url + "/chat/completions", {"Authorization": f"Bearer {self.api_key}"}, body)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"unexpected chat completion response: {response}") from error


class AnthropicModel:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None, timeout: float = 120.0, retry: RetryPolicy | None = None, max_tokens: int = 4096):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = (base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")).rstrip("/")
        self.max_tokens = max_tokens
        self.client = _HttpClient(timeout, retry)

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        body = {"model": self.model, "max_tokens": kwargs.pop("max_tokens", self.max_tokens), "messages": [{"role": "user", "content": prompt}], **kwargs}
        response = self.client.post(self.base_url + "/v1/messages", {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body)
        try:
            return "".join(str(block.get("text", "")) for block in response["content"] if block.get("type") == "text")
        except (KeyError, TypeError) as error:
            raise ProviderError(f"unexpected messages response: {response}") from error


def model_from_environment(provider: str, model: str, **kwargs: Any) -> Any:
    provider = provider.lower()
    if provider in {"openai", "compatible", "azure"}:
        return OpenAICompatibleModel(model, **kwargs)
    if provider in {"anthropic", "claude"}:
        return AnthropicModel(model, **kwargs)
    raise ValueError(f"unsupported provider: {provider}")
