"""Credential-free HTTP model adapters for OpenAI-compatible and Anthropic APIs.

The adapters implement the small callable interface consumed by the harnesses
(`call()` / `__call__`) plus `chat()`, a tool-calling turn over provider-neutral
messages (see `chat_types`). They use urllib from the standard library, so the
repository does not force an SDK or a vendor-specific dependency on users.

`call()` returns `(text, usage)` rather than stashing usage on the instance, so a
single model object can be shared by concurrently evaluated candidates.
"""

from __future__ import annotations

import email.utils
import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .redact import REDACTED, redact
from .chat_types import AssistantTurn, ToolCall, ToolSpec, Usage

UNIKEY_BASE_URL = "https://www.getunikey.ai/v1"
UNIKEY_ANTHROPIC_BASE_URL = "https://www.getunikey.ai"
RETRY_AFTER_FLOOR = 0.5
ORPHAN_RESULT = "error: no result (turn ended)"
RETRY_AFTER_CAP = 60.0


class ProviderError(RuntimeError):
    """Provider failure. The message is always redacted: secret patterns, loaded *_API_KEY values
    and any extra `secrets` (e.g. an explicitly passed key) never appear in str(exc)."""

    def __init__(self, message: str = "", secrets: Sequence[str] = (), status: int | None = None):
        self.status = status  # the HTTP status when the failure was an HTTP response, else None
        text = str(message)
        for secret in sorted((s for s in secrets if s), key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        super().__init__(redact(text))


@dataclass
class RetryPolicy:
    attempts: int = 4


class _Retryable(Exception):
    """A provider failure of a retryable class: EMPTY_RESPONSE, RATE_LIMIT, SERVER, TIMEOUT, TRANSPORT."""

    def __init__(self, kind: str, message: str, retry_after: float | None = None,
                 status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.retry_after = retry_after
        self.status = status


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        seconds = when.timestamp() - time.time()
    return max(RETRY_AFTER_FLOOR, min(RETRY_AFTER_CAP, seconds))


def _is_empty(value: Mapping[str, Any]) -> bool:
    if "data" in value or ("content" in value and value.get("stop_reason")):
        return False
    return not value.get("choices") and not value.get("content")


class _HttpClient:
    """POSTs JSON. Retries only HTTP 429, 5xx, empty 200 bodies, timeouts and transport errors."""

    def __init__(self, timeout: float = 120.0, retry: RetryPolicy | None = None, *,
                 urlopen: Callable[..., Any] | None = None, sleep: Callable[[float], Any] | None = None,
                 rng: random.Random | None = None):
        self.timeout = timeout
        self.retry = retry or RetryPolicy()
        self.urlopen = urlopen or urllib.request.urlopen
        self.sleep = sleep or time.sleep
        self.rng = rng or random.Random()

    def _once(self, url: str, headers: Mapping[str, str], payload: bytes) -> dict[str, Any]:
        request = urllib.request.Request(url, payload, {"Content-Type": "application/json", **headers},
                                         method="POST")
        try:
            with self.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            body_text = error.read().decode("utf-8", errors="replace")
            message = f"HTTP {error.code}: {body_text[:1000]}"
            if error.code == 429:
                raise _Retryable("RATE_LIMIT", message, _retry_after(error.headers.get("Retry-After")
                                                                   if error.headers else None),
                                 error.code) from error
            if error.code >= 500:
                raise _Retryable("SERVER", message, _retry_after(error.headers.get("Retry-After")
                                                               if error.headers else None),
                                 error.code) from error
            raise ProviderError(message, status=error.code) from error
        except (TimeoutError, OSError) as error:
            # URLError is an OSError; so are connection resets and socket timeouts.
            kind = "TIMEOUT" if isinstance(error, TimeoutError) else "TRANSPORT"
            raise _Retryable(kind, f"{kind.lower()}: {error}") from error
        if not raw or not raw.strip():
            raise _Retryable("EMPTY_RESPONSE", "provider returned an empty body")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise ProviderError(f"provider returned invalid JSON: {raw[:500]!r}") from error
        if not isinstance(value, dict):
            raise ProviderError("provider returned a non-object response")
        if _is_empty(value):
            raise _Retryable("EMPTY_RESPONSE", f"provider returned no choices or content: {str(value)[:500]}")
        return value

    def post(self, url: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        secrets = [value.split()[-1] for name, value in headers.items()
                   if name.lower() in {"authorization", "x-api-key"} and value.strip()]
        try:
            return self._post(url, headers, payload)
        except ProviderError as error:
            raise ProviderError(str(error.args[0] if error.args else error), secrets,
                                status=error.status) from error.__cause__

    def _post(self, url: str, headers: Mapping[str, str], payload: bytes) -> dict[str, Any]:
        attempts = max(1, self.retry.attempts)
        for attempt in range(attempts):
            try:
                return self._once(url, headers, payload)
            except _Retryable as error:
                if attempt + 1 >= attempts:
                    raise ProviderError(f"{error.kind}: {error}", status=error.status) from error
                if error.retry_after is not None:
                    delay = error.retry_after
                else:
                    base = min(10.0, 0.5 * 2 ** attempt)
                    delay = base * (1 + self.rng.uniform(-0.1, 0.1))
                self.sleep(delay)
        raise ProviderError("provider request failed")  # unreachable


def _http_get_json(url: str, headers: Mapping[str, str], timeout: float = 60.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=dict(headers), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise ProviderError(f"HTTP {error.code}: {error.read().decode('utf-8', errors='replace')[:1000]}",
                            status=error.code) from error
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProviderError(str(error)) from error
    if not isinstance(value, dict):
        raise ProviderError("provider returned a non-object response")
    return value


# --- pricing ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


def price_for(model: str, overrides: Mapping[str, Price] | None = None) -> Price | None:
    """Price from explicit overrides (`--price`) or META_HARNESS_PRICES JSON; None when unknown."""
    if overrides and model in overrides:
        return overrides[model]
    text = os.environ.get("META_HARNESS_PRICES", "")
    if not text:
        return None
    try:
        table = json.loads(text)
        entry = table.get(model) if isinstance(table, dict) else None
        if not isinstance(entry, dict):
            return None
        return Price(float(entry["input"]), float(entry["output"]),
                     float(entry.get("cache_read", 0.0)), float(entry.get("cache_write", 0.0)))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def cost_usd(usage: Usage, price: Price | None) -> float | None:
    """Billed cost, separate from context tokens. Unknown price -> None, never a guess."""
    if price is None:
        return None
    uncached = usage.input
    if usage.input_includes_cache:
        uncached = max(0, usage.input - usage.cache_read - usage.cache_write)
    return (uncached * price.input + usage.output * price.output
            + usage.cache_read * price.cache_read + usage.cache_write * price.cache_write) / 1_000_000


# --- neutral <-> wire translation -------------------------------------------------------------

def _close_orphans(messages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Insert an error tool result for every assistant call left unanswered."""
    out: list[Mapping[str, Any]] = []
    pending: list[ToolCall] = []

    def flush() -> None:
        for call in pending:
            out.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": ORPHAN_RESULT,
                        "is_error": True})
        pending.clear()

    for message in messages:
        role = message.get("role")
        if role in {"user", "assistant"}:
            flush()
        out.append(message)
        if role == "assistant":
            pending.extend(message.get("tool_calls") or [])
        elif role == "tool":
            pending[:] = [c for c in pending if c.id != message.get("tool_call_id")]
    flush()
    return out


def _anthropic_stop(reason: str) -> str:
    return {"end_turn": "end", "stop_sequence": "end", "tool_use": "tool_use",
            "max_tokens": "length"}.get(reason, "other")


def _openai_stop(reason: str) -> str:
    return {"stop": "end", "tool_calls": "tool_use", "function_call": "tool_use",
            "length": "length"}.get(reason, "other")


def _int(mapping: Any, key: str) -> int:
    if not isinstance(mapping, Mapping):
        return 0
    try:
        return int(mapping.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def anthropic_request(model: str, system: str, messages: list[dict], tools: Sequence[ToolSpec],
                      max_tokens: int) -> dict:
    wire: list[dict] = []
    for message in _close_orphans(messages):
        role = message["role"]
        if role == "user":
            wire.append({"role": "user", "content": str(message["content"])})
        elif role == "assistant":
            if message.get("provider_blocks"):
                # Echo raw blocks unchanged so thinking signatures stay valid.
                blocks = [dict(b) for b in message["provider_blocks"]]
            else:
                blocks = [{"type": "text", "text": message["text"]}] if message.get("text") else []
                blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                           for c in message.get("tool_calls") or []]
            wire.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                     "content": str(message["content"])}
            if message.get("is_error"):
                block["is_error"] = True
            # Several results for one assistant turn go in ONE user message.
            if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                wire[-1]["content"].append(block)
            else:
                wire.append({"role": "user", "content": [block]})
        else:
            raise ValueError(f"unknown role {role!r}")
    body: dict[str, Any] = {"model": model, "max_tokens": max_tokens, "system": system, "messages": wire}
    if tools:
        body["tools"] = [{"name": t.name, "description": t.description, "input_schema": t.parameters}
                         for t in tools]
    return body


def parse_anthropic(response: Mapping[str, Any]) -> AssistantTurn:
    try:
        blocks = [dict(b) for b in response["content"]]
    except (KeyError, TypeError, ValueError) as error:
        raise ProviderError(f"unexpected messages response: {str(response)[:500]}") from error
    text = "".join(str(b.get("text", "")) for b in blocks if b.get("type") == "text")
    calls = [ToolCall(str(b["id"]), str(b["name"]), dict(b.get("input") or {}))
             for b in blocks if b.get("type") == "tool_use"]
    raw = response.get("usage") or {}
    usage = Usage(input=_int(raw, "input_tokens"), output=_int(raw, "output_tokens"),
                  cache_read=_int(raw, "cache_read_input_tokens"),
                  cache_write=_int(raw, "cache_creation_input_tokens"), input_includes_cache=False)
    return AssistantTurn(text, calls, usage, _anthropic_stop(str(response.get("stop_reason") or "")),
                         provider_blocks=blocks)


def _wire_arguments(call: ToolCall) -> str:
    if set(call.arguments) == {"_raw"}:
        return str(call.arguments["_raw"])
    return json.dumps(call.arguments)


def openai_request(model: str, system: str, messages: list[dict], tools: Sequence[ToolSpec],
                   max_tokens: int) -> dict:
    wire: list[dict] = [{"role": "system", "content": system}] if system else []
    for message in _close_orphans(messages):
        role = message["role"]
        if role == "user":
            wire.append({"role": "user", "content": str(message["content"])})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": message.get("text") or None}
            if message.get("tool_calls"):
                item["tool_calls"] = [{"id": c.id, "type": "function",
                                       "function": {"name": c.name, "arguments": _wire_arguments(c)}}
                                      for c in message["tool_calls"]]
            wire.append(item)
        elif role == "tool":
            wire.append({"role": "tool", "tool_call_id": message["tool_call_id"],
                         "content": str(message["content"])})
        else:
            raise ValueError(f"unknown role {role!r}")
    body: dict[str, Any] = {"model": model, "messages": wire, "max_tokens": max_tokens}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                           "parameters": t.parameters}} for t in tools]
    return body


def parse_openai(response: Mapping[str, Any]) -> AssistantTurn:
    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise ProviderError(f"unexpected chat completion response: {str(response)[:500]}") from error
    calls = []
    for raw in message.get("tool_calls") or []:
        function = raw.get("function") or {}
        text = function.get("arguments") or "{}"
        try:
            arguments = json.loads(text) if isinstance(text, str) else dict(text)
            if not isinstance(arguments, dict):
                arguments = {"_raw": text}
        except (json.JSONDecodeError, TypeError, ValueError):
            arguments = {"_raw": text}
        calls.append(ToolCall(str(raw.get("id", "")), str(function.get("name", "")), arguments))
    raw_usage = response.get("usage") or {}
    usage = Usage(input=_int(raw_usage, "prompt_tokens"), output=_int(raw_usage, "completion_tokens"),
                  cache_read=_int(raw_usage.get("prompt_tokens_details"), "cached_tokens"),
                  reasoning=_int(raw_usage.get("completion_tokens_details"), "reasoning_tokens"),
                  input_includes_cache=True)
    return AssistantTurn(str(message.get("content") or ""), calls, usage,
                         _openai_stop(str(choice.get("finish_reason") or "")))



# --- models -----------------------------------------------------------------------------------

def _require_key(cls: type, api_key: str | None) -> str:
    key = api_key or os.environ.get(cls.API_KEY_ENV, "")  # type: ignore[attr-defined]
    if not key:
        raise ProviderError(f"{cls.API_KEY_ENV} is not set")  # type: ignore[attr-defined]
    return key


class OpenAICompatibleModel:
    """Callable adapter for OpenAI, Azure-compatible gateways, and local servers."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    API_KEY_ENV = "OPENAI_API_KEY"
    BASE_URL_ENV = "OPENAI_BASE_URL"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, retry: RetryPolicy | None = None):
        self.model = model
        self.api_key = _require_key(type(self), api_key)
        self.base_url = (base_url or os.environ.get(self.BASE_URL_ENV) or self.DEFAULT_BASE_URL).rstrip("/")
        self.client = _HttpClient(timeout, retry)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        messages = kwargs.pop("messages", [{"role": "user", "content": prompt}])
        body = {"model": self.model, "messages": messages, **kwargs}
        response = self.client.post(self.base_url + "/chat/completions", self._headers(), body)
        try:
            text = str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(f"unexpected chat completion response: {response}") from error
        usage = response.get("usage") or {}
        return text, {"prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                      "completion_tokens": int(usage.get("completion_tokens", 0) or 0)}

    def chat(self, system: str, messages: list[dict], tools: Sequence[ToolSpec], max_tokens: int = 4096,
             **kwargs: Any) -> AssistantTurn:
        body = {**openai_request(self.model, system, messages, tools, max_tokens), **kwargs}
        return parse_openai(self.client.post(self.base_url + "/chat/completions", self._headers(), body))

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]


class UnikeyModel(OpenAICompatibleModel):
    """Unikey gateway (https://www.getunikey.ai) - OpenAI-compatible chat completions.

    The Anthropic-compatible root is reachable as `--provider anthropic --base-url
    https://www.getunikey.ai`.
    """

    DEFAULT_BASE_URL = UNIKEY_BASE_URL
    API_KEY_ENV = "UNIKEY_API_KEY"
    BASE_URL_ENV = "UNIKEY_BASE_URL"


class GroqModel(OpenAICompatibleModel):
    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    API_KEY_ENV = "GROQ_API_KEY"
    BASE_URL_ENV = "GROQ_BASE_URL"


class AnthropicModel:
    DEFAULT_BASE_URL = "https://api.anthropic.com"
    API_KEY_ENV = "ANTHROPIC_API_KEY"
    BASE_URL_ENV = "ANTHROPIC_BASE_URL"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, retry: RetryPolicy | None = None, max_tokens: int = 4096):
        self.model = model
        self.api_key = _require_key(type(self), api_key)
        self.base_url = (base_url or os.environ.get(self.BASE_URL_ENV) or self.DEFAULT_BASE_URL).rstrip("/")
        self.max_tokens = max_tokens
        self.client = _HttpClient(timeout, retry)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        body = {"model": self.model, "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
                "messages": [{"role": "user", "content": prompt}], **kwargs}
        response = self.client.post(self.base_url + "/v1/messages", self._headers(), body)
        try:
            text = "".join(str(block.get("text", "")) for block in response["content"] if block.get("type") == "text")
        except (KeyError, TypeError) as error:
            raise ProviderError(f"unexpected messages response: {response}") from error
        usage = response.get("usage") or {}
        return text, {"prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                      "completion_tokens": int(usage.get("output_tokens", 0) or 0)}

    def chat(self, system: str, messages: list[dict], tools: Sequence[ToolSpec], max_tokens: int = 4096,
             **kwargs: Any) -> AssistantTurn:
        body = {**anthropic_request(self.model, system, messages, tools, max_tokens), **kwargs}
        return parse_anthropic(self.client.post(self.base_url + "/v1/messages", self._headers(), body))

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]


OPENAI_COMPATIBLE = {"openai": OpenAICompatibleModel, "compatible": OpenAICompatibleModel,
                     "unikey": UnikeyModel, "groq": GroqModel}


def list_models(provider: str, api_key: str | None = None, base_url: str | None = None,
                timeout: float = 60.0) -> list[str]:
    """GET <base>/models for an OpenAI-compatible provider; returns sorted `data[].id`."""
    cls = OPENAI_COMPATIBLE.get(provider.lower())
    if cls is None:
        raise ValueError(f"list_models supports {', '.join(OPENAI_COMPATIBLE)}; got {provider!r}")
    key = _require_key(cls, api_key)
    root = (base_url or os.environ.get(cls.BASE_URL_ENV) or cls.DEFAULT_BASE_URL).rstrip("/")
    try:
        payload = _http_get_json(root + "/models", {"Authorization": f"Bearer {key}"}, timeout)
    except ProviderError as error:
        raise ProviderError(str(error), [key]) from error
    return sorted(str(item.get("id", "")) for item in payload.get("data", []) if item.get("id"))


def list_unikey_models(api_key: str | None = None, base_url: str | None = None, timeout: float = 60.0) -> list[str]:
    """GET /v1/models against the Unikey gateway."""
    return list_models("unikey", api_key=api_key, base_url=base_url, timeout=timeout)


def model_from_environment(provider: str, model: str, **kwargs: Any) -> Any:
    provider = provider.lower()
    if provider in {"unikey", "getunikey"}:
        return UnikeyModel(model, **kwargs)
    if provider == "groq":
        return GroqModel(model, **kwargs)
    if provider == "compatible":
        if not (kwargs.get("base_url") or os.environ.get(OpenAICompatibleModel.BASE_URL_ENV)):
            raise ValueError("compatible needs OPENAI_BASE_URL or --base-url")
        return OpenAICompatibleModel(model, **kwargs)
    if provider in {"openai", "azure"}:
        return OpenAICompatibleModel(model, **kwargs)
    if provider in {"anthropic", "claude"}:
        return AnthropicModel(model, **kwargs)
    if provider == "scripted":
        from .scripted import ScriptedModel  # Task 4; imported lazily
        return ScriptedModel(model, **kwargs)
    raise ValueError(f"unsupported provider: {provider}")


__all__ = [
    "AnthropicModel", "AssistantTurn", "GroqModel", "OpenAICompatibleModel", "Price", "ProviderError",
    "RetryPolicy", "ToolCall", "ToolSpec", "UnikeyModel", "Usage", "anthropic_request", "cost_usd",
    "list_models", "list_unikey_models", "model_from_environment", "openai_request", "parse_anthropic",
    "parse_openai", "price_for",
]
