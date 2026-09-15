# Meta-Harness End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing `meta_harness` scaffolding into a runnable end-to-end harness-search system backed by the Unikey model gateway, with a real coding-agent proposer, a held-out test split, seeded baselines, isolated evaluation, honest token accounting, and the paper's proposer-view ablation.

**Architecture:** The outer loop in `core.py` stays the spine. Around it we add: a Unikey provider that reports token usage, a disk cache, a metered model wrapper that writes every model call into the execution trace, subprocess-isolated interface validation, concurrent candidate evaluation, a filtered "view" of the experience directory handed to a Claude Code subprocess proposer, and real seed harnesses on disk.

**Tech Stack:** Python 3.10+, standard library only (`urllib`, `json`, `concurrent.futures`, `subprocess`, `sqlite3`-free flat-file cache). `pytest` for tests. Claude Code CLI as the proposer binary. Docker only for the optional terminal domain.

**Spec:** `docs/superpowers/specs/2026-09-15-meta-harness-end-to-end.md`

## Global Constraints

- Python `>=3.10`. **No new runtime dependencies** — `pyproject.toml` must keep an empty/absent `dependencies` list. `pytest` may be added under an optional dev extra only.
- Standard library only in `meta_harness/**`. No `requests`, no `httpx`, no `numpy`.
- Windows-compatible: no `os.symlink`, no `signal.alarm`, no `fcntl`. Use `shutil.copy2` and `subprocess` timeouts.
- API keys come from environment variables only: `UNIKEY_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Never write a key into the experience root, a trace, `run.json`, or any log.
- Unikey base URL constant: `https://www.getunikey.ai/v1`. Anthropic-shaped base for Claude Code: `https://www.getunikey.ai`.
- Every commit message is plain conventional-commit text. **Do not add any AI/assistant co-author, `Co-Authored-By`, or "Generated with" attribution line to any commit or PR.**
- Run the full suite with `python -m pytest tests -q` from the repo root.

---

### Task 1: Unikey provider and token-usage reporting

Adds Unikey to `providers.py` and changes every provider to expose `call()` returning
`(text, usage)` so token accounting is thread-safe (no mutable `last_usage` attribute shared
between concurrently evaluated candidates).

**Files:**
- Modify: `meta_harness/providers.py`
- Modify: `meta_harness/__init__.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `OpenAICompatibleModel.call(prompt: str, **kwargs) -> tuple[str, dict]`
  - `AnthropicModel.call(prompt: str, **kwargs) -> tuple[str, dict]`
  - `class UnikeyModel(OpenAICompatibleModel)` with `DEFAULT_BASE_URL = "https://www.getunikey.ai/v1"`
  - `list_unikey_models(api_key: str | None = None, base_url: str | None = None) -> list[str]`
  - `model_from_environment(provider: str, model: str, **kwargs) -> Any` accepting `"unikey"`
  - usage dicts use OpenAI key names: `{"prompt_tokens": int, "completion_tokens": int}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_providers.py`:

```python
import json
from types import SimpleNamespace

import pytest

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
    model = providers.UnikeyModel("gpt-5.2")
    assert model.base_url == "https://www.getunikey.ai/v1"
    assert model.api_key == "unikey-test-key"


def test_call_returns_text_and_usage():
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
    payload = {"data": [{"id": "gpt-5.2"}, {"id": "claude-sonnet-4-6"}]}

    def fake_get(url, headers, timeout):
        assert url == "https://www.getunikey.ai/v1/models"
        assert headers["Authorization"] == "Bearer k"
        return payload

    monkeypatch.setattr(providers, "_http_get_json", fake_get)
    assert providers.list_unikey_models(api_key="k") == ["claude-sonnet-4-6", "gpt-5.2"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_providers.py -q`
Expected: FAIL — `AttributeError: module 'meta_harness.providers' has no attribute 'UnikeyModel'`

- [ ] **Step 3: Write minimal implementation**

In `meta_harness/providers.py`, add a module-level GET helper after `_HttpClient`:

```python
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
```

Replace `OpenAICompatibleModel.__call__` with a `call` + `__call__` pair:

```python
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
```

Give `OpenAICompatibleModel` a class-level default and use it in `__init__`:

```python
class OpenAICompatibleModel:
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    API_KEY_ENV = "OPENAI_API_KEY"
    BASE_URL_ENV = "OPENAI_BASE_URL"

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 120.0, retry: RetryPolicy | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV, "")
        self.base_url = (base_url or os.environ.get(self.BASE_URL_ENV) or self.DEFAULT_BASE_URL).rstrip("/")
        self.client = _HttpClient(timeout, retry)
```

Do the same split for `AnthropicModel`:

```python
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
```

Add Unikey and the model listing:

```python
UNIKEY_BASE_URL = "https://www.getunikey.ai/v1"
UNIKEY_ANTHROPIC_BASE_URL = "https://www.getunikey.ai"


class UnikeyModel(OpenAICompatibleModel):
    """Unikey gateway (https://www.getunikey.ai) — OpenAI-compatible chat completions."""

    DEFAULT_BASE_URL = UNIKEY_BASE_URL
    API_KEY_ENV = "UNIKEY_API_KEY"
    BASE_URL_ENV = "UNIKEY_BASE_URL"


def list_unikey_models(api_key: str | None = None, base_url: str | None = None, timeout: float = 60.0) -> list[str]:
    key = api_key or os.environ.get("UNIKEY_API_KEY", "")
    root = (base_url or os.environ.get("UNIKEY_BASE_URL") or UNIKEY_BASE_URL).rstrip("/")
    payload = _http_get_json(root + "/models", {"Authorization": f"Bearer {key}"}, timeout)
    return sorted(str(item.get("id", "")) for item in payload.get("data", []) if item.get("id"))
```

Extend the factory:

```python
def model_from_environment(provider: str, model: str, **kwargs: Any) -> Any:
    provider = provider.lower()
    if provider in {"unikey", "getunikey"}:
        return UnikeyModel(model, **kwargs)
    if provider in {"openai", "compatible", "azure"}:
        return OpenAICompatibleModel(model, **kwargs)
    if provider in {"anthropic", "claude"}:
        return AnthropicModel(model, **kwargs)
    raise ValueError(f"unsupported provider: {provider}")
```

Export from `meta_harness/__init__.py`: add `UnikeyModel`, `list_unikey_models`,
`UNIKEY_BASE_URL`, `UNIKEY_ANTHROPIC_BASE_URL` to the `providers` import line and to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_providers.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the existing suite for regressions**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/providers.py meta_harness/__init__.py tests/test_providers.py
git commit -m "feat(providers): add Unikey gateway adapter and token-usage reporting"
```

---

### Task 2: Metrics — token estimation, exact match, math equivalence

**Files:**
- Create: `meta_harness/metrics.py`
- Modify: `meta_harness/__init__.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `estimate_tokens(text: str) -> int`
  - `normalize_answer(text: str) -> str`
  - `exact_match(prediction: Any, task: Mapping[str, Any]) -> float`
  - `math_equivalence(prediction: Any, task: Mapping[str, Any], tolerance: float = 1e-6) -> float`
  - `METRICS: dict[str, Callable[[Any, Mapping[str, Any]], float]]` keyed `"classification"`, `"math"`, `"terminal"` (the `"terminal"` entry is added in Task 12; until then the dict has two entries)

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
from meta_harness import metrics


def test_estimate_tokens_is_roughly_chars_over_four():
    assert metrics.estimate_tokens("") == 0
    assert metrics.estimate_tokens("abcd") == 1
    assert metrics.estimate_tokens("a" * 401) == 101


def test_normalize_answer_extracts_boxed():
    assert metrics.normalize_answer(r"So the answer is \boxed{42}.") == "42"
    assert metrics.normalize_answer(r"\boxed{\frac{1}{2}}") == "1/2"


def test_normalize_answer_strips_latex_noise():
    assert metrics.normalize_answer(r"$\left( 3 \right)$") == "3"
    assert metrics.normalize_answer("Answer: 1,024") == "1024"
    assert metrics.normalize_answer(r"\text{yes}") == "yes"


def test_exact_match_is_case_insensitive():
    assert metrics.exact_match("Fruit", {"label": "fruit"}) == 1.0
    assert metrics.exact_match("vehicle", {"label": "fruit"}) == 0.0


def test_math_equivalence_handles_numeric_forms():
    assert metrics.math_equivalence(r"\boxed{0.5}", {"answer": "1/2"}) == 1.0
    assert metrics.math_equivalence(r"the answer is \boxed{42}", {"answer": 42}) == 1.0
    assert metrics.math_equivalence(r"\boxed{43}", {"answer": 42}) == 0.0


def test_math_equivalence_falls_back_to_string_compare():
    assert metrics.math_equivalence(r"\boxed{x+1}", {"answer": "x + 1"}) == 1.0


def test_math_equivalence_without_answer_is_zero():
    assert metrics.math_equivalence("anything", {"problem": "p"}) == 0.0


def test_metrics_registry_exposes_task_types():
    assert set(metrics.METRICS) >= {"classification", "math"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/metrics.py`:

```python
"""Scoring functions and token accounting shared by evaluators and harnesses."""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Mapping

_BOXED = re.compile(r"\\boxed\s*\{")
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_TEXTLIKE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_WRAPPERS = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", "\\(", "\\)", "\\[", "\\]")
_TRAILING_PREFIX = re.compile(r"^(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:=]?\s*", re.IGNORECASE)


def estimate_tokens(text: str) -> int:
    """Cheap chars/4 proxy used when a provider reports no usage."""
    return math.ceil(len(str(text)) / 4)


def _extract_boxed(text: str) -> str:
    match = _BOXED.search(text)
    if not match:
        return text
    depth, start = 1, match.end()
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return text[start:]


def normalize_answer(text: Any) -> str:
    value = _extract_boxed(str(text)).strip()
    value = _TRAILING_PREFIX.sub("", value).strip()
    value = _FRAC.sub(r"\1/\2", value)
    value = _TEXTLIKE.sub(r"\1", value)
    for wrapper in _WRAPPERS:
        value = value.replace(wrapper, "")
    value = value.replace("\\%", "").replace("%", "")
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    return value.lower()


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        pass
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def exact_match(prediction: Any, task: Mapping[str, Any]) -> float:
    return float(str(prediction).strip().lower() == str(task.get("label", "")).strip().lower())


def math_equivalence(prediction: Any, task: Mapping[str, Any], tolerance: float = 1e-6) -> float:
    if "answer" not in task:
        return 0.0
    left, right = normalize_answer(prediction), normalize_answer(task["answer"])
    if left and left == right:
        return 1.0
    left_value, right_value = _as_float(left), _as_float(right)
    if left_value is not None and right_value is not None:
        return float(abs(left_value - right_value) <= tolerance * max(1.0, abs(right_value)))
    return 0.0


METRICS: dict[str, Callable[[Any, Mapping[str, Any]], float]] = {
    "classification": exact_match,
    "math": math_equivalence,
}

__all__ = ["METRICS", "estimate_tokens", "exact_match", "math_equivalence", "normalize_answer"]
```

Export `estimate_tokens`, `exact_match`, `math_equivalence`, `normalize_answer`, `METRICS`
from `meta_harness/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_metrics.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/metrics.py meta_harness/__init__.py tests/test_metrics.py
git commit -m "feat(metrics): add token estimation, exact match, and math answer equivalence"
```

---

### Task 3: Disk-backed model call cache

**Files:**
- Create: `meta_harness/cache.py`
- Modify: `meta_harness/__init__.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `meta_harness.providers` call protocol (`call(prompt, **kwargs) -> (text, usage)`).
- Produces:
  - `class DiskCache` with `get(key: str) -> dict | None`, `put(key: str, value: Mapping[str, Any]) -> None`
  - `class CachedModel` — callable wrapper exposing `call(prompt, **kwargs) -> tuple[str, dict]`, `__call__(prompt, **kwargs) -> str`, and attributes `hits: int`, `misses: int`
  - `cache_key(model_id: str, prompt: str, kwargs: Mapping[str, Any]) -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cache.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.cache'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/cache.py`:

```python
"""Disk cache for model calls. Repeats and re-runs must not re-bill the gateway."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Mapping


def cache_key(model_id: str, prompt: str, kwargs: Mapping[str, Any]) -> str:
    payload = json.dumps({"model": model_id, "prompt": prompt, "kwargs": kwargs},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """One JSON file per key, sharded by the first two hex characters."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def put(self, key: str, value: Mapping[str, Any]) -> None:
        path = self._path(key)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(dict(value), default=str), encoding="utf-8")
            temporary.replace(path)


class CachedModel:
    """Wraps any model callable, memoizing (model id, prompt, kwargs) -> (text, usage)."""

    def __init__(self, model: Callable[..., Any], root: Path | str, enabled: bool = True):
        self.model = model
        self.cache = DiskCache(root)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return str(getattr(self.model, "model", self.model.__class__.__name__))

    def _upstream(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        call = getattr(self.model, "call", None)
        if callable(call):
            text, usage = call(prompt, **kwargs)
            return str(text), dict(usage or {})
        return str(self.model(prompt, **kwargs)), {}

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            return self._upstream(prompt, **kwargs)
        key = cache_key(self.model_id, prompt, kwargs)
        stored = self.cache.get(key)
        if stored is not None:
            self.hits += 1
            return str(stored.get("text", "")), dict(stored.get("usage") or {})
        self.misses += 1
        text, usage = self._upstream(prompt, **kwargs)
        self.cache.put(key, {"text": text, "usage": usage})
        return text, usage

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]


__all__ = ["CachedModel", "DiskCache", "cache_key"]
```

Export `CachedModel`, `DiskCache` from `meta_harness/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cache.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/cache.py meta_harness/__init__.py tests/test_cache.py
git commit -m "feat(cache): add disk-backed model call cache"
```

---

### Task 4: Metered model — real token context cost, rich traces

Replaces the `trace.count` event-count fallback (`core.py:184`) with token accounting, and
makes every model call appear in the execution trace so the proposer can diagnose failures.

**Files:**
- Modify: `meta_harness/core.py`
- Test: `tests/test_core_metering.py`

**Interfaces:**
- Consumes: `meta_harness.metrics.estimate_tokens`, `CachedModel.call` protocol.
- Produces:
  - `class MeteredModel` in `core.py` — `__init__(model, trace)`, `call(prompt, **kwargs) -> tuple[str, dict]`, `__call__(prompt, **kwargs) -> str`
  - `CandidateEvaluator.__init__(model, metric=None, task_timeout: float = 600.0)`
  - `EvaluationResult.metrics` now always contains `{"model_calls": float, "prompt_tokens": float}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_metering.py`:

```python
import json
from pathlib import Path

from meta_harness.core import CandidateEvaluator, MeteredModel, TraceRecorder

HARNESS = '''
class Harness:
    def run(self, task, model, trace):
        return model("classify: " + task["input"])
'''


def test_metered_model_records_provider_usage(tmp_path: Path):
    class Upstream:
        model = "m"

        def call(self, prompt, **kwargs):
            return "ok", {"prompt_tokens": 123, "completion_tokens": 4}

    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        metered = MeteredModel(Upstream(), trace)
        assert metered("hello") == "ok"
        assert trace.context_cost == 123
    events = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "model_call"
    assert events[0]["payload"]["prompt"] == "hello"
    assert events[0]["payload"]["response"] == "ok"


def test_metered_model_estimates_when_usage_absent(tmp_path: Path):
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        metered = MeteredModel(lambda prompt, **_: "ok", trace)
        metered("a" * 400)
        assert trace.context_cost == 100


def test_evaluator_context_cost_is_tokens_not_events(tmp_path: Path):
    source = tmp_path / "harness.py"
    source.write_text(HARNESS)
    evaluator = CandidateEvaluator(lambda prompt, **_: "fruit")
    result = evaluator.evaluate(source, [{"input": "apple", "label": "fruit"}], "c0", tmp_path / "trace.jsonl")
    assert result.score == 1.0
    # "classify: apple" is 15 chars -> ceil(15/4) == 4 tokens.
    assert result.context_cost == 4.0
    assert result.metrics["model_calls"] == 1.0


def test_evaluator_enforces_task_timeout(tmp_path: Path):
    source = tmp_path / "slow.py"
    source.write_text('''
import time


class Harness:
    def run(self, task, model, trace):
        time.sleep(5)
        return "late"
''')
    evaluator = CandidateEvaluator(lambda prompt, **_: "x", task_timeout=0.5)
    result = evaluator.evaluate(source, [{"input": "a", "label": "a"}], "c1", tmp_path / "trace.jsonl")
    assert result.valid is False
    assert "timeout" in (result.error or "").lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core_metering.py -q`
Expected: FAIL — `ImportError: cannot import name 'MeteredModel'`

- [ ] **Step 3: Write minimal implementation**

In `meta_harness/core.py`, add the import and the class:

```python
from .metrics import estimate_tokens


class MeteredModel:
    """Wraps the base model so every call is priced and written into the trace."""

    def __init__(self, model: Callable[..., Any], trace: "TraceRecorder"):
        self.model = model
        self.trace = trace
        self.calls = 0

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        upstream = getattr(self.model, "call", None)
        if callable(upstream):
            text, usage = upstream(prompt, **kwargs)
            text, usage = str(text), dict(usage or {})
        else:
            text, usage = str(self.model(prompt, **kwargs)), {}
        tokens = int(usage.get("prompt_tokens") or 0) or estimate_tokens(prompt)
        self.calls += 1
        self.trace.add_context_cost(tokens)
        self.trace.event("model_call", {"prompt": prompt, "response": text, "tokens": tokens,
                                        "usage": usage, "kwargs": {k: v for k, v in kwargs.items()}})
        return text, usage

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]
```

Rewrite `CandidateEvaluator` so metering replaces the event-count hack and a wall-clock
deadline bounds each task:

```python
class CandidateEvaluator:
    def __init__(self, model: Callable[..., Any], metric: Callable[[Any, Any], float] | None = None,
                 task_timeout: float = 600.0):
        self.model = model
        self.metric = metric or (lambda prediction, task: float(prediction == task.get("label")))
        self.task_timeout = task_timeout

    def validate(self, source: Path | str) -> None:
        harness = _load_harness(Path(source))
        with tempfile.TemporaryDirectory() as td:
            with TraceRecorder(Path(td) / "validation.jsonl") as trace:
                try:
                    harness.run({"input": "validation", "label": "ok"}, MeteredModel(self.model, trace), trace)
                except Exception as exc:
                    raise HarnessValidationError(f"validation run failed: {exc}") from exc

    def evaluate(self, source: Path | str, tasks: Sequence[Mapping[str, Any]], candidate_id: str,
                 trace_path: Path) -> EvaluationResult:
        started = time.perf_counter()
        try:
            harness = _load_harness(Path(source))
            score_sum = 0.0
            tokens = 0.0
            calls = 0
            with TraceRecorder(trace_path) as trace:
                metered = MeteredModel(self.model, trace)
                for index, task in enumerate(tasks):
                    trace.event("task_start", {"index": index, "task": dict(task)})
                    task_started = time.perf_counter()
                    before = trace.context_cost
                    prediction = harness.run(task, metered, trace)
                    elapsed = time.perf_counter() - task_started
                    if elapsed > self.task_timeout:
                        raise TimeoutError(f"task {index} exceeded task_timeout ({elapsed:.1f}s)")
                    value = float(self.metric(prediction, task))
                    score_sum += value
                    tokens += trace.context_cost - before
                    trace.event("task_end", {"index": index, "prediction": prediction, "reward": value})
                calls = metered.calls
            n = len(tasks) or 1
            return EvaluationResult(candidate_id, score_sum / n,
                                    {"model_calls": float(calls), "prompt_tokens": tokens},
                                    tokens / n, len(tasks), time.perf_counter() - started)
        except Exception as exc:
            return EvaluationResult(candidate_id, float("-inf"), {}, float("inf"), len(tasks),
                                    time.perf_counter() - started, False, repr(exc))
```

Note on the deadline: it is checked after each task returns rather than interrupting mid-task.
That bounds a slow harness to one overrun task, which is the cheap correct-enough guard;
hard interruption is Task 5's subprocess validation. Add a marker comment above the check:

```python
                    # ponytail: post-hoc deadline check, not preemption; subprocess
                    # validation (see validate_in_subprocess) catches hard hangs first.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_core_metering.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Fix the existing suite**

`tests/test_core.py::test_end_to_end_search` passes `lambda p: p` as the model; `MeteredModel`
calls it with keyword arguments in some paths, so change that fixture model to
`lambda p, **_: p`. Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/core.py tests/test_core_metering.py tests/test_core.py
git commit -m "feat(core): price context cost in tokens and trace every model call"
```

---

### Task 5: Subprocess interface validation with enforced timeout

`SearchConfig.validation_timeout` is currently declared and never used, and `_load_harness`
executes LLM-written module code in-process. This task isolates validation.

**Files:**
- Create: `meta_harness/sandbox.py`
- Modify: `meta_harness/core.py`
- Test: `tests/test_sandbox.py`

**Interfaces:**
- Consumes: `HarnessValidationError` from `core.py`.
- Produces:
  - `validate_in_subprocess(source: Path, timeout: float) -> None` — raises `HarnessValidationError`
  - module entry point `python -m meta_harness.sandbox <source>` printing `OK` on success
- Note: `sandbox.py` must not import `core.py` at module import time (circular import). It
  defines its own local exception and `core.py` translates.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sandbox.py`:

```python
from pathlib import Path

import pytest

from meta_harness.core import HarnessValidationError
from meta_harness.sandbox import validate_in_subprocess


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_valid_harness_passes(tmp_path: Path):
    source = _write(tmp_path, "good.py", '''
class Harness:
    def run(self, task, model, trace):
        return model(task["input"])
''')
    validate_in_subprocess(source, timeout=30.0)


def test_build_harness_factory_passes(tmp_path: Path):
    source = _write(tmp_path, "factory.py", '''
class _H:
    def run(self, task, model, trace):
        return "x"


def build_harness():
    return _H()
''')
    validate_in_subprocess(source, timeout=30.0)


def test_missing_interface_fails(tmp_path: Path):
    source = _write(tmp_path, "bad.py", "VALUE = 1\n")
    with pytest.raises(HarnessValidationError, match="build_harness"):
        validate_in_subprocess(source, timeout=30.0)


def test_import_error_fails(tmp_path: Path):
    source = _write(tmp_path, "boom.py", "raise RuntimeError('nope')\n")
    with pytest.raises(HarnessValidationError, match="nope"):
        validate_in_subprocess(source, timeout=30.0)


def test_infinite_loop_is_killed(tmp_path: Path):
    source = _write(tmp_path, "hang.py", '''
class Harness:
    def run(self, task, model, trace):
        while True:
            pass
''')
    with pytest.raises(HarnessValidationError, match="timed out"):
        validate_in_subprocess(source, timeout=2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sandbox.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.sandbox'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/sandbox.py`:

```python
"""Out-of-process interface validation for LLM-written candidate harnesses.

Candidate code is untrusted in the "may hang or explode" sense, not the security sense:
it runs with the same privileges as the search. The subprocess exists so an infinite loop
or an interpreter-killing import cannot take the outer loop down with it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


class SandboxError(RuntimeError):
    pass


class _NullTrace:
    def __init__(self) -> None:
        self.count = 0
        self.context_cost = 0.0

    def add_context_cost(self, amount: float) -> None:
        self.context_cost += max(0.0, float(amount))

    def event(self, name: str, payload=None, **fields) -> None:
        self.count += 1


def _null_model(prompt: str, **_: object) -> str:
    return "validation-response"


def _check(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("meta_harness_sandbox_candidate", path)
    if spec is None or spec.loader is None:
        raise SandboxError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_harness"):
        harness = module.build_harness()
    elif hasattr(module, "Harness"):
        harness = module.Harness()
    else:
        raise SandboxError("candidate must expose build_harness() or Harness")
    if not callable(getattr(harness, "run", None)):
        raise SandboxError("harness must implement run(task, model, trace)")
    harness.run({"input": "validation", "label": "ok"}, _null_model, _NullTrace())


def validate_in_subprocess(source: Path | str, timeout: float = 20.0) -> None:
    """Import and smoke-run a candidate in a child process. Raises HarnessValidationError."""
    from .core import HarnessValidationError  # local import breaks the import cycle

    source = Path(source)
    try:
        completed = subprocess.run([sys.executable, "-m", "meta_harness.sandbox", str(source)],
                                   capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise HarnessValidationError(f"validation timed out after {timeout}s") from None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise HarnessValidationError(detail[-1] if detail else f"validation exited {completed.returncode}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m meta_harness.sandbox <candidate.py>", file=sys.stderr)
        return 2
    try:
        _check(Path(argv[1]))
    except Exception as exc:  # surfaced to the parent through stderr
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
```

In `core.py`, use it from the runner. Replace the `self.evaluator.validate(source)` call in
`SearchRunner._evaluate_source` with:

```python
        try:
            validate_in_subprocess(source, self.config.validation_timeout)
        except Exception as exc:
```

and add `from .sandbox import validate_in_subprocess` to `core.py`'s imports (placed at the
bottom of the import block; `sandbox` imports `core` lazily inside the function, so this
direction is safe).

Keep `CandidateEvaluator.validate` as-is for in-process use by tests and library callers.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sandbox.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/sandbox.py meta_harness/core.py tests/test_sandbox.py
git commit -m "feat(sandbox): validate candidates out-of-process under a timeout"
```

---

### Task 6: Repeats and concurrent candidate evaluation

**Files:**
- Modify: `meta_harness/core.py`
- Test: `tests/test_core_concurrency.py`

**Interfaces:**
- Consumes: `CandidateEvaluator.evaluate` from Task 4.
- Produces:
  - `SearchConfig` gains `repeats: int = 1`, `max_workers: int = 1`, `task_timeout: float = 600.0`
  - `EvaluationResult` gains `repeats: int = 1`, `score_std: float = 0.0`, `split: str = "search"`
  - `SearchRunner._evaluate_source(source, tasks, label, split="search") -> EvaluationResult | None`
  - `SearchRunner._evaluate_many(sources_and_labels: Sequence[tuple[Path, str]], tasks, split) -> list[EvaluationResult]`

Trace files for repeats are `traces.jsonl` (repeat 0) and `traces-1.jsonl`, `traces-2.jsonl`, …
so the proposer sees every rollout.

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_concurrency.py`:

```python
import threading
from pathlib import Path

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner

FLAKY = '''
import itertools

_COUNTER = itertools.count()


class Harness:
    def run(self, task, model, trace):
        return task["label"] if next(_COUNTER) % 2 == 0 else "wrong"
'''

SLEEPER = '''
import time


class Harness:
    def run(self, task, model, trace):
        time.sleep(0.4)
        return task["label"]
'''


def test_repeats_average_and_record_spread(tmp_path: Path):
    source = tmp_path / "flaky.py"
    source.write_text(FLAKY)
    config = SearchConfig(tmp_path / "store", iterations=0, repeats=2)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])
    frontier = runner.run([source], [{"input": "a", "label": "a"}])
    result = frontier[0] if frontier else runner.experience.results()[0]
    assert result.repeats == 2
    assert result.score == 0.5
    assert result.score_std > 0
    directory = runner.experience.candidate_dir(result.candidate_id)
    assert (directory / "traces.jsonl").exists()
    assert (directory / "traces-1.jsonl").exists()


def test_candidates_evaluate_concurrently(tmp_path: Path):
    import time

    sources = []
    for index in range(4):
        path = tmp_path / f"s{index}.py"
        path.write_text(SLEEPER)
        sources.append(path)
    config = SearchConfig(tmp_path / "store", iterations=0, max_workers=4)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])
    started = time.perf_counter()
    runner.run(sources, [{"input": "a", "label": "a"}])
    # 4 candidates x 0.4s serial would be >= 1.6s; concurrent should stay well under.
    assert time.perf_counter() - started < 1.4


def test_results_are_tagged_with_split(tmp_path: Path):
    source = tmp_path / "ok.py"
    source.write_text('''
class Harness:
    def run(self, task, model, trace):
        return task["label"]
''')
    config = SearchConfig(tmp_path / "store", iterations=0)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])
    runner.run([source], [{"input": "a", "label": "a"}])
    assert runner.experience.results()[0].split == "search"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core_concurrency.py -q`
Expected: FAIL — `TypeError: SearchConfig.__init__() got an unexpected keyword argument 'repeats'`

- [ ] **Step 3: Write minimal implementation**

Extend the dataclasses in `core.py`:

```python
@dataclass
class EvaluationResult:
    candidate_id: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    context_cost: float = 0.0
    examples: int = 0
    elapsed_seconds: float = 0.0
    valid: bool = True
    error: str | None = None
    repeats: int = 1
    score_std: float = 0.0
    split: str = "search"
```

```python
@dataclass
class SearchConfig:
    root: Path | str = Path(".meta-harness")
    iterations: int = 10
    candidates_per_iteration: int = 1
    validation_timeout: float = 20.0
    keep_invalid: bool = True
    maximize: str = "score"
    repeats: int = 1
    max_workers: int = 1
    task_timeout: float = 600.0

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.iterations < 0 or self.candidates_per_iteration < 1:
            raise ValueError("iterations must be non-negative and candidates_per_iteration positive")
        if self.repeats < 1 or self.max_workers < 1:
            raise ValueError("repeats and max_workers must be positive")
```

Add `import statistics` and `from concurrent.futures import ThreadPoolExecutor` to `core.py`.

Replace `SearchRunner._evaluate_source` and add `_evaluate_many`:

```python
    def _evaluate_source(self, source: Path, tasks: Sequence[Mapping[str, Any]], label: str,
                         split: str = "search") -> EvaluationResult | None:
        try:
            validate_in_subprocess(source, self.config.validation_timeout)
        except Exception as exc:
            candidate_id = self.experience.add_source(source, label)
            result = EvaluationResult(candidate_id, float("-inf"), valid=False, error=str(exc), split=split)
            self.experience.write_result(result)
            return result if self.config.keep_invalid else None
        candidate_id = self.experience.add_source(source, label)
        directory = self.experience.candidate_dir(candidate_id)
        harness_path = directory / "harness.py"
        runs: list[EvaluationResult] = []
        for repeat in range(self.config.repeats):
            trace_name = "traces.jsonl" if repeat == 0 else f"traces-{repeat}.jsonl"
            runs.append(self.evaluator.evaluate(harness_path, tasks, candidate_id, directory / trace_name))
        result = self._merge_repeats(candidate_id, runs, split)
        self.experience.write_result(result)
        return result

    @staticmethod
    def _merge_repeats(candidate_id: str, runs: Sequence[EvaluationResult], split: str) -> EvaluationResult:
        failed = next((run for run in runs if not run.valid), None)
        if failed is not None:
            return EvaluationResult(candidate_id, float("-inf"), {}, float("inf"), failed.examples,
                                    sum(run.elapsed_seconds for run in runs), False, failed.error,
                                    len(runs), 0.0, split)
        scores = [run.score for run in runs]
        merged_metrics: dict[str, float] = {}
        for key in {key for run in runs for key in run.metrics}:
            values = [run.metrics.get(key, 0.0) for run in runs]
            merged_metrics[key] = sum(values) / len(values)
        return EvaluationResult(
            candidate_id,
            sum(scores) / len(scores),
            merged_metrics,
            sum(run.context_cost for run in runs) / len(runs),
            runs[0].examples,
            sum(run.elapsed_seconds for run in runs),
            True,
            None,
            len(runs),
            statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            split,
        )

    def _evaluate_many(self, work: Sequence[tuple[Path, str]], tasks: Sequence[Mapping[str, Any]],
                       split: str = "search") -> list[EvaluationResult]:
        if self.config.max_workers == 1 or len(work) <= 1:
            return [r for r in (self._evaluate_source(s, tasks, label, split) for s, label in work) if r]
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(work))) as pool:
            futures = [pool.submit(self._evaluate_source, source, tasks, label, split) for source, label in work]
            return [r for r in (future.result() for future in futures) if r]
```

Rewrite `SearchRunner.run` to use `_evaluate_many` (test-split handling arrives in Task 7):

```python
    def run(self, initial: Sequence[Path | str], tasks: Sequence[Mapping[str, Any]]) -> list[EvaluationResult]:
        self._write_run_manifest(tasks)
        results = self._evaluate_many([(Path(s), f"initial-{i:04d}") for i, s in enumerate(initial)], tasks)
        for iteration in range(1, self.config.iterations + 1):
            try:
                proposals = self.proposer(self.experience, iteration, self.config.candidates_per_iteration)
            except Exception as exc:
                proposal_dir = self.experience.root / "proposals" / f"iteration-{iteration:04d}"
                proposal_dir.mkdir(parents=True, exist_ok=True)
                (proposal_dir / "proposer.error").write_text(repr(exc), encoding="utf-8")
                proposals = []
            work = [(Path(s), f"iteration-{iteration:04d}-{i:02d}") for i, s in enumerate(proposals)]
            results.extend(self._evaluate_many(work, tasks))
        frontier = ParetoFrontier.select(results)
        (self.experience.root / "frontier.json").write_text(
            json.dumps([asdict(x) for x in frontier], indent=2), encoding="utf-8")
        return frontier
```

Note: `FilesystemExperience.add_source` mutates the filesystem and is now reached from worker
threads. Give `FilesystemExperience` a `threading.Lock` created in `__init__` and hold it
around the directory-uniquifying loop and `mkdir` in `add_source`:

```python
        import threading  # at module top, not here
        ...
    def __init__(self, root: Path | str):
        ...
        self._lock = threading.Lock()

    def add_source(self, source, candidate_id=None):
        ...
        with self._lock:
            suffix = 1
            while directory.exists():
                directory = self.candidate_dir(f"{candidate_id}-{suffix}")
                suffix += 1
            candidate_id = directory.name
            directory.mkdir(parents=True, exist_ok=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_core_concurrency.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/core.py tests/test_core_concurrency.py
git commit -m "feat(core): add repeated evaluation and concurrent candidate scoring"
```

---

### Task 7: Search/test split with results the proposer cannot read

**Files:**
- Modify: `meta_harness/core.py`
- Modify: `meta_harness/datasets.py`
- Test: `tests/test_core_split.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `SearchRunner._evaluate_many` from Task 6.
- Produces:
  - `SearchConfig.test_root: Path | str | None = None` — resolved in `__post_init__` to
    `root.with_name(root.name + "-test")` when `None`
  - `SearchRunner.run(initial, search_tasks, test_tasks=None) -> list[EvaluationResult]`
  - `SearchRunner.evaluate_on_test(frontier, test_tasks) -> list[EvaluationResult]` — writes
    `<test_root>/test_results.json`, writes nothing under `config.root`
  - `split_tasks(tasks, search_fraction=0.7, seed=0) -> tuple[list[dict], list[dict]]` in `datasets.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_split.py`:

```python
import json
from pathlib import Path

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner

HARNESS = '''
class Harness:
    def run(self, task, model, trace):
        return task["input"]
'''


def _runner(tmp_path: Path) -> SearchRunner:
    config = SearchConfig(tmp_path / "store", iterations=0)
    return SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])


def test_test_results_land_outside_the_experience_root(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    runner = _runner(tmp_path)
    search = [{"input": "a", "label": "a"}]
    held_out = [{"input": "b", "label": "b"}]
    frontier = runner.run([source], search, test_tasks=held_out)
    assert frontier

    test_root = Path(runner.config.test_root)
    payload = json.loads((test_root / "test_results.json").read_text())
    assert payload["results"][0]["split"] == "test"
    assert payload["results"][0]["score"] == 1.0

    # Nothing under the experience root may mention the test split.
    for path in runner.experience.root.rglob("*"):
        if path.is_file():
            assert "\"split\": \"test\"" not in path.read_text(encoding="utf-8", errors="ignore")


def test_no_test_tasks_means_no_test_file(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    runner = _runner(tmp_path)
    runner.run([source], [{"input": "a", "label": "a"}])
    assert not (Path(runner.config.test_root) / "test_results.json").exists()


def test_run_manifest_records_only_search_tasks(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    runner = _runner(tmp_path)
    runner.run([source], [{"input": "a", "label": "a"}], test_tasks=[{"input": "b", "label": "b"}])
    manifest = json.loads((runner.experience.root / "run.json").read_text())
    assert manifest["task_count"] == 1
    assert manifest["tasks"] == [{"input": "a", "label": "a"}]
    assert "test_task_count" in manifest and manifest["test_task_count"] == 1
    assert "b" not in json.dumps(manifest["tasks"])
```

Create `tests/test_datasets.py`:

```python
import pytest

from meta_harness.datasets import split_tasks


def test_split_is_deterministic_and_disjoint():
    tasks = [{"input": str(i), "label": "x"} for i in range(10)]
    a_search, a_test = split_tasks(tasks, 0.7, seed=1)
    b_search, b_test = split_tasks(tasks, 0.7, seed=1)
    assert a_search == b_search and a_test == b_test
    assert len(a_search) == 7 and len(a_test) == 3
    assert not [t for t in a_search if t in a_test]


def test_split_rejects_degenerate_fractions():
    tasks = [{"input": "1", "label": "x"}, {"input": "2", "label": "x"}]
    with pytest.raises(ValueError):
        split_tasks(tasks, 0.0)
    with pytest.raises(ValueError):
        split_tasks(tasks, 1.0)


def test_split_keeps_every_task():
    tasks = [{"input": str(i), "label": "x"} for i in range(9)]
    search, test = split_tasks(tasks, 0.5, seed=3)
    assert len(search) + len(test) == 9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_core_split.py tests/test_datasets.py -q`
Expected: FAIL — `ImportError: cannot import name 'split_tasks'` and
`TypeError: run() got an unexpected keyword argument 'test_tasks'`

- [ ] **Step 3: Write minimal implementation**

In `meta_harness/datasets.py` add:

```python
import random


def split_tasks(tasks: Sequence[Mapping[str, Any]], search_fraction: float = 0.7,
                seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle deterministically, then cut into a search split and a held-out test split."""
    if not 0.0 < search_fraction < 1.0:
        raise ValueError("search_fraction must be strictly between 0 and 1")
    items = [dict(task) for task in tasks]
    random.Random(seed).shuffle(items)
    cut = max(1, min(len(items) - 1, round(len(items) * search_fraction)))
    return items[:cut], items[cut:]
```

(add `Sequence` to the `typing` import there, and `Any` is already imported).

In `core.py`, resolve `test_root` in `SearchConfig.__post_init__`:

```python
        self.test_root = Path(self.test_root) if self.test_root else self.root.with_name(self.root.name + "-test")
```

with the field declared as `test_root: Path | str | None = None`.

Add the test evaluation path to `SearchRunner`. It deliberately bypasses
`FilesystemExperience` so nothing lands under `config.root`:

```python
    def evaluate_on_test(self, frontier: Sequence[EvaluationResult],
                         test_tasks: Sequence[Mapping[str, Any]]) -> list[EvaluationResult]:
        """Score the frontier once on held-out tasks. Paper §3: the proposer never sees this."""
        test_root = Path(self.config.test_root)
        test_root.mkdir(parents=True, exist_ok=True)
        results: list[EvaluationResult] = []
        for item in frontier:
            harness_path = self.experience.candidate_dir(item.candidate_id) / "harness.py"
            trace_path = test_root / "traces" / f"{item.candidate_id}.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            runs = [self.evaluator.evaluate(harness_path, test_tasks, item.candidate_id,
                                            trace_path if repeat == 0 else
                                            trace_path.with_name(f"{item.candidate_id}-{repeat}.jsonl"))
                    for repeat in range(self.config.repeats)]
            results.append(self._merge_repeats(item.candidate_id, runs, "test"))
        payload = {"search_root": str(self.experience.root), "results": [asdict(r) for r in results]}
        (test_root / "test_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return results
```

Change `run` to accept and use the test split:

```python
    def run(self, initial: Sequence[Path | str], search_tasks: Sequence[Mapping[str, Any]],
            test_tasks: Sequence[Mapping[str, Any]] | None = None) -> list[EvaluationResult]:
        self._write_run_manifest(search_tasks, test_tasks)
        results = self._evaluate_many([(Path(s), f"initial-{i:04d}") for i, s in enumerate(initial)], search_tasks)
        for iteration in range(1, self.config.iterations + 1):
            try:
                proposals = self.proposer(self.experience, iteration, self.config.candidates_per_iteration)
            except Exception as exc:
                proposal_dir = self.experience.root / "proposals" / f"iteration-{iteration:04d}"
                proposal_dir.mkdir(parents=True, exist_ok=True)
                (proposal_dir / "proposer.error").write_text(repr(exc), encoding="utf-8")
                proposals = []
            work = [(Path(s), f"iteration-{iteration:04d}-{i:02d}") for i, s in enumerate(proposals)]
            results.extend(self._evaluate_many(work, search_tasks))
        frontier = ParetoFrontier.select(results)
        (self.experience.root / "frontier.json").write_text(
            json.dumps([asdict(x) for x in frontier], indent=2), encoding="utf-8")
        if test_tasks:
            self.evaluate_on_test(frontier, test_tasks)
        return frontier
```

And record only search tasks in the manifest:

```python
    def _write_run_manifest(self, tasks: Sequence[Mapping[str, Any]],
                            test_tasks: Sequence[Mapping[str, Any]] | None = None) -> None:
        manifest = {
            "created_at": time.time(),
            "python": sys.version,
            "config": {"root": str(self.config.root), "iterations": self.config.iterations,
                       "candidates_per_iteration": self.config.candidates_per_iteration,
                       "keep_invalid": self.config.keep_invalid, "repeats": self.config.repeats,
                       "max_workers": self.config.max_workers},
            "task_count": len(tasks),
            # Only the search split is written here. Test tasks stay out of the proposer's reach.
            "tasks": [dict(task) for task in tasks],
            "test_task_count": len(test_tasks or []),
        }
        (self.experience.root / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
```

Export `split_tasks` from `meta_harness/__init__.py`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_core_split.py tests/test_datasets.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/core.py meta_harness/datasets.py meta_harness/__init__.py tests/test_core_split.py tests/test_datasets.py
git commit -m "feat(core): hold out a test split the proposer never sees"
```

---

### Task 8: Link proposer reasoning into candidate directories

Paper Figure 2 stores reasoning traces alongside code and scores. Today the proposer's stdout
lands in `proposals/iteration-NNNN/proposer.stdout`, unlinked from the candidate it produced.

**Files:**
- Modify: `meta_harness/core.py`
- Test: `tests/test_core_reasoning.py`

**Interfaces:**
- Consumes: `FilesystemExperience.add_source` from Task 6.
- Produces:
  - `FilesystemExperience.add_source(source, candidate_id=None, notes: Mapping[str, str] | None = None) -> str`
  - Sidecar convention: a proposal at `X.py` may be accompanied by `X.reasoning.md`; the runner
    copies it into the candidate directory as `proposer_reasoning.md`
  - `SearchRunner._sidecar_notes(source: Path) -> dict[str, str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_core_reasoning.py`:

```python
from pathlib import Path

from meta_harness.core import CandidateEvaluator, FilesystemExperience, SearchConfig, SearchRunner

HARNESS = '''
class Harness:
    def run(self, task, model, trace):
        return task["label"]
'''


def test_add_source_writes_notes(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    experience = FilesystemExperience(tmp_path / "store")
    candidate_id = experience.add_source(source, "c0", notes={"proposer_reasoning.md": "because X"})
    assert (experience.candidate_dir(candidate_id) / "proposer_reasoning.md").read_text() == "because X"


def test_runner_copies_reasoning_sidecar(tmp_path: Path):
    proposal = tmp_path / "prop.py"
    proposal.write_text(HARNESS)
    (tmp_path / "prop.reasoning.md").write_text("isolated the prompt change from the retrieval change")

    def proposer(experience, iteration, count):
        return [proposal]

    config = SearchConfig(tmp_path / "store", iterations=1)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), proposer)
    runner.run([], [{"input": "a", "label": "a"}])
    directory = runner.experience.candidate_dir("iteration-0001-00")
    assert "isolated the prompt change" in (directory / "proposer_reasoning.md").read_text()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_core_reasoning.py -q`
Expected: FAIL — `TypeError: add_source() got an unexpected keyword argument 'notes'`

- [ ] **Step 3: Write minimal implementation**

In `FilesystemExperience.add_source`, add the parameter and write the notes after copying:

```python
    def add_source(self, source: Path | str, candidate_id: str | None = None,
                   notes: Mapping[str, str] | None = None) -> str:
        ...
        shutil.copy2(source, directory / "harness.py")
        for name, body in (notes or {}).items():
            safe = Path(name).name  # never let a proposal write outside its own directory
            (directory / safe).write_text(str(body), encoding="utf-8")
        (directory / "proposal.json").write_text(json.dumps({"candidate_id": candidate_id}, indent=2), encoding="utf-8")
        return candidate_id
```

In `SearchRunner`, add the sidecar reader and pass notes through both `add_source` calls in
`_evaluate_source`:

```python
    @staticmethod
    def _sidecar_notes(source: Path) -> dict[str, str]:
        sidecar = source.with_suffix("").with_suffix(".reasoning.md") if source.suffix == ".py" else None
        candidates = [source.with_name(source.stem + ".reasoning.md")]
        if sidecar is not None:
            candidates.append(sidecar)
        for path in candidates:
            if path.is_file():
                return {"proposer_reasoning.md": path.read_text(encoding="utf-8", errors="replace")}
        return {}
```

and in `_evaluate_source`:

```python
        notes = self._sidecar_notes(source)
        ...
            candidate_id = self.experience.add_source(source, label, notes)   # invalid branch
        ...
        candidate_id = self.experience.add_source(source, label, notes)       # valid branch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_core_reasoning.py -q`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/core.py tests/test_core_reasoning.py
git commit -m "feat(core): store proposer reasoning next to each candidate"
```

---

### Task 9: Seed baseline harnesses on disk

Six runnable seed files. Classification: zero-shot, few-shot(N), ACE, MCE. Math: zero-shot and
BM25 retrieval. Each is a standalone single-file harness with no `meta_harness` import except
the math BM25 one, which may import `meta_harness.retrieval` (the paper's §4.2 baselines share
the same lexical stack).

**Files:**
- Create: `baselines/zero_shot.py`
- Create: `baselines/few_shot.py`
- Create: `baselines/ace.py`
- Create: `baselines/mce.py`
- Create: `baselines/math_zero_shot.py`
- Create: `baselines/math_bm25.py`
- Create: `baselines/README.md`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `meta_harness.retrieval.BM25Index` (math BM25 only).
- Produces: each file exposes `class Harness` with `run(task, model, trace) -> str`.
  Tunables read from the environment with defaults:
  `META_HARNESS_FEW_SHOT_N` (default `8`), `META_HARNESS_ACE_BULLETS` (default `24`),
  `META_HARNESS_MCE_SKILLS` (default `12`), `META_HARNESS_RETRIEVAL_K` (default `3`),
  `META_HARNESS_CORPUS` (path to a JSONL retrieval corpus for the math baselines).

- [ ] **Step 1: Write the failing test**

Create `tests/test_baselines.py`:

```python
import json
from pathlib import Path

import pytest

from meta_harness.core import TraceRecorder
from meta_harness.sandbox import validate_in_subprocess

BASELINES = ["zero_shot", "few_shot", "ace", "mce", "math_zero_shot", "math_bm25"]
CLASSIFICATION = ["zero_shot", "few_shot", "ace", "mce"]


def _load(name: str):
    import importlib.util

    path = Path("baselines") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"baseline_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Harness()


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_passes_interface_validation(name):
    validate_in_subprocess(Path("baselines") / f"{name}.py", timeout=60.0)


@pytest.mark.parametrize("name", CLASSIFICATION)
def test_classification_baseline_predicts_a_declared_label(tmp_path: Path, name):
    harness = _load(name)
    tasks = [
        {"input": "an apple", "label": "fruit", "labels": ["fruit", "vehicle"]},
        {"input": "a fast car", "label": "vehicle", "labels": ["fruit", "vehicle"]},
    ]
    model = lambda prompt, **_: "vehicle" if "car" in prompt.rsplit("Input:", 1)[-1] else "fruit"
    with TraceRecorder(tmp_path / f"{name}.jsonl") as trace:
        predictions = [harness.run(task, model, trace) for task in tasks]
    assert predictions == ["fruit", "vehicle"]


def test_few_shot_respects_n(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("META_HARNESS_FEW_SHOT_N", "2")
    harness = _load("few_shot")
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return "a"

    with TraceRecorder(tmp_path / "fs.jsonl") as trace:
        for index in range(5):
            harness.run({"input": f"text {index}", "label": "a", "labels": ["a", "b"]}, model, trace)
    assert prompts[-1].count("Example:") <= 2


def test_math_bm25_retrieves_from_corpus(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(x) for x in [
        {"problem": "Find the angle in a triangle", "solution": "Angles sum to 180"},
        {"problem": "Show that 7 is prime", "solution": "Check divisors"},
    ]), encoding="utf-8")
    monkeypatch.setenv("META_HARNESS_CORPUS", str(corpus))
    harness = _load("math_bm25")
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return r"\boxed{60}"

    with TraceRecorder(tmp_path / "m.jsonl") as trace:
        answer = harness.run({"problem": "What is the angle of an equilateral triangle?", "answer": "60"}, model, trace)
    assert answer == r"\boxed{60}"
    assert "Angles sum to 180" in prompts[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_baselines.py -q`
Expected: FAIL — `FileNotFoundError: baselines/zero_shot.py`

- [ ] **Step 3: Write the baselines**

`baselines/zero_shot.py`:

```python
"""Zero-shot classification baseline (paper Table 2, "Zero-Shot")."""


def _labels(task, memory):
    declared = task.get("labels")
    if declared:
        return [str(x) for x in declared]
    seen = sorted({label for _, label in memory})
    if "label" in task and str(task["label"]) not in seen:
        seen = sorted(seen + [str(task["label"])])
    return seen


def _pick(output, labels):
    text = str(output).strip()
    for label in labels:
        if label.lower() in text.lower():
            return label
    return text.splitlines()[0].strip() if text else (labels[0] if labels else text)


class Harness:
    def __init__(self):
        self.memory = []

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        prompt = ("Classify the input.\nValid labels: " + ", ".join(labels) +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("zero_shot_prompt", {"prompt": prompt, "labels": len(labels)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            self.memory.append((query, str(task["label"])))
        trace.event("zero_shot_result", {"prediction": prediction})
        return prediction
```

`baselines/few_shot.py`:

```python
"""Few-shot classification baseline. N from META_HARNESS_FEW_SHOT_N (default 8)."""

import os


def _labels(task, memory):
    declared = task.get("labels")
    if declared:
        return [str(x) for x in declared]
    seen = sorted({label for _, label in memory})
    if "label" in task and str(task["label"]) not in seen:
        seen = sorted(seen + [str(task["label"])])
    return seen


def _pick(output, labels):
    text = str(output).strip()
    for label in labels:
        if label.lower() in text.lower():
            return label
    return text.splitlines()[0].strip() if text else (labels[0] if labels else text)


class Harness:
    def __init__(self, shots=None):
        self.shots = int(shots if shots is not None else os.environ.get("META_HARNESS_FEW_SHOT_N", "8"))
        self.memory = []

    def _examples(self):
        if self.shots <= 0:
            return []
        # Most recent first keeps the window fresh in an online stream.
        return list(reversed(self.memory))[: self.shots]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        examples = self._examples()
        block = "\n".join(f"Example: {text}\nLabel: {label}" for text, label in examples)
        prompt = ("Classify the input.\nValid labels: " + ", ".join(labels) + "\n\n" + block +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("few_shot_prompt", {"prompt": prompt, "shots": len(examples)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            self.memory.append((query, str(task["label"])))
        trace.event("few_shot_result", {"prediction": prediction})
        return prediction
```

`baselines/ace.py`:

```python
"""Agentic Context Engineering baseline: a reflectively curated bullet playbook.

Follows Zhang et al. (paper ref [59]) in shape: predict, then on an error ask the model to
write one durable bullet, keeping a bounded deduplicated playbook.
"""

import os


def _labels(task, memory):
    declared = task.get("labels")
    if declared:
        return [str(x) for x in declared]
    seen = sorted({label for _, label in memory})
    if "label" in task and str(task["label"]) not in seen:
        seen = sorted(seen + [str(task["label"])])
    return seen


def _pick(output, labels):
    text = str(output).strip()
    for label in labels:
        if label.lower() in text.lower():
            return label
    return text.splitlines()[0].strip() if text else (labels[0] if labels else text)


class Harness:
    def __init__(self, max_bullets=None):
        self.max_bullets = int(max_bullets if max_bullets is not None
                               else os.environ.get("META_HARNESS_ACE_BULLETS", "24"))
        self.playbook = []
        self.memory = []

    def _reflect(self, query, prediction, truth, labels, model, trace):
        prompt = ("A classifier made a mistake.\n"
                  f"Input: {query}\nPredicted: {prediction}\nCorrect: {truth}\n"
                  "Write ONE short reusable rule (max 20 words) that would prevent this mistake. "
                  "Do not mention this specific input.\nRule:")
        bullet = str(model(prompt)).strip().splitlines()[0].strip(" -*")
        trace.event("ace_reflection", {"bullet": bullet})
        if bullet and bullet not in self.playbook:
            self.playbook.append(bullet)
            del self.playbook[: max(0, len(self.playbook) - self.max_bullets)]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        rules = "\n".join(f"- {bullet}" for bullet in self.playbook)
        prompt = ("Classify the input using the playbook.\nValid labels: " + ", ".join(labels) +
                  (f"\n\nPlaybook:\n{rules}" if rules else "") +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("ace_prompt", {"prompt": prompt, "bullets": len(self.playbook)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            truth = str(task["label"])
            if prediction != truth:
                self._reflect(query, prediction, truth, labels, model, trace)
            self.memory.append((query, truth))
        trace.event("ace_result", {"prediction": prediction})
        return prediction
```

`baselines/mce.py`:

```python
"""Meta Context Engineering baseline: an evolving library of natural-language skills.

Follows Ye et al. (paper ref [52]) in shape: a small library of named skills, each a short
construction recipe; the model selects a skill, applies it, and the library grows on failure.
"""

import os


def _labels(task, memory):
    declared = task.get("labels")
    if declared:
        return [str(x) for x in declared]
    seen = sorted({label for _, label in memory})
    if "label" in task and str(task["label"]) not in seen:
        seen = sorted(seen + [str(task["label"])])
    return seen


def _pick(output, labels):
    text = str(output).strip()
    for label in labels:
        if label.lower() in text.lower():
            return label
    return text.splitlines()[0].strip() if text else (labels[0] if labels else text)


class Harness:
    def __init__(self, max_skills=None):
        self.max_skills = int(max_skills if max_skills is not None
                              else os.environ.get("META_HARNESS_MCE_SKILLS", "12"))
        self.skills = []
        self.memory = []

    def _evolve(self, query, prediction, truth, model, trace):
        prompt = ("You maintain a library of classification skills. A prediction was wrong.\n"
                  f"Input: {query}\nPredicted: {prediction}\nCorrect: {truth}\n"
                  "Write one skill as 'NAME: recipe' where the recipe is at most 25 words and "
                  "describes what evidence to look for.\nSkill:")
        skill = str(model(prompt)).strip().splitlines()[0].strip(" -*")
        trace.event("mce_skill", {"skill": skill})
        if skill and skill not in self.skills:
            self.skills.append(skill)
            del self.skills[: max(0, len(self.skills) - self.max_skills)]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        library = "\n".join(f"{index + 1}. {skill}" for index, skill in enumerate(self.skills))
        prompt = ("Classify the input. Pick the most relevant skill from the library, then apply it.\n"
                  "Valid labels: " + ", ".join(labels) +
                  (f"\n\nSkill library:\n{library}" if library else "") +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("mce_prompt", {"prompt": prompt, "skills": len(self.skills)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            truth = str(task["label"])
            if prediction != truth:
                self._evolve(query, prediction, truth, model, trace)
            self.memory.append((query, truth))
        trace.event("mce_result", {"prediction": prediction})
        return prediction
```

`baselines/math_zero_shot.py`:

```python
"""No-retrieval math baseline (paper Table 6, "No Retriever")."""


class Harness:
    def run(self, task, model, trace):
        problem = str(task.get("problem", task.get("input", "")))
        prompt = ("Solve the problem rigorously. End with the final answer inside \\boxed{}.\n\n"
                  f"Problem:\n{problem}\n\nSolution:")
        trace.event("math_prompt", {"prompt": prompt})
        answer = str(model(prompt)).strip()
        trace.event("math_result", {"answer": answer})
        return answer
```

`baselines/math_bm25.py`:

```python
"""BM25 retrieval math baseline (paper Table 6, "BM25 Retrieval").

Corpus path comes from META_HARNESS_CORPUS (JSONL with "problem" and "solution" fields).
With no corpus configured the harness degrades to zero-shot, which keeps it valid.
"""

import json
import os

from meta_harness.retrieval import BM25Index


def _load_corpus():
    path = os.environ.get("META_HARNESS_CORPUS", "")
    if not path or not os.path.isfile(path):
        return []
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("problem") and record.get("solution"):
                items.append(record)
    return items


class Harness:
    def __init__(self, k=None):
        self.k = int(k if k is not None else os.environ.get("META_HARNESS_RETRIEVAL_K", "3"))
        self.corpus = _load_corpus()
        self.index = BM25Index(self.corpus, [str(x["problem"]) for x in self.corpus],
                               math_mode=True) if self.corpus else None

    def run(self, task, model, trace):
        problem = str(task.get("problem", task.get("input", "")))
        retrieved = self.index.search(problem, self.k) if self.index else []
        context = "\n\n".join(
            f"Reference problem:\n{x.item['problem']}\nSolution:\n{str(x.item['solution'])[:3000]}"
            for x in retrieved)
        prompt = ("Solve the problem rigorously. Reference solutions may help. "
                  "End with the final answer inside \\boxed{}.\n\n" +
                  (context + "\n\n" if context else "") + f"Problem:\n{problem}\n\nSolution:")
        trace.event("math_retrieval", {"retrieved": [x.index for x in retrieved], "k": self.k})
        answer = str(model(prompt)).strip()
        trace.event("math_result", {"answer": answer})
        return answer
```

`baselines/README.md`:

```markdown
# Seed harnesses

Starting population for `meta-harness run --baseline ...`. Each file exposes
`class Harness` with `run(task, model, trace)`.

| File | Domain | Paper reference | Tunable |
|---|---|---|---|
| `zero_shot.py` | classification | Table 2, Zero-Shot | — |
| `few_shot.py` | classification | Table 2, Few-Shot (N) | `META_HARNESS_FEW_SHOT_N` (8) |
| `ace.py` | classification | Table 2, ACE [59] | `META_HARNESS_ACE_BULLETS` (24) |
| `mce.py` | classification | Table 2, MCE [52] | `META_HARNESS_MCE_SKILLS` (12) |
| `math_zero_shot.py` | math | Table 6, No Retriever | — |
| `math_bm25.py` | math | Table 6, BM25 Retrieval | `META_HARNESS_CORPUS`, `META_HARNESS_RETRIEVAL_K` (3) |

These are seeds, not frozen references: the proposer is expected to rewrite them.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_baselines.py -q`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add baselines tests/test_baselines.py
git commit -m "feat(baselines): add classification and math seed harnesses"
```

---

### Task 10: Coding-agent proposer, skill file, and proposer-view ablation

The paper's core contribution. Also implements the Table 3 ablation, since the ablation is
just a filter on what the proposer's working directory contains.

**Files:**
- Create: `meta_harness/agent_proposer.py`
- Create: `meta_harness/skill/SKILL.md`
- Modify: `meta_harness/__init__.py`
- Modify: `pyproject.toml` (package the skill data file)
- Test: `tests/test_agent_proposer.py`

**Interfaces:**
- Consumes: `FilesystemExperience` (Tasks 6/8), the `.reasoning.md` sidecar convention (Task 8).
- Produces:
  - `SKILL_PATH: Path` — absolute path to the bundled `SKILL.md`
  - `build_view(experience, mode: str, destination: Path, summarizer=None) -> Path`
    with `mode in {"scores", "summary", "full"}`
  - `class ClaudeCodeProposer` — `__init__(binary="claude", model=None, view="full", summarizer=None, timeout=1800.0, extra_args=(), skill_path=None)`, `__call__(experience, iteration, count) -> list[Path]`
  - `PROPOSER_PROMPT: str`
- Environment passed to the agent: `META_HARNESS_ROOT`, `META_HARNESS_OUTPUT`,
  `META_HARNESS_ITERATION`, `META_HARNESS_COUNT`, `META_HARNESS_VIEW`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_agent_proposer.py`:

```python
import json
import os
import sys
from pathlib import Path

from meta_harness.agent_proposer import SKILL_PATH, ClaudeCodeProposer, build_view
from meta_harness.core import EvaluationResult, FilesystemExperience

HARNESS = "class Harness:\n    def run(self, task, model, trace):\n        return 'x'\n"


def _experience(tmp_path: Path) -> FilesystemExperience:
    experience = FilesystemExperience(tmp_path / "store")
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    candidate_id = experience.add_source(source, "initial-0000")
    experience.write_result(EvaluationResult(candidate_id, 0.5, context_cost=100.0))
    directory = experience.candidate_dir(candidate_id)
    (directory / "traces.jsonl").write_text(json.dumps({"event": "model_call"}) + "\n", encoding="utf-8")
    return experience


def test_skill_file_is_bundled_and_describes_the_contract():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "META_HARNESS_OUTPUT" in text
    assert "run(task, model, trace)" in text
    assert ".reasoning.md" in text


def test_full_view_returns_the_root_itself(tmp_path: Path):
    experience = _experience(tmp_path)
    view = build_view(experience, "full", tmp_path / "view")
    assert view == experience.root


def test_scores_view_hides_traces(tmp_path: Path):
    experience = _experience(tmp_path)
    view = build_view(experience, "scores", tmp_path / "view")
    candidate = next((view / "candidates").iterdir())
    assert (candidate / "harness.py").exists()
    assert (candidate / "scores.json").exists()
    assert not (candidate / "traces.jsonl").exists()


def test_summary_view_adds_summaries_but_still_hides_traces(tmp_path: Path):
    experience = _experience(tmp_path)
    view = build_view(experience, "summary", tmp_path / "view",
                      summarizer=lambda prompt, **_: "it mispredicted rare labels")
    candidate = next((view / "candidates").iterdir())
    assert not (candidate / "traces.jsonl").exists()
    assert "mispredicted" in (candidate / "summary.md").read_text()


def test_proposer_invokes_the_binary_and_collects_candidates(tmp_path: Path):
    experience = _experience(tmp_path)
    fake = tmp_path / "fake_agent.py"
    fake.write_text('''
import os
from pathlib import Path

out = Path(os.environ["META_HARNESS_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
(out / "candidate-00.py").write_text("class Harness:\\n    def run(self, task, model, trace):\\n        return 'y'\\n")
(out / "candidate-00.reasoning.md").write_text("tried a wider label primer")
print("done")
''', encoding="utf-8")

    proposer = ClaudeCodeProposer(binary=sys.executable, extra_args=(str(fake),), prompt_as_argument=False)
    paths = proposer(experience, iteration=1, count=1)
    assert [p.name for p in paths] == ["candidate-00.py"]
    assert paths[0].with_name("candidate-00.reasoning.md").is_file()
    stdout = (experience.root / "proposals" / "iteration-0001" / "proposer.stdout").read_text()
    assert "done" in stdout


def test_proposer_records_failure_without_raising_into_the_loop(tmp_path: Path):
    experience = _experience(tmp_path)
    proposer = ClaudeCodeProposer(binary=sys.executable, extra_args=("-c", "raise SystemExit(3)"),
                                  prompt_as_argument=False)
    try:
        proposer(experience, iteration=2, count=1)
    except RuntimeError as exc:
        assert "3" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert (experience.root / "proposals" / "iteration-0002" / "proposer.stderr").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_proposer.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.agent_proposer'`

- [ ] **Step 3: Write the skill file**

Create `meta_harness/skill/SKILL.md`:

```markdown
---
name: meta-harness-proposer
description: Propose the next candidate harness by reading the Meta-Harness experience filesystem.
---

# Meta-Harness proposer

You are the proposer in a harness-search loop. Your job this iteration: read the accumulated
experience, form a hypothesis about why the current best harness fails, and write new
candidate harness code that tests that hypothesis.

## Where you are

Your working directory is a view of the experience filesystem. Environment variables:

- `META_HARNESS_ROOT` — the view root (same as your working directory).
- `META_HARNESS_OUTPUT` — the ONLY directory you may write candidates into.
- `META_HARNESS_ITERATION` — current iteration number.
- `META_HARNESS_COUNT` — how many candidates to produce.
- `META_HARNESS_VIEW` — `full`, `summary`, or `scores`. In non-`full` modes some files are
  deliberately absent; do not go looking for them elsewhere.

## Layout

```
run.json                      search config and the search-split tasks
frontier.json                 Pareto frontier from the last completed run, if any
candidates/<id>/harness.py    that candidate's full source
candidates/<id>/scores.json   score, context_cost, repeats, score_std, valid, error
candidates/<id>/traces.jsonl  one JSON object per line: model_call, task_start, task_end, ...
candidates/<id>/traces-N.jsonl  additional repeats
candidates/<id>/proposer_reasoning.md  why the previous proposer wrote that candidate
candidates/<id>/summary.md    present only in `summary` view mode
proposals/iteration-NNNN/     previous proposer stdout/stderr
```

## How to read it

The filesystem is far larger than your context. Query it, do not ingest it.

```bash
# rank every candidate by score
for f in candidates/*/scores.json; do python -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"{d['score']:.3f} {d['context_cost']:.0f} {d['candidate_id']} {d.get('error') or ''}\")" "$f"; done | sort -rn

# what did the best candidate actually send to the model?
grep -h '"event": "model_call"' candidates/<id>/traces.jsonl | head -3

# which tasks did it get wrong?
grep -h '"event": "task_end"' candidates/<id>/traces.jsonl | grep '"reward": 0'

# what did the previous proposer think it was doing?
cat candidates/<id>/proposer_reasoning.md
```

Read broadly before editing. Compare a winner's trace against a loser's trace on the same
task. Prefer a hypothesis you can point at a trace line for.

## What to write

Write exactly `META_HARNESS_COUNT` Python files into `$META_HARNESS_OUTPUT`, named
`candidate-00.py`, `candidate-01.py`, … Next to each, write `candidate-NN.reasoning.md`
containing: the failure you diagnosed, the file and trace lines that support it, the change
you made, and what score movement would confirm or refute it. That file is stored with the
candidate and the next proposer will read it.

Each candidate is a **single self-contained Python file** exposing:

```python
class Harness:
    def run(self, task, model, trace):
        """Return the prediction for one task.

        task  - dict. Classification: {"input", "label"?, "labels"?}.
                       Math: {"problem", "answer"?}.
                       The label is absent at prediction time in some setups; never depend on it
                       to produce the prediction, only to update memory afterwards.
        model - callable. model(prompt) -> str. Also model.call(prompt) -> (str, usage).
                Every call is priced and logged; fewer and shorter prompts score better on
                context cost.
        trace - trace.event(name, payload_dict) writes a line to traces.jsonl. Log the prompts
                and decisions you would want to read next iteration.
        """
```

Alternatively expose `def build_harness():` returning such an object.

The instance persists across the tasks of one evaluation, so `self` is your memory between
tasks. It is recreated for each evaluation, so it does not persist across candidates.

Allowed imports: the Python standard library, and `meta_harness.retrieval`
(`TfidfIndex`, `BM25Index`, `reciprocal_rank_fusion`). Nothing else — there are no
third-party packages installed.

## Rules

- Write ONLY into `$META_HARNESS_OUTPUT`. Never modify `candidates/`, `run.json`,
  `frontier.json`, the `meta_harness` package, the baselines, or the tests.
- Never hard-code answers, label lists, or dataset-specific strings copied from `run.json`
  tasks. That is scored as overfitting and audited.
- Do not try to read a test split. There isn't one here; it is stored outside this tree.
- Do not print or store API keys.
- Optimize two objectives: reward, and context cost (input tokens). A candidate that matches
  the incumbent score with fewer tokens is a real win and lands on the Pareto frontier.
- Your candidate must survive a smoke run against a stub model that returns a fixed string
  for any prompt. Never crash on an empty `labels`, an empty memory, or an unexpected model
  reply.
```

- [ ] **Step 4: Write the proposer**

Create `meta_harness/agent_proposer.py`:

```python
"""Coding-agent proposer: the paper's §3 design, plus the Table 3 view ablation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Sequence

from .core import FilesystemExperience

SKILL_PATH = Path(__file__).resolve().parent / "skill" / "SKILL.md"

VIEW_MODES = ("scores", "summary", "full")

PROPOSER_PROMPT = (
    "Read SKILL.md in this directory and follow it exactly. You are the Meta-Harness proposer "
    "at iteration $META_HARNESS_ITERATION. Inspect the experience filesystem in this directory "
    "(candidates/, run.json, frontier.json), diagnose why the leading harnesses fail, then write "
    "$META_HARNESS_COUNT candidate harness file(s) plus their .reasoning.md sidecars into "
    "$META_HARNESS_OUTPUT. Do not modify anything else."
)

_SUMMARY_PROMPT = (
    "Summarize this harness evaluation for an engineer who will try to improve it. "
    "Cover: what the harness does, its score and context cost, and the failure pattern in the "
    "trace. Maximum 150 words.\n\nSource:\n{source}\n\nScores:\n{scores}\n\nTrace excerpt:\n{trace}"
)


def _copy_view(experience: FilesystemExperience, destination: Path, keep: Sequence[str]) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidates").mkdir(exist_ok=True)
    for name in ("run.json", "frontier.json"):
        source = experience.root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for directory in sorted(experience.candidates.iterdir()):
        if not directory.is_dir():
            continue
        target = destination / "candidates" / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in keep:
            source = directory / name
            if source.is_file():
                shutil.copy2(source, target / name)
    return destination


def _trace_excerpt(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return "(no trace)"
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(line for _, line in zip(range(lines), handle))


def build_view(experience: FilesystemExperience, mode: str, destination: Path,
               summarizer: Callable[..., str] | None = None) -> Path:
    """Materialize what the proposer is allowed to read. Paper Table 3 ablation."""
    if mode not in VIEW_MODES:
        raise ValueError(f"view must be one of {VIEW_MODES}")
    if mode == "full":
        return experience.root
    view = _copy_view(experience, Path(destination), ("harness.py", "scores.json", "proposal.json"))
    if mode == "summary":
        if summarizer is None:
            raise ValueError("summary view requires a summarizer model")
        for directory in sorted(experience.candidates.iterdir()):
            if not directory.is_dir():
                continue
            prompt = _SUMMARY_PROMPT.format(
                source=(directory / "harness.py").read_text(encoding="utf-8", errors="replace")[:6000],
                scores=(directory / "scores.json").read_text(encoding="utf-8", errors="replace")
                if (directory / "scores.json").is_file() else "{}",
                trace=_trace_excerpt(directory / "traces.jsonl"))
            (view / "candidates" / directory.name / "summary.md").write_text(
                str(summarizer(prompt)), encoding="utf-8")
    return view


class ClaudeCodeProposer:
    """Runs a coding-agent CLI over the experience filesystem.

    Defaults target Claude Code:
        claude -p "<prompt>" --permission-mode acceptEdits [--model M]
    Point it at Unikey by exporting, before the run:
        ANTHROPIC_BASE_URL=https://www.getunikey.ai
        ANTHROPIC_AUTH_TOKEN=$UNIKEY_API_KEY
    """

    def __init__(self, binary: str = "claude", model: str | None = None, view: str = "full",
                 summarizer: Callable[..., str] | None = None, timeout: float = 1800.0,
                 extra_args: Sequence[str] = (), skill_path: Path | str | None = None,
                 permission_mode: str = "acceptEdits", prompt_as_argument: bool = True):
        if view not in VIEW_MODES:
            raise ValueError(f"view must be one of {VIEW_MODES}")
        self.binary = binary
        self.model = model
        self.view = view
        self.summarizer = summarizer
        self.timeout = timeout
        self.extra_args = list(extra_args)
        self.skill_path = Path(skill_path) if skill_path else SKILL_PATH
        self.permission_mode = permission_mode
        self.prompt_as_argument = prompt_as_argument

    def _command(self) -> list[str]:
        if not self.prompt_as_argument:
            return [self.binary, *self.extra_args]
        command = [self.binary, "-p", PROPOSER_PROMPT, "--permission-mode", self.permission_mode]
        if self.model:
            command += ["--model", self.model]
        return command + self.extra_args

    def __call__(self, experience: FilesystemExperience, iteration: int, count: int) -> list[Path]:
        output = experience.root / "proposals" / f"iteration-{iteration:04d}"
        output.mkdir(parents=True, exist_ok=True)
        view = build_view(experience, self.view,
                          experience.root / "views" / f"iteration-{iteration:04d}", self.summarizer)
        shutil.copy2(self.skill_path, view / "SKILL.md")
        env = os.environ.copy()
        env.update({
            "META_HARNESS_ROOT": str(Path(view).resolve()),
            "META_HARNESS_OUTPUT": str(output.resolve()),
            "META_HARNESS_ITERATION": str(iteration),
            "META_HARNESS_COUNT": str(count),
            "META_HARNESS_VIEW": self.view,
        })
        completed = subprocess.run(self._command(), cwd=view, env=env, text=True,
                                   capture_output=True, timeout=self.timeout)
        (output / "proposer.stdout").write_text(completed.stdout or "", encoding="utf-8")
        (output / "proposer.stderr").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"proposer exited with {completed.returncode}")
        candidates = sorted(output.glob("*.py"))[:count]
        for path in candidates:
            sidecar = path.with_name(path.stem + ".reasoning.md")
            if not sidecar.is_file():
                sidecar.write_text(completed.stdout or "(no reasoning recorded)", encoding="utf-8")
        return candidates


__all__ = ["PROPOSER_PROMPT", "SKILL_PATH", "VIEW_MODES", "ClaudeCodeProposer", "build_view"]
```

Note that `views/` lives under the experience root, so a `scores`/`summary` view is itself
visible to a later `full` run. That is fine — views only ever contain a subset of what the
root already holds.

Package the skill by adding to `pyproject.toml`:

```toml
[tool.setuptools.package-data]
meta_harness = ["skill/*.md"]
```

Export `ClaudeCodeProposer`, `build_view`, `SKILL_PATH` from `meta_harness/__init__.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_proposer.py -q`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add meta_harness/agent_proposer.py meta_harness/skill/SKILL.md meta_harness/__init__.py pyproject.toml tests/test_agent_proposer.py
git commit -m "feat(proposer): add coding-agent proposer with skill file and view ablation"
```

---

### Task 11: CLI — wire the whole run together

**Files:**
- Modify: `meta_harness/__main__.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1–10.
- Produces: `meta_harness.__main__.main(argv: Sequence[str] | None = None) -> int`

Subcommands:

```
meta-harness demo    [--iterations N] [--root DIR]
meta-harness inspect [ROOT]
meta-harness models  --provider unikey                 # GET /v1/models
meta-harness run     --tasks FILE
                     [--test-tasks FILE | --search-fraction F] [--split-seed N]
                     [--baseline FILE ...]            # defaults to the bundled seeds
                     --provider {unikey,openai,anthropic,compatible}
                     --model ID
                     [--task-type {classification,math}]
                     [--root DIR] [--test-root DIR]
                     [--iterations N] [--candidates N] [--repeats N] [--max-workers N]
                     [--proposer-command CMD | --proposer-model ID]
                     [--proposer-view {scores,summary,full}]
                     [--cache-dir DIR] [--no-cache]
                     [--validation-timeout S] [--task-timeout S]
```

Defaults: `--task-type classification` → baselines `zero_shot, few_shot, ace, mce`;
`--task-type math` → `math_zero_shot, math_bm25`. `--search-fraction` default `0.7`.
Cache default `<root>/../.meta-harness-cache`. Proposer defaults to `ClaudeCodeProposer`;
`--proposer-command` switches to `CommandProposer`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import json
import sys
from pathlib import Path

import pytest

from meta_harness import __main__ as cli


def test_run_end_to_end_with_a_stub_proposer(tmp_path: Path, monkeypatch, capsys):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"input": text, "label": label}) for text, label in [
        ("an apple", "fruit"), ("a pear", "fruit"), ("a fast car", "vehicle"),
        ("a red truck", "vehicle"), ("a banana", "fruit"), ("a blue van", "vehicle"),
    ]), encoding="utf-8")

    agent = tmp_path / "agent.py"
    agent.write_text('''
import os
from pathlib import Path

out = Path(os.environ["META_HARNESS_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
(out / "candidate-00.py").write_text(
    "class Harness:\\n    def run(self, task, model, trace):\\n        return model(task['input'])\\n")
(out / "candidate-00.reasoning.md").write_text("echo the input through the model")
''', encoding="utf-8")

    monkeypatch.setattr(cli, "model_from_environment",
                        lambda provider, model, **kw: (lambda prompt, **_: "fruit" if "apple" in prompt or "pear" in prompt or "banana" in prompt else "vehicle"))

    code = cli.main([
        "run", "--tasks", str(tasks), "--provider", "unikey", "--model", "gpt-5.2",
        "--root", str(tmp_path / "store"), "--iterations", "1",
        "--proposer-command", f"{sys.executable} {agent}",
        "--search-fraction", "0.5",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frontier"]
    assert Path(payload["test_results"]).is_file()


def test_run_defaults_to_bundled_baselines(tmp_path: Path, monkeypatch):
    parsed = cli.build_parser().parse_args([
        "run", "--tasks", "t.jsonl", "--provider", "unikey", "--model", "m"])
    assert parsed.baseline is None
    assert [p.name for p in cli.default_baselines("classification")] == [
        "zero_shot.py", "few_shot.py", "ace.py", "mce.py"]
    assert [p.name for p in cli.default_baselines("math")] == [
        "math_zero_shot.py", "math_bm25.py"]


def test_models_subcommand_lists_ids(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_unikey_models", lambda **kw: ["claude-sonnet-4-6", "gpt-5.2"])
    assert cli.main(["models", "--provider", "unikey"]) == 0
    assert "gpt-5.2" in capsys.readouterr().out


def test_explicit_test_tasks_file_is_used(tmp_path: Path, monkeypatch):
    search = tmp_path / "s.jsonl"
    search.write_text(json.dumps({"input": "a", "label": "a"}), encoding="utf-8")
    held = tmp_path / "t.jsonl"
    held.write_text(json.dumps({"input": "b", "label": "b"}), encoding="utf-8")
    monkeypatch.setattr(cli, "model_from_environment", lambda *a, **k: (lambda prompt, **_: "a"))
    code = cli.main([
        "run", "--tasks", str(search), "--test-tasks", str(held),
        "--provider", "unikey", "--model", "m", "--root", str(tmp_path / "store"),
        "--iterations", "0", "--baseline", "baselines/zero_shot.py"])
    assert code == 0
    assert (Path(str(tmp_path / "store") + "-test") / "test_results.json").is_file()


def test_demo_still_runs(tmp_path: Path, capsys):
    assert cli.main(["demo", "--iterations", "1", "--root", str(tmp_path / "demo")]) == 0
    assert json.loads(capsys.readouterr().out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli.py -q`
Expected: FAIL — `AttributeError: module 'meta_harness.__main__' has no attribute 'build_parser'`

- [ ] **Step 3: Write minimal implementation**

Rewrite `meta_harness/__main__.py`:

```python
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

from .agent_proposer import ClaudeCodeProposer
from .cache import CachedModel
from .core import (CandidateEvaluator, CommandProposer, FilesystemExperience, ParetoFrontier,
                   SearchConfig, SearchRunner)
from .datasets import classification_tasks, math_tasks, read_records, split_tasks
from .demo import run as run_demo
from .metrics import METRICS
from .providers import list_unikey_models, model_from_environment

BASELINE_ROOT = Path(__file__).resolve().parent.parent / "baselines"
DEFAULT_BASELINES = {
    "classification": ["zero_shot.py", "few_shot.py", "ace.py", "mce.py"],
    "math": ["math_zero_shot.py", "math_bm25.py"],
}


def default_baselines(task_type: str) -> list[Path]:
    return [BASELINE_ROOT / name for name in DEFAULT_BASELINES[task_type]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meta-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a deterministic end-to-end example")
    demo.add_argument("--iterations", type=int, default=3)
    demo.add_argument("--root", default=".meta-harness-demo")

    inspect = sub.add_parser("inspect", help="show stored results and Pareto frontier")
    inspect.add_argument("root", nargs="?", default=".meta-harness")

    models = sub.add_parser("models", help="list models offered by the gateway")
    models.add_argument("--provider", choices=["unikey"], default="unikey")

    run = sub.add_parser("run", help="run a provider-backed harness search")
    run.add_argument("--tasks", required=True, help="JSON, JSONL, or CSV dataset")
    run.add_argument("--test-tasks", help="held-out dataset; omit to split --tasks")
    run.add_argument("--search-fraction", type=float, default=0.7)
    run.add_argument("--split-seed", type=int, default=0)
    run.add_argument("--baseline", nargs="+", help="seed harness files (default: bundled baselines)")
    run.add_argument("--provider", choices=["unikey", "openai", "anthropic", "compatible"], default="unikey")
    run.add_argument("--model", required=True)
    run.add_argument("--task-type", choices=["classification", "math"], default="classification")
    run.add_argument("--root", default=".meta-harness")
    run.add_argument("--test-root")
    run.add_argument("--iterations", type=int, default=10)
    run.add_argument("--candidates", type=int, default=1)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--max-workers", type=int, default=1)
    run.add_argument("--proposer-command", help="shell command writing .py files to META_HARNESS_OUTPUT")
    run.add_argument("--proposer-model", help="model id passed to the coding agent")
    run.add_argument("--proposer-binary", default="claude")
    run.add_argument("--proposer-view", choices=["scores", "summary", "full"], default="full")
    run.add_argument("--proposer-timeout", type=float, default=1800.0)
    run.add_argument("--cache-dir")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--validation-timeout", type=float, default=20.0)
    run.add_argument("--task-timeout", type=float, default=600.0)
    return parser


def _load_tasks(args) -> tuple[list[dict], list[dict]]:
    adapt = classification_tasks if args.task_type == "classification" else math_tasks
    search = adapt(read_records(args.tasks))
    if args.test_tasks:
        return search, adapt(read_records(args.test_tasks))
    if len(search) < 2:
        return search, []
    return split_tasks(search, args.search_fraction, args.split_seed)


def _command_run(args) -> int:
    search_tasks, test_tasks = _load_tasks(args)
    base_model = model_from_environment(args.provider, args.model)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.root).parent / ".meta-harness-cache"
    model = CachedModel(base_model, cache_dir, enabled=not args.no_cache)
    if args.proposer_command:
        proposer = CommandProposer(shlex.split(args.proposer_command), timeout=args.proposer_timeout)
    else:
        proposer = ClaudeCodeProposer(binary=args.proposer_binary, model=args.proposer_model,
                                      view=args.proposer_view, summarizer=model,
                                      timeout=args.proposer_timeout)
    config = SearchConfig(root=args.root, iterations=args.iterations,
                          candidates_per_iteration=args.candidates,
                          validation_timeout=args.validation_timeout,
                          repeats=args.repeats, max_workers=args.max_workers,
                          task_timeout=args.task_timeout, test_root=args.test_root)
    evaluator = CandidateEvaluator(model, METRICS[args.task_type], task_timeout=args.task_timeout)
    runner = SearchRunner(config, evaluator, proposer)
    baselines = [Path(p) for p in (args.baseline or default_baselines(args.task_type))]
    frontier = runner.run(baselines, search_tasks, test_tasks or None)
    test_path = Path(config.test_root) / "test_results.json"
    print(json.dumps({
        "root": str(config.root),
        "search_tasks": len(search_tasks),
        "test_tasks": len(test_tasks),
        "cache": {"hits": model.hits, "misses": model.misses},
        "frontier": [item.__dict__ for item in frontier],
        "test_results": str(test_path) if test_path.is_file() else None,
    }, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "demo":
        root = run_demo(args.iterations, args.root)
        print(json.dumps([r.__dict__ for r in ParetoFrontier.select(FilesystemExperience(root).results())], indent=2))
        return 0
    if args.command == "models":
        print("\n".join(list_unikey_models()))
        return 0
    if args.command == "run":
        return _command_run(args)
    experience = FilesystemExperience(args.root)
    frontier_file = experience.root / "frontier.json"
    print(json.dumps({
        "results": experience.manifest(),
        "frontier": json.loads(frontier_file.read_text(encoding="utf-8")) if frontier_file.exists() else [],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Also add a console script to `pyproject.toml`:

```toml
[project.scripts]
meta-harness = "meta_harness.__main__:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_cli.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/__main__.py pyproject.toml tests/test_cli.py
git commit -m "feat(cli): wire splits, cache, metrics, and the agent proposer into one run command"
```

---

### Task 12: Agentic coding domain — TerminalHarness and task runner

`terminal.py` has a bounded agent loop that nothing uses. This exposes it as a searchable
harness (paper §4.3).

**Files:**
- Create: `meta_harness/terminal_harness.py`
- Modify: `meta_harness/datasets.py`
- Modify: `meta_harness/metrics.py`
- Modify: `meta_harness/__main__.py`
- Modify: `meta_harness/__init__.py`
- Create: `baselines/terminal_basic.py`
- Test: `tests/test_terminal_harness.py`

**Interfaces:**
- Consumes: `TerminalAgent`, `ShellPolicy`, `run_command` from `terminal.py`.
- Produces:
  - `terminal_tasks(records) -> list[dict]` in `datasets.py`, fields
    `{"instruction": str, "test_command": str, "image": str | None, "workdir": str | None, "timeout": float}`
  - `class DockerPolicy(ShellPolicy)` in `terminal_harness.py` — wraps each command as
    `docker exec -w <workdir> <container> bash -lc <command>`
  - `start_container(image: str, workdir: str) -> str` / `stop_container(name: str) -> None`
  - `class TerminalHarness` — `__init__(bootstrap="", max_steps=40, allow_local_shell=False)`,
    `run(task, model, trace) -> str`
  - `terminal_metric(prediction, task) -> float` added to `metrics.METRICS["terminal"]`
  - CLI: `--task-type terminal`, `--allow-local-shell`

- [ ] **Step 1: Write the failing test**

Create `tests/test_terminal_harness.py`:

```python
import json
from pathlib import Path

import pytest

from meta_harness.core import TraceRecorder
from meta_harness.datasets import terminal_tasks
from meta_harness.metrics import METRICS
from meta_harness.terminal_harness import DockerPolicy, TerminalHarness


def test_terminal_tasks_adapter_defaults():
    tasks = terminal_tasks([{"instruction": "make a file", "test_command": "test -f out.txt"}])
    assert tasks[0]["instruction"] == "make a file"
    assert tasks[0]["image"] is None
    assert tasks[0]["timeout"] == 300.0


def test_terminal_tasks_requires_an_instruction():
    with pytest.raises(ValueError):
        terminal_tasks([{"test_command": "true"}])


def test_docker_policy_wraps_commands():
    policy = DockerPolicy(container="c1", workdir="/app")
    wrapped = policy.wrap("ls -la")
    assert wrapped[:4] == ["docker", "exec", "-w", "/app"]
    assert wrapped[4] == "c1"
    assert wrapped[-1] == "ls -la"


def test_terminal_harness_refuses_local_shell_by_default(tmp_path: Path):
    harness = TerminalHarness()
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        with pytest.raises(PermissionError, match="allow_local_shell"):
            harness.run({"instruction": "echo hi", "image": None}, lambda p, **_: "{}", trace)


def test_terminal_harness_runs_locally_when_opted_in(tmp_path: Path):
    replies = iter(['{"command": "echo ready", "done": false}', '{"done": true, "answer": "finished"}'])
    harness = TerminalHarness(allow_local_shell=True, max_steps=4)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        answer = harness.run({"instruction": "say ready", "workdir": str(tmp_path)},
                             lambda prompt, **_: next(replies), trace)
    assert answer == "finished"
    events = [json.loads(line)["event"] for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert "terminal_step" in events


def test_terminal_metric_runs_the_test_command(tmp_path: Path):
    metric = METRICS["terminal"]
    passing = {"test_command": f"{'cd'} {tmp_path!s} && exit 0", "workdir": str(tmp_path)}
    failing = {"test_command": "exit 1", "workdir": str(tmp_path)}
    assert metric("done", passing) == 1.0
    assert metric("done", failing) == 0.0


def test_terminal_metric_without_command_is_zero():
    assert METRICS["terminal"]("done", {"instruction": "x"}) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_terminal_harness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.terminal_harness'`

- [ ] **Step 3: Write minimal implementation**

Add to `meta_harness/datasets.py`:

```python
def terminal_tasks(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        instruction = record.get("instruction") or record.get("input")
        if not instruction:
            raise ValueError("terminal records need 'instruction'")
        result.append({
            "instruction": str(instruction),
            "test_command": str(record.get("test_command", "")),
            "image": str(record["image"]) if record.get("image") else None,
            "workdir": str(record["workdir"]) if record.get("workdir") else None,
            "timeout": float(record.get("timeout", 300.0)),
        })
    return result
```

Create `meta_harness/terminal_harness.py`:

```python
"""Agentic-coding domain (paper §4.3): the bounded terminal agent as a searchable harness.

Commands run inside a Docker container by default. Running model-authored commands on the
host is a real risk, so it requires an explicit opt-in.
"""

from __future__ import annotations

import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .terminal import ShellPolicy, TerminalAgent, TerminalResult


@dataclass
class DockerPolicy(ShellPolicy):
    container: str = ""
    workdir: str = "/app"

    def wrap(self, command: str) -> list[str]:
        return ["docker", "exec", "-w", self.workdir, self.container, "bash", "-lc", command]


def start_container(image: str, workdir: str = "/app") -> str:
    name = f"meta-harness-{uuid.uuid4().hex[:10]}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-w", workdir,
                    image, "sleep", "infinity"], check=True, capture_output=True, text=True)
    return name


def stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)


def _run_in_docker(command: str, policy: DockerPolicy) -> tuple[int, str]:
    policy.check(command)
    try:
        completed = subprocess.run(policy.wrap(command), capture_output=True, text=True,
                                   timeout=policy.timeout_seconds)
        return completed.returncode, (completed.stdout + completed.stderr)[-policy.max_output_chars:]
    except subprocess.TimeoutExpired as error:
        return 124, (str(error.stdout or "")[-policy.max_output_chars:]) + "\n[command timed out]"


class TerminalHarness:
    """A searchable harness around TerminalAgent. The proposer edits bootstrap and step budget."""

    def __init__(self, bootstrap: str = "", max_steps: int = 40, allow_local_shell: bool = False,
                 step_timeout: float = 30.0):
        self.bootstrap = bootstrap or (
            "You are an autonomous terminal agent. Reply ONLY with JSON: "
            '{"command": "<shell command>", "done": false} or {"done": true, "answer": "<summary>"}.')
        self.max_steps = max_steps
        self.allow_local_shell = allow_local_shell
        self.step_timeout = step_timeout

    def run(self, task: Mapping[str, Any], model: Callable[..., Any], trace: Any) -> str:
        image = task.get("image")
        workdir = task.get("workdir") or "/app"
        if not image and not self.allow_local_shell:
            raise PermissionError(
                "no image given and allow_local_shell is False; refusing to run model-authored "
                "commands on the host")
        container = start_container(str(image), workdir) if image else None
        try:
            if container:
                policy = DockerPolicy(timeout_seconds=self.step_timeout, max_steps=self.max_steps,
                                      container=container, workdir=workdir)
                runner = lambda command: _run_in_docker(command, policy)
            else:
                policy = ShellPolicy(timeout_seconds=self.step_timeout, max_steps=self.max_steps,
                                     cwd=task.get("workdir"))
                from .terminal import run_command
                runner = lambda command: run_command(command, policy)
            agent = TerminalAgent(model, policy, self.bootstrap)
            agent_result = _drive(agent, policy, runner, task, trace)
            return agent_result.answer
        finally:
            if container:
                stop_container(container)


def _drive(agent: TerminalAgent, policy: ShellPolicy, runner, task: Mapping[str, Any],
           trace: Any) -> TerminalResult:
    """TerminalAgent.run with a pluggable command runner (host or container)."""
    import json

    instruction = str(task.get("instruction", task.get("input", "")))
    history: list[dict[str, Any]] = []
    for step in range(policy.max_steps):
        prompt = agent.bootstrap + "\n\nTask:\n" + instruction + (
            "\n\nExecution history:\n" + json.dumps(history, ensure_ascii=False) if history else "")
        action = agent.parse_action(str(agent.model(prompt)))
        if bool(action.get("done")):
            result = TerminalResult(True, str(action.get("answer", "")), step + 1, history)
            trace.event("terminal_complete", result.__dict__)
            return result
        command = str(action.get("command", ""))
        if command:
            try:
                code, observation = runner(command)
            except PermissionError as exc:
                code, observation = 126, f"blocked: {exc}"
        else:
            code, observation = 2, "No command was supplied. Return a command or set done=true."
        item = {"step": step + 1, "action": action, "returncode": code, "observation": observation}
        history.append(item)
        trace.event("terminal_step", item)
    trace.event("terminal_limit", {"max_steps": policy.max_steps})
    return TerminalResult(False, "step limit reached", policy.max_steps, history)


__all__ = ["DockerPolicy", "TerminalHarness", "start_container", "stop_container"]
```

Add the metric to `meta_harness/metrics.py`:

```python
def terminal_metric(prediction: Any, task: Mapping[str, Any], timeout: float = 120.0) -> float:
    """Run the task's verification command. Empty command scores zero, never one."""
    import subprocess

    command = str(task.get("test_command", "")).strip()
    if not command:
        return 0.0
    image, workdir = task.get("image"), task.get("workdir")
    argv = (["docker", "exec", "-w", str(workdir or "/app"), str(task["container"]), "bash", "-lc", command]
            if task.get("container") else command)
    try:
        completed = subprocess.run(argv, shell=not task.get("container"), capture_output=True,
                                   text=True, timeout=timeout,
                                   cwd=None if task.get("container") else (workdir or None))
    except (OSError, subprocess.SubprocessError):
        return 0.0
    return float(completed.returncode == 0)
```

and register it: `METRICS["terminal"] = terminal_metric`, plus add `"terminal_metric"` to
`__all__`.

Create `baselines/terminal_basic.py`:

```python
"""Seed terminal harness (paper §4.3 initialization, Terminus-2 shape)."""

from meta_harness.terminal_harness import TerminalHarness

BOOTSTRAP = """You are an autonomous terminal agent solving one task without human help.

Reply ONLY with a single JSON object, no prose:
  {"command": "<one shell command>", "done": false}
  {"done": true, "answer": "<one-line summary of what you did>"}

Guidance:
- Inspect before you change: ls, cat, and the project's own test command first.
- One command per turn. Chain with && only when the steps are inseparable.
- Verify your work before declaring done.
"""


def build_harness():
    return TerminalHarness(bootstrap=BOOTSTRAP, max_steps=40)
```

Wire the CLI: add `"terminal"` to `--task-type` choices, add `--allow-local-shell`, map it in
`_load_tasks` via `terminal_tasks`, add `DEFAULT_BASELINES["terminal"] = ["terminal_basic.py"]`,
and when `args.task_type == "terminal"` and `args.allow_local_shell` set
`os.environ["META_HARNESS_ALLOW_LOCAL_SHELL"] = "1"` — with `TerminalHarness.__init__`
defaulting `allow_local_shell` from that variable:

```python
        self.allow_local_shell = (allow_local_shell or
                                  os.environ.get("META_HARNESS_ALLOW_LOCAL_SHELL") == "1")
```

(add `import os` to `terminal_harness.py`).

Export `TerminalHarness`, `DockerPolicy` from `meta_harness/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_terminal_harness.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/terminal_harness.py meta_harness/datasets.py meta_harness/metrics.py meta_harness/__main__.py meta_harness/__init__.py baselines/terminal_basic.py tests/test_terminal_harness.py
git commit -m "feat(terminal): add searchable terminal harness with docker isolation"
```

---

### Task 13: Documentation and a no-network end-to-end smoke test

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `tests/test_smoke_end_to_end.py`

**Interfaces:**
- Consumes: the CLI from Task 11.
- Produces: no new code interfaces; a regression gate over the assembled system.

- [ ] **Step 1: Write the failing test**

Create `tests/test_smoke_end_to_end.py`:

```python
import json
import sys
from pathlib import Path

from meta_harness import __main__ as cli

AGENT = '''
import json
import os
from pathlib import Path

root = Path(os.environ["META_HARNESS_ROOT"])
out = Path(os.environ["META_HARNESS_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)

# The proposer must actually be able to read prior experience.
scores = sorted((root / "candidates").glob("*/scores.json"))
assert scores, "no prior candidates visible to the proposer"
best = max(json.loads(p.read_text())["score"] for p in scores)
traces = sorted((root / "candidates").glob("*/traces.jsonl"))
assert traces, "no execution traces visible to the proposer"

(out / "candidate-00.py").write_text(
    "class Harness:\\n"
    "    def run(self, task, model, trace):\\n"
    "        labels = task.get('labels') or ['fruit', 'vehicle']\\n"
    "        reply = model('labels: ' + ', '.join(labels) + ' input: ' + task['input'])\\n"
    "        for label in labels:\\n"
    "            if label in reply:\\n"
    "                return label\\n"
    "        return labels[0]\\n")
(out / "candidate-00.reasoning.md").write_text(f"best prior score was {best}; forcing a declared label")
'''


def test_full_loop_produces_frontier_traces_reasoning_and_test_results(tmp_path: Path, monkeypatch, capsys):
    tasks = tmp_path / "tasks.jsonl"
    rows = [("an apple", "fruit"), ("a pear", "fruit"), ("a plum", "fruit"),
            ("a fast car", "vehicle"), ("a red truck", "vehicle"), ("a blue van", "vehicle")]
    tasks.write_text("\n".join(json.dumps({"input": t, "label": l, "labels": ["fruit", "vehicle"]})
                               for t, l in rows), encoding="utf-8")

    agent = tmp_path / "agent.py"
    agent.write_text(AGENT, encoding="utf-8")

    fruit_words = ("apple", "pear", "plum", "banana")
    monkeypatch.setattr(cli, "model_from_environment", lambda *a, **k: (
        lambda prompt, **_: "fruit" if any(w in prompt for w in fruit_words) else "vehicle"))

    root = tmp_path / "store"
    assert cli.main([
        "run", "--tasks", str(tasks), "--provider", "unikey", "--model", "gpt-5.2",
        "--root", str(root), "--iterations", "1", "--candidates", "1", "--repeats", "2",
        "--max-workers", "2", "--search-fraction", "0.5",
        "--proposer-command", f"{sys.executable} {agent}",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["frontier"], "search produced no frontier"

    candidates = sorted((root / "candidates").iterdir())
    assert len(candidates) >= 5, "four baselines plus one proposal expected"
    proposed = root / "candidates" / "iteration-0001-00"
    assert (proposed / "proposer_reasoning.md").is_file()
    assert (proposed / "traces.jsonl").is_file()
    assert (proposed / "traces-1.jsonl").is_file()

    scores = json.loads((proposed / "scores.json").read_text())
    assert scores["repeats"] == 2
    assert scores["context_cost"] > 0, "context cost must be measured in tokens"

    test_results = json.loads(Path(payload["test_results"]).read_text())
    assert all(item["split"] == "test" for item in test_results["results"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_smoke_end_to_end.py -q`
Expected: FAIL before Tasks 1–11 land; after them it should pass on the first run. If it fails
here, fix the code, not the test.

- [ ] **Step 3: Update the README**

Replace the README with (keeping any existing sections that still describe live code):

````markdown
# Meta-Harness

Filesystem-backed end-to-end optimization of executable LLM harnesses, following
*Meta-Harness: End-to-End Optimization of Model Harnesses* (`paper.pdf`).

A **harness** is the code around a fixed model: what it stores, retrieves, and shows the model
at each step. This repository searches over that code. A coding-agent proposer reads the full
experience filesystem — every prior candidate's source, scores, execution traces, and the
reasoning that produced it — and writes new candidates. Candidates are evaluated on a search
split; a Pareto frontier over (accuracy, context tokens) is scored once on a held-out split
the proposer never sees.

## Install

```bash
python -m venv venv && venv/Scripts/activate      # Windows
pip install -e .
pip install pytest                                 # tests only
```

No runtime dependencies. Python 3.10+.

## Configure the model gateway (Unikey)

All model traffic goes through [Unikey](https://www.getunikey.ai), an OpenAI-compatible
gateway (`https://www.getunikey.ai/v1`).

```bash
export UNIKEY_API_KEY=sk-...          # PowerShell: $env:UNIKEY_API_KEY="sk-..."
meta-harness models --provider unikey  # GET /v1/models
```

The proposer is a coding agent, billed separately. To route Claude Code through Unikey's
Anthropic-compatible endpoint:

```bash
export ANTHROPIC_BASE_URL=https://www.getunikey.ai
export ANTHROPIC_AUTH_TOKEN=$UNIKEY_API_KEY
```

## Run a search

```bash
meta-harness run \
  --tasks data/classification.jsonl \
  --task-type classification \
  --provider unikey --model gpt-5.2 \
  --iterations 20 --candidates 2 --repeats 3 --max-workers 4 \
  --proposer-model claude-sonnet-4-6 \
  --root .meta-harness
```

Dataset formats: `.jsonl`, `.json`, `.csv`.
Classification rows need `input` and `label` (optional `labels`); math rows need `problem` and
`answer`; terminal rows need `instruction` and `test_command`.

Without `--test-tasks`, `--tasks` is split 70/30 by `--split-seed`.
Without `--baseline`, the seeds in `baselines/` for the task type are used.

## What a run produces

```
.meta-harness/
  run.json                       config + the search-split tasks
  frontier.json                  Pareto frontier over (score, context_cost)
  candidates/<id>/harness.py     candidate source
  candidates/<id>/scores.json    score, context_cost, repeats, score_std, valid, error
  candidates/<id>/traces.jsonl   every model call, task start/end, harness event
  candidates/<id>/proposer_reasoning.md
  proposals/iteration-NNNN/      proposer stdout/stderr
  views/iteration-NNNN/          what the proposer was allowed to read (ablation modes)
.meta-harness-test/
  test_results.json              held-out scores, written once, outside the proposer's reach
.meta-harness-cache/             model-call cache
```

Inspect a finished run: `meta-harness inspect .meta-harness`

## Proposer-view ablation (paper Table 3)

```bash
meta-harness run ... --proposer-view scores    # source + scores only
meta-harness run ... --proposer-view summary   # + LLM summaries, no raw traces
meta-harness run ... --proposer-view full      # everything (default)
```

## Agentic coding domain

Terminal tasks run the model's commands inside Docker:

```bash
meta-harness run --task-type terminal --tasks data/terminal.jsonl \
  --provider unikey --model claude-sonnet-4-6
```

Each row needs an `image`. Rows without one are refused unless you pass
`--allow-local-shell`, which executes model-authored commands **on your machine** — only do
that in a throwaway environment.

## Writing your own harness

```python
class Harness:
    def run(self, task, model, trace):
        prompt = f"Classify: {task['input']}"
        trace.event("prompt", {"prompt": prompt})
        return model(prompt)
```

The instance persists across the tasks of one evaluation (that's your memory) and is recreated
per evaluation. `model(prompt) -> str`; `model.call(prompt) -> (str, usage)`. Every call is
priced in input tokens and written to the trace. `meta_harness.retrieval` provides `TfidfIndex`,
`BM25Index`, and `reciprocal_rank_fusion`. See `baselines/` for six working examples and
`meta_harness/skill/SKILL.md` for the contract handed to the proposer.

## Tests

```bash
python -m pytest tests -q
```

The suite makes no network calls.
````

- [ ] **Step 4: Update .gitignore and pyproject**

`.gitignore` (note the current file has no trailing newline — fix that too):

```
venv/
.env
.meta-harness/
.meta-harness-test/
.meta-harness-demo/
.meta-harness-cache/
__pycache__/
*.py[cod]
.pytest_cache/
*.egg-info/
```

`pyproject.toml` — add the dev extra:

```toml
[project.optional-dependencies]
dev = ["pytest>=7"]
```

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest tests -q`
Expected: PASS (all tests, no network)

- [ ] **Step 6: Remove tracked bytecode if present**

Run: `git rm -r --cached --ignore-unmatch "*/__pycache__" "**/*.pyc"`
Then: `git status` — confirm only intended files are staged.

- [ ] **Step 7: Commit**

```bash
git add README.md .gitignore pyproject.toml tests/test_smoke_end_to_end.py
git commit -m "docs: document the end-to-end run and add a no-network smoke test"
```

---

## Self-Review

**Spec coverage**

| Spec requirement | Task |
|---|---|
| §2 Unikey provider | 1 |
| R1 coding-agent proposer + skill | 10 |
| R2 search/test separation | 7 |
| R3 seeded population | 9 |
| R4 proposer reasoning stored with candidate | 8 |
| R5 candidate isolation | 5 (subprocess), 4 (deadline) |
| R6 token-based context cost | 4 |
| R7 repeats | 6 |
| R8 parallel candidates | 6 |
| R9 task-type metrics | 2, 12 |
| R10 proposer-view ablation | 10 (`build_view`), 11 (CLI flag) |
| R11 caching | 3, 11 |
| R12 terminal domain | 12 |
| Global: no new deps, stdlib only | every task; 13 verifies |
| Global: no AI attribution on commits | every commit step |

**Placeholder scan:** no TBDs; every code step carries real code; no "similar to Task N".

**Type consistency checks performed:**
- `call(prompt, **kwargs) -> tuple[str, dict]` is used identically in Tasks 1, 3, 4.
- `usage` keys are `prompt_tokens`/`completion_tokens` everywhere; `AnthropicModel` translates
  `input_tokens`/`output_tokens` at the boundary (Task 1).
- `EvaluationResult` field order is fixed in Task 6 and every positional construction in Tasks
  6 and 7 matches it.
- `SearchConfig.test_root` is declared in Task 7 and consumed by the CLI in Task 11.
- `add_source(source, candidate_id, notes)` — the third positional in Task 8 matches both call
  sites in `_evaluate_source`.
- `build_view(experience, mode, destination, summarizer)` — same signature in Task 10's tests,
  implementation, and the `ClaudeCodeProposer.__call__` call.
- `METRICS` keys `classification`/`math` (Task 2) and `terminal` (Task 12) match the CLI's
  `--task-type` choices in Tasks 11 and 12.

**Known ordering constraint:** Task 5 changes `SearchRunner._evaluate_source` and Task 6
rewrites it; Task 7 rewrites `run` and Task 8 touches `_evaluate_source` again. Execute 4 → 5 →
6 → 7 → 8 in order. Tasks 1, 2, 3, 9 are independent and may run in parallel.
