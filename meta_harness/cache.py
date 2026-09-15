"""Disk cache for model calls. Repeats and re-runs must not re-bill the gateway."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable, Mapping


def cache_key(model_id: str, prompt: str, kwargs: Mapping[str, Any]) -> str:
    payload = json.dumps({"model": model_id, "prompt": prompt, "kwargs": dict(kwargs)},
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
