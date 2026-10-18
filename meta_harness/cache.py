"""Disk cache for model calls. Repeats and re-runs must not re-bill the gateway."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

PROPOSER_CACHE_ENV = "META_HARNESS_PROPOSER_CACHE"


def cache_key(model_id: str, prompt: str, kwargs: Mapping[str, Any],
              sample_index: int = 0) -> str:
    payload = json.dumps({"model": model_id, "prompt": prompt, "kwargs": dict(kwargs),
                          "sample": sample_index},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def proposer_cache_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """The proposer call cache is opt-in so live runs draw fresh proposals by default."""
    return (os.environ if environ is None else environ).get(PROPOSER_CACHE_ENV, "") == "1"


def default_proposer_cache_dir() -> Path:
    return Path.home() / ".cache" / "meta_harness" / "proposer"


# Adapted from google-research/harness2 (Apache-2.0), harness2/core/proposer.py
# (_call_fingerprint): content-address a proposer call by every evidence file and its workspace.
def proposer_call_key(view_root: Path | str, workspace: Path | str, model_id: str, prompt: str,
                      skip: Iterable[str] = ()) -> str:
    """sha256 over model id, prompt, resolved workspace path and every visible file (path+bytes).

    The workspace path is part of the key: two runs with byte-identical evidence stay independent
    draws. Top-level directories in `skip` (the proposer cannot read them) are left out.
    """
    skipped = set(skip)
    digest = hashlib.sha256()
    for value in ("v1", model_id, prompt, str(Path(workspace).resolve())):
        digest.update(value.encode("utf-8", errors="replace") + b"\0")
    root = Path(view_root)
    if root.is_dir():
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root).as_posix()
            if not path.is_file() or rel.split("/")[0] in skipped or "__pycache__" in path.parts:
                continue
            digest.update(rel.encode("utf-8", errors="replace") + b"\0")
            try:
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
            except OSError as exc:
                digest.update(f"<unreadable:{type(exc).__name__}>".encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


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

    supports_sample_index = True

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

    def call(self, prompt: str, *, sample_index: int = 0, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            return self._upstream(prompt, **kwargs)
        key = cache_key(self.model_id, prompt, kwargs, sample_index)
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

    @property
    def chat(self) -> Callable[..., Any]:
        """Native tool-calling turns pass through uncached; absent when the wrapped model has none."""
        upstream = getattr(self.model, "chat", None)
        if not callable(upstream):
            raise AttributeError("chat")
        return upstream


__all__ = ["CachedModel", "DiskCache", "PROPOSER_CACHE_ENV", "cache_key", "default_proposer_cache_dir",
           "proposer_cache_enabled", "proposer_call_key"]
