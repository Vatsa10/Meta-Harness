"""Secret redaction and bounded, file-name-safe text."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any

SECRET = re.compile(r"(?i)(api[_-]?key\S*\s*[:=]\s*\S+|bearer\s+\S+|sk-[A-Za-z0-9_-]{8,})")
REDACTED = "[redacted]"


def _secret_values() -> list[str]:
    values = {value for name, value in os.environ.items()
              if (name.endswith("_API_KEY") or name.endswith("_TOKEN")) and len(value) >= 8}
    return sorted(values, key=len, reverse=True)


def redact(text: str) -> str:
    """Replace every SECRET match and every loaded *_API_KEY/*_TOKEN value (>= 8 chars)."""
    for value in _secret_values():
        text = text.replace(value, REDACTED)
    return SECRET.sub(REDACTED, text)


def bounded(text: str, limit: int = 2000) -> str:
    """Redact, collapse to one line, cap with '...[+N chars]'."""
    line = " ".join(redact(text).split())
    if len(line) <= limit:
        return line
    return f"{line[:limit]}...[+{len(line) - limit} chars]"


def safe_id(text: str) -> str:
    """ASCII [A-Za-z0-9._-], at most 64 chars, sha256 suffix when shortened or altered."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "id"
    if cleaned == text and len(cleaned) <= 64:
        return cleaned
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[:64 - len(digest) - 1]}-{digest}"


def redact_obj(value: Any) -> Any:
    """Recursive redaction over dict/list/tuple/str."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: redact_obj(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(redact_obj(item) for item in value)
    return value


__all__ = ["SECRET", "redact", "bounded", "safe_id", "redact_obj"]
