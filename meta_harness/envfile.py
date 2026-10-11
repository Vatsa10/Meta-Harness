"""Minimal .env loader: fills variables that are not already set, never prints values."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


def _strip_comment(value: str) -> str:
    """Drop a whitespace-preceded `#` comment that sits outside any quotes."""
    quote = ""
    for i, ch in enumerate(value):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"" and i == 0:
            quote = ch
        elif ch == "#" and i > 0 and value[i - 1] in " 	":
            return value[:i]
    return value


def _value(raw: str) -> str:
    value = _strip_comment(raw.strip()).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def parse_env(text: str) -> dict[str, str]:
    """KEY=value lines. Strips whitespace around key and value, an optional leading
    `export `, and one pair of matching surrounding quotes. Ignores blank lines, `#`
    comment lines and lines without `=`. Inline ` #` after an unquoted value is a comment."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip().lstrip("﻿")
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export "):]
        key, raw = stripped.split("=", 1)
        key = key.strip()
        if key:
            result[key] = _value(raw)
    return result


def load_env(path: Path | str = ".env",
             environ: MutableMapping[str, str] = os.environ) -> list[str]:
    """Set variables that are not already set; return the NAMES set (never values).
    Missing file -> []."""
    file = Path(path)
    if not file.is_file():
        return []
    loaded = []
    for key, value in parse_env(file.read_text(encoding="utf-8-sig")).items():
        if key not in environ:
            environ[key] = value
            loaded.append(key)
    return loaded


__all__ = ["parse_env", "load_env"]
