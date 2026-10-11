"""The domain-free default: records pass through unchanged."""

from __future__ import annotations

NAME = "generic"
PROMPT = ""
METRICS: dict = {}
SEED_DIR = None
PROTOCOL = "plain"


def adapt(records: list[dict]) -> list[dict]:
    return [dict(r) for r in records]
