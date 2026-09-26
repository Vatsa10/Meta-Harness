"""Evidence ages. A failure class fixed six months ago should not outrank this week's.

Weights are multiplicative and never zero: old evidence is quieter, never deleted. A session
with no usable clock or version keeps full weight, because absence of a timestamp is not
evidence of staleness.
"""
from __future__ import annotations

from datetime import datetime, timezone

HALF_LIFE_DAYS = 30.0
WEIGHT_FLOOR = 0.05


def _parse(stamp: str) -> datetime | None:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def recency_weight(timestamp: str, now: datetime | None = None,
                   half_life_days: float = HALF_LIFE_DAYS) -> float:
    """1.0 for now, halving every `half_life_days`, floored so nothing vanishes entirely."""
    seen = _parse(timestamp)
    if seen is None:
        return 1.0
    moment = now or datetime.now(timezone.utc)
    days = (moment - seen).total_seconds() / 86400.0
    if days <= 0:                       # a clock ahead of ours is not extra credible
        return 1.0
    return max(WEIGHT_FLOOR, 0.5 ** (days / half_life_days))


def _minor(version: str) -> tuple[int, int] | None:
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def version_weight(seen: str, current: str) -> float:
    """Discount evidence from a distant Claude Code version: the tool may have changed."""
    left, right = _minor(seen), _minor(current)
    if left is None or right is None:
        return 1.0
    distance = abs(left[0] - right[0]) * 100 + abs(left[1] - right[1])
    if distance == 0:
        return 1.0
    return 0.6 if distance == 1 else 0.3


__all__ = ["HALF_LIFE_DAYS", "WEIGHT_FLOOR", "recency_weight", "version_weight"]
