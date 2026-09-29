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


from collections.abc import Mapping, Sequence, Set as AbstractSet

QUIET_DAYS = 90.0


def retirement_candidates(installed: "Sequence[object]", live_signatures: AbstractSet[str],
                          now: datetime | None = None,
                          quiet_days: float = QUIET_DAYS,
                          verdicts: "Mapping[str, str] | None" = None) -> list[tuple[object, str]]:
    """Installed artifacts whose origin failure has not been seen for `quiet_days`.

    Proposed, never performed. An artifact that is doing its job prevents the very evidence
    that would justify keeping it, so silence is ambiguous and a human has to decide.

    `verdicts` maps an artifact id to its receipt verdict; an artifact whose verdict is
    "no measurable effect" is also proposed, on the measured evidence.
    """
    moment = now or datetime.now(timezone.utc)
    proposed: list[tuple[object, str]] = []
    for artifact in installed:
        reasons: list[str] = []
        signature = (getattr(artifact, "origin", {}) or {}).get("signature", "")
        created = _parse(getattr(artifact, "created", "") or "")
        if signature and signature not in live_signatures and created is not None:
            age_days = (moment - created).total_seconds() / 86400.0
            if age_days >= quiet_days:
                reasons.append(f"{signature} has not been seen in "
                               f"{quiet_days:.0f} days (installed {age_days:.0f} days ago)")
        if verdicts and verdicts.get(str(getattr(artifact, "id", ""))) == "no measurable effect":
            reasons.append("receipts show no measurable effect: recurrence with it acting is "
                           "not lower than with it held out")
        if reasons:
            proposed.append((artifact, "; ".join(reasons)))
    return proposed


__all__ = ["HALF_LIFE_DAYS", "WEIGHT_FLOOR", "QUIET_DAYS", "recency_weight", "version_weight",
           "retirement_candidates"]
