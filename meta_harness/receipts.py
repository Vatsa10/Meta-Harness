"""Attribute outcomes to decision receipts (spec 2026-09-29-decision-receipts-design.md)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .replay import _cause

K = 10
MIN_PER_ARM = 5
MIN_EFFECT = 0.2
Z = 1.96
USER_DRIVEN = {"session-rule", "rejection-memory"}


def _read_jsonl(paths, session_prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                if session_prefix and path.stem.startswith(session_prefix):
                    # The hook keeps the session in the file name, not in the record.
                    row.setdefault("session", path.stem[len(session_prefix):])
                rows.append(row)
    return rows


def load_receipts(home: Path) -> list[dict[str, Any]]:
    return _read_jsonl(sorted(Path(home).glob("receipts-*.jsonl"))) if Path(home).exists() else []


def load_observations(home: Path) -> list[dict[str, Any]]:
    return _read_jsonl(sorted(Path(home).glob("observed-*.jsonl")), "observed-") if Path(home).exists() else []


def _signature(obs: dict[str, Any]) -> str | None:
    """The registry's signature for an observation (same rules as learn.observed_failures)."""
    tool = str(obs.get("tool", "unknown"))
    if obs.get("kind") == "repeat":
        return f"thrash:{tool}"
    if obs.get("kind") == "tool_error":
        cause = obs.get("cause") or _cause(str(obs.get("text", "")))
        return f"tool_error:{tool}:{cause}"
    return None


def attribute(receipts: Sequence[dict[str, Any]],
              observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        if _signature(obs) and isinstance(obs.get("call"), int):
            by_session[str(obs.get("session", ""))].append(obs)
    out = []
    for receipt in receipts:
        call, signature = receipt.get("call"), receipt.get("signature")
        recurred = isinstance(call, int) and bool(signature) and any(
            call < obs["call"] <= call + K and _signature(obs) == signature
            for obs in by_session.get(str(receipt.get("session", "")), []))
        out.append({**receipt, "recurred": recurred})
    return out


def _verdict(source: str, acted: list, held: list) -> tuple[str, float | None, float | None]:
    acted_rate = sum(r["recurred"] for r in acted) / len(acted) if acted else None
    held_rate = sum(r["recurred"] for r in held) / len(held) if held else None
    if source in USER_DRIVEN:
        return "no control arm", acted_rate, held_rate
    if len(acted) < MIN_PER_ARM or len(held) < MIN_PER_ARM:
        return "not enough data", acted_rate, held_rate
    pooled = (sum(r["recurred"] for r in acted) + sum(r["recurred"] for r in held)) / (len(acted) + len(held))
    se = math.sqrt(pooled * (1 - pooled) * (1 / len(acted) + 1 / len(held)))
    effect = held_rate - acted_rate
    z = effect / se if se > 0 else 0.0
    if effect >= MIN_EFFECT and z >= Z:
        return "helps", acted_rate, held_rate
    return "no measurable effect", acted_rate, held_rate


def summarize(attributed: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in attributed:
        groups[(str(row.get("source", "")), row.get("artifact"))].append(row)
    table = []
    for (source, artifact), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        acted = [r for r in rows if r.get("decision") == "acted"]
        held = [r for r in rows if r.get("decision") == "held"]
        verdict, acted_rate, held_rate = _verdict(source, acted, held)
        effect = (held_rate - acted_rate) if acted_rate is not None and held_rate is not None else None
        table.append({"source": source, "artifact": artifact, "acted": len(acted), "held": len(held),
                      "acted_rate": acted_rate, "held_rate": held_rate, "effect": effect,
                      "verdict": verdict})
    return table
