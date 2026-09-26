"""What a developer loses to work that was going the wrong way.

Reads Claude Code transcripts directly rather than through cc_history, because the unit here
is the stretch - a run of tool calls with no human input - which cc_history does not model.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# A human saying the work went the wrong way. A keyword heuristic: it will both over- and
# under-count, which every report built on it has to say out loud.
CORRECTION_RE = re.compile(
    r"\b(no,|not what|wrong|don'?t do|stop|revert|undo|why did you|i said|actually,"
    r"|that'?s not|instead of|you broke|doesn'?t work|still (broken|failing))",
    re.I,
)

PATH_KEYS = ("file_path", "path", "notebook_path")


@dataclass
class Stretch:
    """A run of assistant tool calls with no human input, and how it ended."""

    session_id: str
    project: str
    start_turn: int
    calls: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    ended_by: str = "end"          # correction | user | end
    correction_text: str = ""


def _user_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text")
    return ""


def _paths_in(tool_input: object) -> list[str]:
    if not isinstance(tool_input, dict):
        return []
    found = [str(tool_input[key]) for key in PATH_KEYS if tool_input.get(key)]
    command = tool_input.get("command")
    if isinstance(command, str):
        found.extend(re.findall(r"[\w./\\-]+\.[A-Za-z]{1,4}\b", command)[:4])
    return found


def iter_stretches(path: Path, min_calls: int = 3) -> Iterator[Stretch]:
    """Stream one transcript, yielding each stretch of at least `min_calls` tool calls.

    Streams line by line and holds only the current stretch, because a single session on this
    machine reaches 23,000 turns. A malformed line is skipped, not fatal: a session may be
    written while we read it.
    """
    path = Path(path)
    session_id = path.stem
    project = path.parent.name
    current = Stretch(session_id, project, 0)
    turn = 0
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            turn += 1
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                text = _user_text(message).strip()
                if text:
                    if len(current.calls) >= min_calls:
                        current.ended_by = "correction" if CORRECTION_RE.search(text[:400]) else "user"
                        current.correction_text = text[:400] if current.ended_by == "correction" else ""
                        yield current
                    current = Stretch(session_id, project, turn)
                    continue

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if not current.calls:
                            current.start_turn = turn
                        current.calls.append(str(block.get("name", "?")))
                        current.paths.extend(_paths_in(block.get("input")))

    if len(current.calls) >= min_calls:
        yield current


import collections
from typing import Any

from .cc_history import claude_home, is_artifact_project
from .replay import _cause

CAVEAT = ("Correction counts come from a keyword heuristic over your own messages. It will both "
          "over- and under-count. Treat this as a cost estimate, not an audit.")


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile. Empty input is 0, which is what an empty history should report."""
    if not values:
        return 0
    index = min(len(values) - 1, int(len(values) * fraction))
    return values[index]


def _transcripts(home: Path, project: str | None) -> list[Path]:
    if not home.exists():
        return []
    found: list[Path] = []
    for directory in sorted(home.iterdir()):
        if not directory.is_dir() or is_artifact_project(directory.name):
            continue
        if project and project not in directory.name:
            continue
        found.extend(sorted(directory.glob("*.jsonl")))
    return found


def waste_report(home: Path | None = None, project: str | None = None,
                 limit: int | None = None, since: str = "") -> dict[str, Any]:
    """What wrong-direction work and repeated failure cost, across the transcript store."""
    root = Path(home) if home is not None else claude_home() / "projects"
    paths = _transcripts(root, project)
    if since:
        try:
            cutoff = datetime.fromisoformat(since).timestamp()
        except ValueError:
            cutoff = 0.0          # an unusable date filters nothing rather than everything
        if cutoff:
            paths = [p for p in paths if p.stat().st_mtime >= cutoff]
    if limit:
        paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    lags: list[int] = []
    per_project: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"corrections": 0, "calls_burned": 0})
    repeats: collections.Counter[tuple[str, str]] = collections.Counter()
    sessions_with_repeat = 0
    calls = 0
    stretches = 0

    for path in paths:
        seen: collections.Counter[tuple[str, str]] = collections.Counter()
        for stretch in iter_stretches(path):
            stretches += 1
            calls += len(stretch.calls)
            if stretch.ended_by == "correction":
                lags.append(len(stretch.calls))
                bucket = per_project[stretch.project]
                bucket["corrections"] += 1
                bucket["calls_burned"] += len(stretch.calls)
        for key, count in _repeat_signatures(path).items():
            if count > 1:
                seen[key] += count - 1
        if seen:
            sessions_with_repeat += 1
            repeats.update(seen)

    lags.sort()
    return {
        "sessions": len(paths),
        "tool_calls": calls,
        "stretches": stretches,
        "corrections": {
            "count": len(lags),
            "median": _percentile(lags, 0.5),
            "p75": _percentile(lags, 0.75),
            "p90": _percentile(lags, 0.90),
            "max": lags[-1] if lags else 0,
            "calls_burned": sum(lags),
        },
        "repeats": {
            "wasted_retries": sum(repeats.values()),
            "sessions_affected": sessions_with_repeat,
            "top": [{"tool": tool, "cause": cause, "wasted": n}
                    for (tool, cause), n in repeats.most_common(10)],
        },
        "by_project": sorted(
            ({"project": name, **counts} for name, counts in per_project.items()),
            key=lambda row: row["calls_burned"], reverse=True),
        "caveat": CAVEAT,
    }


def _repeat_signatures(path: Path) -> dict[tuple[str, str], int]:
    """(tool, cause) -> occurrences, for error results in one transcript."""
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    names: dict[str, str] = {}
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            content = (record.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    names[str(block.get("id"))] = str(block.get("name", "?"))
                elif block.get("type") == "tool_result" and block.get("is_error"):
                    body = block.get("content")
                    if isinstance(body, list):
                        body = " ".join(part.get("text", "") for part in body
                                        if isinstance(part, dict))
                    tool = names.get(str(block.get("tool_use_id")), "?")
                    counts[(tool, _cause(str(body)))] += 1
    return dict(counts)
