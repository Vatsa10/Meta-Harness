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
