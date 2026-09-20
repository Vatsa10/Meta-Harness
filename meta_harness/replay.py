"""Turning an observed failure into a replay: the smallest run that shows whether it recurs.

A failure carries its own test. A tool error passes its replay when that error class does not
occur again; a thrash episode passes when the tool is no longer hammered inside the window.
Neither needs a human-written assertion, which is what makes ~1,266 of the mined episodes usable
without a hand-built benchmark.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cc_history import FailureEpisode

# Matched against the error text of a failed tool result, most specific first. The slug is what
# makes two instances of one fault share a signature; without it dedupe never fires.
ERROR_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"UnicodeEncodeError", "unicode-encode"),
    (r"UnicodeDecodeError|charmap|cp1252", "unicode-decode"),
    (r"No such file or directory|cannot find the (file|path)", "missing-path"),
    (r"Permission denied|EACCES", "permission"),
    (r"command not found|is not recognized as", "missing-command"),
    (r"timed out|TimeoutExpired", "timeout"),
    (r"has not been read yet|must read.*before", "unread-edit"),
    (r"String to replace not found|old_string", "edit-mismatch"),
    (r"SyntaxError|unterminated", "syntax"),
)


def _cause(text: str) -> str:
    for pattern, slug in ERROR_PATTERNS:
        if re.search(pattern, text or "", re.IGNORECASE):
            return slug
    return "other"


def episode_signature(episode: FailureEpisode) -> str:
    """A stable key for one failure class, used for dedupe and tombstones."""
    tools = list(episode.tools or [])
    if episode.kind == "thrash":
        return f"thrash:{tools[0] if tools else 'unknown'}"
    if episode.kind == "correction":
        return "correction:" + "+".join(sorted(tools)) if tools else "correction:unknown"
    tool = tools[0] if tools else "unknown"
    return f"tool_error:{tool}:{_cause(episode.assistant_text or episode.detail or '')}"


THRASH_WINDOW = 6
THRASH_THRESHOLD = 4


def expectation_for(episode: FailureEpisode) -> dict[str, Any]:
    """The mechanical check that says whether this failure recurred."""
    tools = list(episode.tools or [])
    tool = tools[0] if tools else "unknown"
    if episode.kind == "thrash":
        return {"no_thrash": {"tool": tool, "window": THRASH_WINDOW,
                              "threshold": THRASH_THRESHOLD}}
    if episode.kind == "tool_error":
        return {"no_tool_error": {"tool": tool,
                                  "cause": _cause(episode.assistant_text or episode.detail or "")}}
    return {}


def load_agent_steps(trace_path: Path | str) -> list[dict[str, Any]]:
    """The `agent_step` payloads of one replay run, in order."""
    path = Path(trace_path)
    if not path.is_file():
        return []
    steps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or '"agent_step"' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") == "agent_step" and isinstance(record.get("payload"), dict):
            steps.append(record["payload"])
    return steps


def _pattern_for_cause(cause: str) -> str:
    for pattern, slug in ERROR_PATTERNS:
        if slug == cause:
            return pattern
    return ""


def verify_expectation(expectation: Mapping[str, Any],
                       steps: Sequence[Mapping[str, Any]]) -> bool:
    """True when the failure this replay was built from did not recur."""
    if not expectation:
        return True

    spec = expectation.get("no_tool_error")
    if spec:
        pattern = _pattern_for_cause(str(spec.get("cause", "")))
        if not pattern:
            return True
        for step in steps:
            if step.get("role") != "tool_result":
                continue
            if re.search(pattern, str(step.get("content", "")), re.IGNORECASE):
                return False
        return True

    spec = expectation.get("no_thrash")
    if spec:
        tool = spec.get("tool")
        window = int(spec.get("window", THRASH_WINDOW))
        threshold = int(spec.get("threshold", THRASH_THRESHOLD))
        uses = [index for index, step in enumerate(steps)
                if step.get("role") == "tool_use" and step.get("name") == tool]
        for start in uses:
            inside = sum(1 for index in uses if start <= index < start + window)
            if inside >= threshold:
                return False
        return True

    return True


__all__ = ["ERROR_PATTERNS", "THRASH_THRESHOLD", "THRASH_WINDOW",
           "episode_signature", "expectation_for", "load_agent_steps", "verify_expectation"]
