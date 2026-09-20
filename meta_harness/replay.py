"""Turning an observed failure into a replay: the smallest run that shows whether it recurs.

A failure carries its own test. A tool error passes its replay when that error class does not
occur again; a thrash episode passes when the tool is no longer hammered inside the window.
Neither needs a human-written assertion, which is what makes ~1,266 of the mined episodes usable
without a hand-built benchmark.
"""

from __future__ import annotations

import re
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


__all__ = ["ERROR_PATTERNS", "episode_signature"]
