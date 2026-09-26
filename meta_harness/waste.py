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
# under-count, which every report built on it has to say out loud. Trailing \b on the
# word-final alternatives so "stop" does not match inside "stopped" (a subagent notice
# containing "<status>stopped</status>" must not be read as a correction); the alternatives
# that end in punctuation ("no,", "actually,") keep no trailing boundary, since \b never
# matches between two non-word characters (comma, then the space that follows it).
#
# This full pattern is kept public for callers that want the plain keyword check (and to keep
# the "stop does not match stopped" guarantee testable on its own), but `_is_correction` below
# - not this regex alone - is what actually classifies a stretch, because a hand-labelled gold
# set (D:\...\correction-gold.json, 64 stretches this detector fired on before this fix, hand
# read: 18 genuine, 46 false) showed that matching this whole pattern anywhere in the message
# is only 28% precise. Most of the 46 false positives were the user pasting the assistant's own
# earlier report (which itself contains "wrong"/"revert"/"instead") ahead of a new, unrelated
# request, or an operational halt ("stop the server") that objects to nothing.
CORRECTION_RE = re.compile(
    r"\b(no,|not what\b|wrong\b|don'?t do\b|stop\b|revert\b|undo\b|why did you\b|i said\b"
    r"|actually,|that'?s not\b|instead of\b|you broke\b|doesn'?t work\b"
    r"|still (broken|failing)\b)",
    re.I,
)

# <pasted_content ...>...</pasted_content> wraps material the user pasted as context, not
# words they typed - stripped before any correction match so a pasted assistant report's own
# "wrong"/"revert"/"instead" cannot be read as the human objecting.
PASTED_CONTENT_RE = re.compile(r"<pasted_content[^>]*>.*?</pasted_content>", re.I | re.S)

# Vocabulary anchored to the OPENING of the human's own words (see `_opening`): a real
# objection leads with it, so a trigger word buried deep in a long message (typically after
# pasted context, or a message that goes on to make an unrelated new request) does not count.
# "stop" gets its own guard instead of a bare \b match: "stop the server/agents/processes/..."
# is an operational halt instruction, not an objection to anything the assistant did.
_STOP_OBJECTS = (
    r"server|servers|agent|agents|process|processes|instance|instances|worker|workers|"
    r"session|sessions|job|jobs|task|tasks|everything|running|background|loop|loops|app|"
    r"application|service|services|container|containers|build|script|scripts|it|them|that|now"
)
_OPENING_CORRECTION_RE = re.compile(
    r"\b(no,|not what\b|wrong\b|don'?t do\b|why did you\b|i said\b"
    r"|actually,|that'?s not\b|instead of\b|you broke\b|doesn'?t work\b"
    r"|still (broken|failing)\b|revert\b|undo\b|roll back\b)"
    r"|\bstop\b(?!(?:\s+\w+){0,3}\s*\b(?:" + _STOP_OBJECTS + r"))",
    re.I,
)
# Fix round 3: "revert"/"undo"/"roll back" used to be checked over the WHOLE message (capped at
# 3000 chars), on the theory they stayed high-precision even said well into a message. Measured
# against the full 88-record gold set (D:\...\correction-gold.json) over ALL current firings,
# that widening bought 6 extra genuine corrections at the cost of 27 false ones (precision
# 15/51 = 29.4% with it, vs 9/18 = 50.0% keeping 9 of 18 genuine without it). A narrower
# "imperative at the start of a clause" form was also tried and measured worse (kept only 9 of
# 18 genuine, same false count, lower precision than plain opening-anchoring). So "revert",
# "undo" and "roll back" are folded into _OPENING_CORRECTION_RE above instead: anchored to the
# message's own opening exactly like every other trigger word, no full-text scan at all.

# User-role messages the harness itself injects, not the human speaking: slash-command
# scaffolding, background-task notifications, local-command output/caveats, system reminders,
# an IDE-injected open-file notice, and the interruption marker. Each one used to wrongly cut
# a stretch or (for the "stopped" task-notification, before CORRECTION_RE's trailing \b) count
# as a correction.
HARNESS_MARKERS = (
    "<task-notification>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<system-reminder>",
    "<ide_opened_file>",
    "[Request interrupted by user",
    "Base directory for this skill:",
)

PATH_KEYS = ("file_path", "path", "notebook_path")


def _is_harness_text(text: str) -> bool:
    """True when a user-role message is harness-injected scaffolding, not human speech."""
    return text.startswith(HARNESS_MARKERS)


def _opening(text: str, max_chars: int = 200) -> str:
    """The first sentence or so of a message: up to the first ./!/? followed by whitespace or
    end of string, a newline, or a length cap, whichever comes first. A real objection leads
    with it; a keyword many sentences later usually belongs to a different point in a longer,
    multi-topic message."""
    text = text.strip()
    window = text[:max_chars]
    match = re.search(r"[.!?](?:\s|$)|\n", window)
    cut = match.end() if match else max_chars
    return text[:cut]


def _is_correction(text: str) -> bool:
    """Whether a human message is a genuine wrong-direction correction, tuned against a
    hand-labelled gold set rather than intuition (see CORRECTION_RE's docstring)."""
    stripped = PASTED_CONTENT_RE.sub(" ", text)
    return _OPENING_CORRECTION_RE.search(_opening(stripped)) is not None


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


def _looks_like_a_path(token: str) -> bool:
    """Reject the noise `_paths_in`'s command regex also matches: git SHA ranges
    (`145181b..HEAD`), URL fragments (`//github.com/...`), and bare hex runs (a commit SHA
    with no path-like structure around it)."""
    if ".." in token or "://" in token or token.startswith("//"):
        return False
    if re.fullmatch(r"[0-9a-fA-F]+", token):
        return False
    return True


def _paths_in(tool_input: object) -> list[str]:
    if not isinstance(tool_input, dict):
        return []
    found = [str(tool_input[key]) for key in PATH_KEYS if tool_input.get(key)]
    command = tool_input.get("command")
    if isinstance(command, str):
        candidates = re.findall(r"[\w./\\-]+\.[A-Za-z]{1,4}\b", command)[:4]
        found.extend(token for token in candidates if _looks_like_a_path(token))
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
                if text and not _is_harness_text(text):
                    if len(current.calls) >= min_calls:
                        current.ended_by = "correction" if _is_correction(text) else "user"
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

# Fix round 3, measured against the full 88-record hand-labelled gold set
# (D:\...\correction-gold.json) over ALL current firings of this detector on the real transcript
# store: 11 of 21 flagged stretches were genuine corrections in a hand-labelled audit (52.4%
# precision). The gold set covers every stretch the detector fires on as of that audit, so this
# measures precision, not recall: a genuine correction this detector never fires on at all is
# invisible to this figure, and recall is unmeasured.
CAVEAT = ("Correction counts come from a keyword heuristic over your own messages. In a "
          "hand-labelled audit, 11 of 21 flagged stretches were genuine corrections (52.4% "
          "precision) - roughly half of what's flagged may not be a genuine correction, and "
          "recall (genuine corrections this misses entirely) is unmeasured. Treat this as a "
          "cost estimate, not an audit.")


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
        scanned_stretches, tool_call_count, signatures = _scan(path)
        calls += tool_call_count
        for stretch in scanned_stretches:
            stretches += 1
            if stretch.ended_by == "correction":
                lags.append(len(stretch.calls))
                bucket = per_project[stretch.project]
                bucket["corrections"] += 1
                bucket["calls_burned"] += len(stretch.calls)
        seen: collections.Counter[tuple[str, str]] = collections.Counter()
        for key, count in signatures.items():
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


def _normalize_error(text: str) -> str:
    """Collapse an unclassified ("other") error to a signature two occurrences of the SAME
    error share, while two different unclassified errors still land on different keys.
    Strips paths, hex ids/SHAs and digits (which vary run to run), collapses whitespace, and
    truncates - "identical" has to mean identical, not merely "both unclassified"."""
    text = re.sub(r"[\\/][\w./-]+", " ", text)
    text = re.sub(r"\b[0-9a-fA-F]{6,}\b", " ", text)
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:100]


def _scan(path: Path, min_calls: int = 3) -> tuple[list[Stretch], int, dict[tuple[str, str], int]]:
    """One read of a transcript, producing everything waste_report needs from it: the stretches
    (grouped exactly as `iter_stretches` groups them), the total count of every tool_use block
    regardless of stretch length, and (tool, cause) occurrence counts for error tool_results.

    Kept separate from `iter_stretches` (whose public signature and per-file streaming behaviour
    other callers rely on) so that `waste_report` reads each transcript exactly once instead of
    once for stretches and again for repeat signatures - on this machine's store that halved the
    report's running time.
    """
    path = Path(path)
    session_id = path.stem
    project = path.parent.name
    stretches: list[Stretch] = []
    current = Stretch(session_id, project, 0)
    turn = 0
    total_calls = 0
    names: dict[str, str] = {}
    signatures: collections.Counter[tuple[str, str]] = collections.Counter()
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return stretches, total_calls, dict(signatures)
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
                if text and not _is_harness_text(text):
                    if len(current.calls) >= min_calls:
                        current.ended_by = "correction" if _is_correction(text) else "user"
                        current.correction_text = text[:400] if current.ended_by == "correction" else ""
                        stretches.append(current)
                    current = Stretch(session_id, project, turn)
                    continue

            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")
                    if btype == "tool_use":
                        total_calls += 1
                        names[str(block.get("id"))] = str(block.get("name", "?"))
                        if not current.calls:
                            current.start_turn = turn
                        current.calls.append(str(block.get("name", "?")))
                        current.paths.extend(_paths_in(block.get("input")))
                    elif btype == "tool_result" and block.get("is_error"):
                        body = block.get("content")
                        if isinstance(body, list):
                            body = " ".join(part.get("text", "") for part in body
                                            if isinstance(part, dict))
                        tool = names.get(str(block.get("tool_use_id")), "?")
                        body_text = str(body)
                        cause = _cause(body_text)
                        key = (tool, cause) if cause != "other" else (tool, _normalize_error(body_text))
                        signatures[key] += 1

    if len(current.calls) >= min_calls:
        stretches.append(current)
    return stretches, total_calls, dict(signatures)
