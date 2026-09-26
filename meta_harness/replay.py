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

from .cc_history import FailureEpisode, Session, read_file_history

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
    # Classes that dominate real Claude Code history. Appended after the originals so the
    # first-match rule cannot change any existing classification.
    (r"contains multiple operations|Compound command changes working directory", "compound-shell"),
    (r"requires approval|denied by the Claude Code auto mode classifier", "needs-approval"),
    (r"doesn't want to proceed|tool use was rejected", "user-rejected"),
    (r"Blocked:|blocked by a deny rule", "blocked-policy"),
    (r"unexpected EOF while looking|simple_expansion|expansion obfuscation", "shell-quoting"),
    (r"not in Claude's tab group|determine which page this action targets", "tab-target"),
    (r"modified since read", "stale-read"),
    (r"EISDIR|illegal operation on a directory", "is-directory"),
    # Second pass: the classes that dominated the remaining "other" bucket once the eight
    # classes above were carved out (measured against .meta-harness/history-text/episodes.jsonl).
    # Appended last so none of the above ordering guarantees are disturbed.
    (r"Traceback \(most recent call last\)", "python-traceback"),
    (r"Permission to (use|read).*has been denied", "permission-denied-tool"),
    (r"File does not exist", "missing-path"),
    (r"is temporarily unavailable", "model-unavailable"),
    (
        r"InputValidationError|Workflow script file not found|No task found with ID"
        r"|Invalid workflow script|scriptPath must be a script path|Unknown skill:"
        r"|Task ID is required",
        "workflow-error",
    ),
    (r"Failed to execute JavaScript|JavaScript execution error", "js-error"),
    (
        r"Error capturing screenshot|actions\[\d+\][^\n]*failed|Failed to find element"
        r"|Failed to execute action|Error capturing zoomed screenshot"
        r"|is not a supported form input|Can't interact with browser-internal",
        "browser-action-failed",
    ),
    (r"No such tool available", "unknown-tool"),
    (r"hook did not respond before|tool did not respond in time", "hook-timeout"),
    (r"Found \d+ matches of the string", "edit-mismatch"),
    (r"Python was not found|pdftoppm is not installed", "missing-command"),
    (
        r"node:internal/modules/(package_json_reader|run_main)|Cannot find module"
        r"|ERR_MODULE_NOT_FOUND",
        "module-not-found",
    ),
    (r'"error":\{"name":"(HttpException|McpError)"|already exists in local config', "api-error"),
    (
        r"fatal: (pathspec|detected dubious ownership|ambiguous argument|.*is outside repository)"
        r"|ignored by one of your \.gitignore|docker: Error response from daemon",
        "git-error",
    ),
    (r"On branch \S+\r?\nYour branch is (up to date|ahead of)|warning: in the working copy of", "git-noise"),
    (r"npm error code|npm warn exec", "npm-error"),
    (r"exceeds maximum allowed tokens", "output-too-large"),
    (r"ConnectionRefusedError|connection refused|ECONNREFUSED", "connection-refused"),
    (r"was blocked\. For security|is blocked\. This path is protected|denied by your permission", "blocked-policy"),
    (r"=+ FAILURES =+|ERROR at setup of|\bAssertionError\b|FAILED \S+::", "test-failure"),
    (r"tab group no longer exists|Missing required parameter tabId", "tab-target"),
    (r"needs design-system authorization", "needs-approval"),
    (r"ENAMETOOLONG", "path-too-long"),
    (r"error TS\d+|imported but unused", "ts-error"),
    (r"not logged into any GitHub hosts", "gh-auth-error"),
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


MAX_REPLAY_FILE_CHARS = 8000


def _request_before(session: Session, turn_index: int) -> str:
    asks = [turn.text for turn in session.turns
            if turn.role == "user" and turn.text and turn.index < turn_index]
    return asks[-1] if asks else ""


def build_replay(episode: FailureEpisode, session: Session, home: Path | None = None,
                 max_files: int = 4) -> dict[str, Any] | None:
    """The smallest run that shows whether this failure recurs, or None if it cannot be built."""
    expectation = expectation_for(episode)
    if not expectation:
        return None
    instruction = _request_before(session, episode.turn_index)
    if not instruction:
        return None

    seeded: dict[str, str] = {}
    for version in read_file_history(session.session_id, home):
        if version.previous_content is None or not version.tracking_path:
            continue
        seeded.setdefault(version.tracking_path, version.previous_content)
    smallest = sorted(seeded.items(), key=lambda kv: len(kv[1]))[:max_files]

    return {
        "instruction": instruction,
        "files": {path.replace("\\", "/"): body[:MAX_REPLAY_FILE_CHARS]
                  for path, body in smallest},
        "expect": expectation,
        "_origin": {"kind": episode.kind, "session": session.session_id,
                    "turn": episode.turn_index, "signature": episode_signature(episode)},
    }


__all__ = ["ERROR_PATTERNS", "MAX_REPLAY_FILE_CHARS", "THRASH_THRESHOLD", "THRASH_WINDOW",
           "build_replay", "episode_signature", "expectation_for", "load_agent_steps",
           "verify_expectation"]
