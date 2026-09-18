"""Read Claude Code's own stored experience.

The paper's central design is a filesystem holding every prior attempt - source, scores and
execution traces - that the proposer queries rather than ingests. Claude Code already keeps one:

    ~/.claude/projects/<project-slug>/<session-id>.jsonl

Each line is a record: user turns, assistant turns with tool_use blocks, tool results (with an
`is_error` flag), subagent turns (`isSidechain`), plus cwd, git branch and CLI version. That is
real harness experience - what the agent was asked, what it did, where it failed, and what the
human said next.

This module reads it. Nothing here sends anything anywhere: the files are read locally and the
outputs are written under the repository. Message text is redacted by default, because a
transcript contains whatever the user typed; pass `include_text=True` only deliberately.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


def claude_home() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or (Path.home() / ".claude"))


def project_slug(path: Path | str) -> str:
    """Claude Code's on-disk name for a working directory.

    One dash per non-alphanumeric character, not one per run: `D:\\a\\b` becomes `D--a-b`.
    """
    return re.sub(r"[^A-Za-z0-9]", "-", str(Path(path).resolve()))


# Search runs create a throwaway workspace per task, and Claude Code records each as its own
# project. Those are artifacts of this tool, not real work, and would drown the signal.
ARTIFACT_MARKERS = ("meta-harness-workspaces", "mh-agent-", "meta-harness-cli",
                    "meta-harness-demo", "cc-probe")


def is_artifact_project(name: str) -> bool:
    return any(marker in name for marker in ARTIFACT_MARKERS)


# A user turn matching one of these, arriving straight after the agent did work, is the
# cheapest reliable signal that the previous turn was wrong.
CORRECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bthat'?s (?:wrong|not right|not what)\b", r"\bno,? (?:that|it|you)\b",
        r"\b(?:still|again) (?:broken|failing|wrong|not working)\b",
        r"\bdoesn'?t work\b", r"\bdidn'?t work\b", r"\bnot what i (?:asked|wanted|meant)\b",
        r"\brevert\b", r"\bundo (?:that|this|it)\b", r"\byou (?:broke|missed|forgot|ignored)\b",
        r"\bwhy did you\b", r"\bi said\b", r"\bstop\b.*\band\b", r"\bwrong\b.*\bfix\b",
        r"\bre-?read\b", r"\bread the\b.*\bagain\b",
    )
]


# Delegation and orchestration show up as ordinary tool calls in the parent transcript.
SUBAGENT_TOOLS = ("Agent", "Task")
WORKFLOW_TOOLS = ("Workflow",)
SKILL_TOOLS = ("Skill",)


@dataclass
class ToolCall:
    name: str
    is_error: bool = False
    input_excerpt: str = ""
    result_excerpt: str = ""


@dataclass
class Turn:
    role: str
    index: int
    text: str = ""
    tools: list[ToolCall] = field(default_factory=list)
    timestamp: str = ""
    is_sidechain: bool = False


@dataclass
class Session:
    session_id: str
    project: str
    path: str
    cwd: str = ""
    git_branch: str = ""
    version: str = ""
    turns: list[Turn] = field(default_factory=list)
    started: str = ""
    ended: str = ""

    @property
    def tool_counts(self) -> Counter:
        return Counter(call.name for turn in self.turns for call in turn.tools)

    @property
    def error_count(self) -> int:
        return sum(1 for turn in self.turns for call in turn.tools if call.is_error)

    @property
    def subagent_turns(self) -> int:
        """Subagent work, counted from delegation calls.

        `isSidechain` is present on every record but false in practice - a subagent's own
        transcript is not written into its parent's file. The delegation tool call is the
        observable signal.
        """
        counts = self.tool_counts
        return sum(counts[name] for name in SUBAGENT_TOOLS)

    @property
    def workflow_calls(self) -> int:
        return sum(self.tool_counts[name] for name in WORKFLOW_TOOLS)

    @property
    def skill_calls(self) -> int:
        return sum(self.tool_counts[name] for name in SKILL_TOOLS)

    @property
    def user_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.role == "user" and t.text and not t.is_sidechain]


@dataclass
class FailureEpisode:
    """One place the harness visibly went wrong, with enough context to diagnose it."""

    session_id: str
    project: str
    kind: str          # tool_error | correction | thrash | turn_limit
    turn_index: int
    detail: str = ""
    matched: str = ""  # which correction pattern fired; carries signal, not private text
    user_text: str = ""
    assistant_text: str = ""
    tools: list[str] = field(default_factory=list)


def _blocks(message: Any) -> list[dict[str, Any]]:
    content = (message or {}).get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [b for b in (content or []) if isinstance(b, dict)]


def parse_session(path: Path, include_text: bool = False, text_limit: int = 600) -> Session:
    """Parse one transcript. Message text is dropped unless include_text is set."""
    session = Session(session_id=path.stem, project=path.parent.name, path=str(path))
    pending: dict[str, ToolCall] = {}
    index = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        session.cwd = record.get("cwd") or session.cwd
        session.git_branch = record.get("gitBranch") or session.git_branch
        session.version = record.get("version") or session.version
        stamp = record.get("timestamp") or ""
        if stamp:
            session.started = session.started or stamp
            session.ended = stamp

        kind = record.get("type")
        if kind not in {"user", "assistant"}:
            continue
        blocks = _blocks(record.get("message"))
        turn = Turn(role=kind, index=index, timestamp=stamp,
                    is_sidechain=bool(record.get("isSidechain")))
        index += 1
        for block in blocks:
            btype = block.get("type")
            if btype == "text" and include_text:
                turn.text = (turn.text + " " + str(block.get("text", "")))[:text_limit].strip()
            elif btype == "text":
                turn.text = str(block.get("text", ""))[:text_limit] if kind == "user" else ""
            elif btype == "tool_use":
                call = ToolCall(name=str(block.get("name", "?")))
                if include_text:
                    call.input_excerpt = json.dumps(block.get("input", {}), default=str)[:400]
                pending[str(block.get("id", ""))] = call
                turn.tools.append(call)
            elif btype == "tool_result":
                call = pending.get(str(block.get("tool_use_id", "")))
                if call is not None:
                    call.is_error = bool(block.get("is_error"))
                    if include_text:
                        call.result_excerpt = str(block.get("content"))[:400]
        session.turns.append(turn)
    return session


def iter_session_paths(home: Path | None = None, project: str | None = None) -> Iterator[Path]:
    root = (home or claude_home()) / "projects"
    if not root.is_dir():
        return
    if project:
        directories = [root / project]
    else:
        directories = sorted(p for p in root.iterdir()
                             if p.is_dir() and not is_artifact_project(p.name))
    for directory in directories:
        if directory.is_dir():
            yield from sorted(directory.glob("*.jsonl"))


def load_sessions(home: Path | None = None, project: str | None = None, limit: int | None = None,
                  include_text: bool = False, min_turns: int = 4) -> list[Session]:
    sessions = []
    paths = sorted(iter_session_paths(home, project), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            session = parse_session(path, include_text=include_text)
        except OSError:
            continue
        if len(session.turns) >= min_turns:
            sessions.append(session)
        if limit and len(sessions) >= limit:
            break
    return sessions


def failure_episodes(session: Session, thrash_window: int = 6, thrash_threshold: int = 4,
                     correction_lookback: int = 40) -> list[FailureEpisode]:
    """Places the harness visibly failed, in this session."""
    episodes: list[FailureEpisode] = []

    for turn in session.turns:
        for call in turn.tools:
            if call.is_error:
                episodes.append(FailureEpisode(
                    session.session_id, session.project, "tool_error", turn.index,
                    detail=f"{call.name} returned an error",
                    assistant_text=call.result_excerpt, tools=[call.name]))

    # A correction is only meaningful after the agent did work - but the human never replies
    # to a tool call directly. Their turn lands after a closing assistant message with no tools,
    # so the work has to be looked for further back.
    for position, turn in enumerate(session.turns):
        if turn.role != "user" or not turn.text:
            continue
        window = session.turns[max(0, position - correction_lookback):position]
        recent_tools = [call.name for previous in window for call in previous.tools]
        if not recent_tools:
            continue
        for pattern in CORRECTION_PATTERNS:
            if pattern.search(turn.text):
                episodes.append(FailureEpisode(
                    session.session_id, session.project, "correction", turn.index,
                    detail=f"user corrected after {len(recent_tools)} tool call(s)",
                    matched=pattern.pattern, user_text=turn.text,
                    tools=sorted(set(recent_tools))))
                break

    # Repeatedly hammering one tool inside a short window is the harness thrashing.
    for start in range(0, max(0, len(session.turns) - thrash_window)):
        window = session.turns[start:start + thrash_window]
        counts = Counter(call.name for turn in window for call in turn.tools)
        for name, count in counts.items():
            if count >= thrash_threshold:
                episodes.append(FailureEpisode(
                    session.session_id, session.project, "thrash", window[0].index,
                    detail=f"{name} called {count}x within {thrash_window} turns", tools=[name]))
                break

    return episodes


def harness_report(sessions: Sequence[Session]) -> dict[str, Any]:
    """Aggregate signal across sessions: what this harness does, and where it breaks."""
    tools: Counter = Counter()
    errors: Counter = Counter()
    kinds: Counter = Counter()
    per_session = []
    for session in sessions:
        episodes = failure_episodes(session)
        tools.update(session.tool_counts)
        kinds.update(e.kind for e in episodes)
        for turn in session.turns:
            for call in turn.tools:
                if call.is_error:
                    errors[call.name] += 1
        per_session.append({
            "session_id": session.session_id,
            "project": session.project,
            "turns": len(session.turns),
            "tool_calls": sum(session.tool_counts.values()),
            "tool_errors": session.error_count,
            "subagent_calls": session.subagent_turns,
            "episodes": len(episodes),
            "version": session.version,
        })
    total_calls = sum(tools.values()) or 1
    read_like = sum(tools[name] for name in ("Read", "Grep", "Glob"))
    write_like = sum(tools[name] for name in ("Write", "Edit", "NotebookEdit"))
    return {
        "sessions": len(sessions),
        "tool_calls": sum(tools.values()),
        "tool_histogram": dict(tools.most_common()),
        "tool_errors": dict(errors.most_common()),
        "error_rate": round(sum(errors.values()) / total_calls, 4),
        "read_to_write_ratio": round(read_like / max(1, write_like), 2),
        "subagent_calls": sum(s.subagent_turns for s in sessions),
        "workflow_calls": sum(s.workflow_calls for s in sessions),
        "skill_calls": sum(s.skill_calls for s in sessions),
        "episode_kinds": dict(kinds),
        "per_session": per_session,
    }


def write_history_view(sessions: Sequence[Session], destination: Path,
                       include_text: bool = False) -> Path:
    """Materialize mined history where a proposer can grep it, alongside candidate traces."""
    destination = Path(destination)
    (destination / "sessions").mkdir(parents=True, exist_ok=True)
    report = harness_report(sessions)
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def dump(episode: FailureEpisode) -> dict[str, Any]:
        record = asdict(episode)
        if not include_text:
            # The privacy claim has to be true: a redacted view keeps the signal (which pattern
            # fired, which tools preceded it) and drops what the human actually typed.
            record["user_text"] = ""
            record["assistant_text"] = ""
        return record

    all_episodes: list[dict[str, Any]] = []
    for session in sessions:
        episodes = failure_episodes(session)
        all_episodes.extend(dump(e) for e in episodes)
        summary = {
            "session_id": session.session_id,
            "project": session.project,
            "cwd": session.cwd,
            "git_branch": session.git_branch,
            "version": session.version,
            "started": session.started,
            "ended": session.ended,
            "turns": len(session.turns),
            "tool_histogram": dict(session.tool_counts),
            "tool_errors": session.error_count,
            "subagent_calls": session.subagent_turns,
            "workflow_calls": session.workflow_calls,
            "skill_calls": session.skill_calls,
            "episodes": [dump(e) for e in episodes],
        }
        if include_text:
            summary["transcript"] = [
                {"role": t.role, "index": t.index, "sidechain": t.is_sidechain, "text": t.text,
                 "tools": [asdict(c) for c in t.tools]}
                for t in session.turns
            ]
        (destination / "sessions" / f"{session.session_id}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")

    with (destination / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in all_episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    (destination / "README.md").write_text(_VIEW_README, encoding="utf-8")
    return destination


_VIEW_README = """# Mined Claude Code history

Read from this machine's `~/.claude/projects/` transcripts. Nothing here was sent anywhere.

```
report.json              aggregate: tool histogram, error rate, read/write ratio, episode kinds
episodes.jsonl           one line per detected failure: tool_error | correction | thrash
sessions/<id>.json       per-session summary; `transcript` present only if mined with text
```

`episodes.jsonl` is the diagnostic substrate: each line is a place the harness visibly failed.
`correction` episodes are the strongest signal - a human told the agent it was wrong right after
it used tools. `thrash` means one tool was hammered inside a short window. `tool_error` is a
tool call that returned an error.

Text is redacted unless the run was mined with `--include-text`.
"""


def draft_tasks(sessions: Sequence[Session], limit: int = 20) -> list[dict[str, Any]]:
    """Draft eval tasks from correction episodes. Each needs a human to finish it.

    A verifiable task needs a test command, and history does not contain one. These drafts carry
    the request and the context; the `test_command` and `test_files` are left blank on purpose.
    """
    drafts = []
    for session in sessions:
        for episode in failure_episodes(session):
            if episode.kind != "correction" or not episode.user_text:
                continue
            drafts.append({
                "instruction": "",
                "_source": {"session": session.session_id, "project": session.project,
                            "cwd": session.cwd, "turn": episode.turn_index},
                "_correction": episode.user_text,
                "_tools_before_correction": episode.tools,
                "files": {},
                "test_files": {},
                "test_command": "",
                "_todo": "Fill instruction, files, test_files and test_command from this episode.",
            })
            if len(drafts) >= limit:
                return drafts
    return drafts


__all__ = ["ARTIFACT_MARKERS", "CORRECTION_PATTERNS", "SKILL_TOOLS", "SUBAGENT_TOOLS",
           "WORKFLOW_TOOLS", "is_artifact_project",
           "FailureEpisode", "Session", "ToolCall", "Turn",
           "claude_home", "draft_tasks", "failure_episodes", "harness_report",
           "iter_session_paths", "load_sessions", "parse_session", "project_slug",
           "write_history_view"]
