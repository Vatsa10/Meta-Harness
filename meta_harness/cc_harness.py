"""Search over Claude Code's own harness.

In the paper's Table 7, Claude Code is itself one of the baseline harnesses being compared
(58.0% on Opus 4.6, 27.5% on Haiku 4.5), and searched harnesses beat it by 18.4 and 10.1 points
on the same frozen model. This module makes Claude Code the executor so that comparison can be
run against your own tasks.

What a candidate controls - the harness, in the paper's sense of "the code that determines what
information to store, retrieve, and present to the model":

- the system-prompt doctrine appended to the session
- how the task instruction is framed before it is sent
- which tools the agent may use
- the turn budget
- the model
- any custom subagents the session can call

What a candidate does NOT control: the verification. Tasks are scored by running their own test
command afterwards, outside the agent's reach, so a candidate cannot mark its own homework.

Each task runs in a throwaway workspace seeded from the task definition. The agent never touches
the real repository.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# Ceiling on what any candidate may grant itself. A proposed harness is LLM-written; it may pick
# from this set but cannot widen it. Bash is off unless explicitly enabled, because the agent does
# not need it to edit files - the test command is run by us, afterwards.
TOOL_CEILING = ("Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit", "TodoWrite")
BASH_ENV = "META_HARNESS_AGENT_BASH"
WORKSPACE_ROOT_ENV = "META_HARNESS_WORKSPACE_ROOT"


class AgentRunError(RuntimeError):
    pass


@dataclass
class AgentConfig:
    """One candidate harness for Claude Code."""

    append_system_prompt: str = ""
    allowed_tools: Sequence[str] = ("Read", "Write", "Edit", "Glob", "Grep")
    max_turns: int = 30
    model: str = "haiku"
    permission_mode: str = "acceptEdits"
    agents: Mapping[str, Any] | None = None
    prompt_template: str = "{instruction}"
    # A candidate may ship skills of its own: {"<name>": "<SKILL.md body>"}. They are written
    # into a throwaway plugin next to the workspace and loaded with --plugin-dir, which is how
    # skills, subagents and their doctrine reach a real session.
    skills: Mapping[str, str] | None = None
    settings: Mapping[str, Any] | None = None

    def render(self, task: Mapping[str, Any], workspace: Path) -> str:
        return self.prompt_template.format(
            instruction=str(task.get("instruction", "")),
            workspace=str(workspace),
            files="\n".join(sorted(p.name for p in workspace.iterdir() if p.is_file())),
        )

    def tools(self) -> list[str]:
        ceiling = set(TOOL_CEILING)
        if os.environ.get(BASH_ENV) == "1":
            ceiling.add("Bash")
        return [tool for tool in self.allowed_tools if tool in ceiling] or ["Read"]


@dataclass
class AgentRun:
    transcript: list[dict[str, Any]] = field(default_factory=list)
    workspace: Path | None = None
    turns: int = 0
    input_tokens: int = 0
    cost_usd: float = 0.0
    completed: bool = False
    error: str | None = None


def prepare_workspace(task: Mapping[str, Any], root: Path | None = None) -> Path:
    """Seed a throwaway workspace from the task. Never touches the real repository."""
    root = Path(root or os.environ.get(WORKSPACE_ROOT_ENV) or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix="mh-agent-", dir=root))
    source = task.get("repo")
    if source:
        source_path = Path(str(source))
        if not source_path.is_dir():
            raise AgentRunError(f"repo not found: {source_path}")
        shutil.copytree(source_path, workspace, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".git", "__pycache__", "node_modules", ".venv"))
    for relative, content in (task.get("files") or {}).items():
        target = workspace / Path(str(relative))
        if not str(target.resolve()).startswith(str(workspace.resolve())):
            raise AgentRunError(f"task file escapes the workspace: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(content), encoding="utf-8")
    return workspace


def write_candidate_plugin(skills: Mapping[str, str], root: Path) -> Path:
    """Materialize candidate-authored skills as a loadable plugin directory."""
    plugin = Path(root) / "candidate-plugin"
    (plugin / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (plugin / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": "candidate-harness", "version": "0.0.0",
                    "description": "Skills proposed by a candidate harness"}, indent=2),
        encoding="utf-8")
    for name, body in skills.items():
        safe = re.sub(r"[^a-z0-9-]+", "-", str(name).lower()).strip("-") or "skill"
        directory = plugin / "skills" / safe
        directory.mkdir(parents=True, exist_ok=True)
        text = str(body)
        if not text.lstrip().startswith("---"):
            header = (f"---\nname: {safe}\n"
                      "description: Proposed by a candidate harness.\n---\n\n")
            text = header + text
        (directory / "SKILL.md").write_text(text, encoding="utf-8")
    return plugin


def _summarize(transcript: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compact the stream into the events a proposer actually reads."""
    events: list[dict[str, Any]] = []
    for row in transcript:
        kind = row.get("type")
        if kind == "assistant":
            for block in row.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    events.append({"role": "assistant", "text": str(block["text"])[:2000]})
                elif block.get("type") == "tool_use":
                    events.append({"role": "tool_use", "name": block.get("name"),
                                   "input": json.dumps(block.get("input", {}), default=str)[:1500]})
        elif kind == "user":
            content = row.get("message", {}).get("content")
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict):
                    events.append({"role": "tool_result", "content": str(block.get("content"))[:1500]})
        elif kind == "result":
            events.append({"role": "result", "subtype": row.get("subtype"),
                           "turns": row.get("num_turns"), "is_error": row.get("is_error")})
    return events


def run_claude_code(workspace: Path, config: AgentConfig, task: Mapping[str, Any],
                    trace: Any, binary: str = "claude", timeout: float = 900.0) -> AgentRun:
    """One agent session against one task, in one throwaway workspace."""
    prompt = config.render(task, workspace)
    argv = [
        binary, "-p",
        "--model", config.model,
        "--output-format", "stream-json", "--verbose",
        "--max-turns", str(config.max_turns),
        "--allowed-tools", ",".join(config.tools()),
        "--permission-mode", config.permission_mode,
        "--add-dir", str(workspace),
        "--setting-sources", "",
        "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
    ]
    if config.append_system_prompt:
        argv += ["--append-system-prompt", config.append_system_prompt]
    if config.agents:
        argv += ["--agents", json.dumps(dict(config.agents))]
    if config.skills:
        plugin = write_candidate_plugin(config.skills, workspace.parent)
        argv += ["--plugin-dir", str(plugin)]
    if config.settings:
        settings_path = workspace.parent / f"{workspace.name}-settings.json"
        settings_path.write_text(json.dumps(dict(config.settings)), encoding="utf-8")
        argv += ["--settings", str(settings_path)]

    environment = os.environ.copy()
    for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY",
                 "CLAUDE_PLUGIN_ROOT"):
        environment.pop(name, None)

    trace.event("agent_start", {"workspace": str(workspace), "model": config.model,
                                "max_turns": config.max_turns, "tools": config.tools(),
                                "prompt": prompt[:4000],
                                "system_prompt": config.append_system_prompt[:4000],
                                "skills": sorted(config.skills or {}),
                                "agents": sorted(config.agents or {}),
                                "settings": dict(config.settings or {})})
    started = time.perf_counter()
    try:
        completed = subprocess.run(argv, input=prompt, cwd=workspace, env=environment,
                                   capture_output=True, text=True, timeout=timeout,
                                   encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        trace.event("agent_timeout", {"seconds": timeout})
        return AgentRun(workspace=workspace, error=f"agent timed out after {timeout}s")
    except OSError as error:
        raise AgentRunError(f"cannot run {binary!r}: {error}") from error

    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)

    run = AgentRun(transcript=rows, workspace=workspace)
    for row in rows:
        if row.get("type") == "result":
            usage = row.get("usage") or {}
            run.turns = int(row.get("num_turns", 0) or 0)
            run.input_tokens = int(usage.get("input_tokens", 0) or 0) + \
                int(usage.get("cache_read_input_tokens", 0) or 0)
            run.cost_usd = float(row.get("total_cost_usd", 0.0) or 0.0)
            run.completed = not row.get("is_error")
            if row.get("is_error"):
                run.error = f"{row.get('subtype')}: {str(row.get('result'))[:300]}"
    if not rows:
        run.error = (completed.stderr or "agent produced no events")[:400]

    # The transcript is the diagnostic substrate the next proposer reads. Store it whole.
    for event in _summarize(rows):
        trace.event("agent_step", event)
    trace.event("agent_end", {"turns": run.turns, "input_tokens": run.input_tokens,
                              "cost_usd": run.cost_usd, "completed": run.completed,
                              "error": run.error,
                              "elapsed_seconds": round(time.perf_counter() - started, 1)})
    trace.add_context_cost(run.input_tokens)
    return run


class ClaudeCodeHarness:
    """Base for candidates. Override `config()`; the runner does the rest.

    `run` returns the workspace path. Scoring happens afterwards, in `agent_metric`, by running
    the task's own test command there - the candidate never reports its own score.
    """

    binary = "claude"
    timeout = 900.0

    def config(self, task: Mapping[str, Any]) -> AgentConfig:
        return AgentConfig()

    def run(self, task: Mapping[str, Any], model: Any, trace: Any) -> str:
        if "instruction" not in task:
            # Interface-validation probe from sandbox.py sends {"input", "label"}.
            trace.event("agent_skip", {"reason": "no instruction"})
            return ""
        workspace = prepare_workspace(task)
        run_claude_code(workspace, self.config(task), task, trace,
                        binary=self.binary, timeout=float(task.get("timeout") or self.timeout))
        return str(workspace)


__all__ = ["AgentConfig", "AgentRun", "AgentRunError", "ClaudeCodeHarness", "TOOL_CEILING",
           "prepare_workspace", "run_claude_code", "write_candidate_plugin"]
