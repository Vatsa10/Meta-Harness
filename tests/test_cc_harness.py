import json
import subprocess
from pathlib import Path

import pytest

from meta_harness.cc_harness import (AgentConfig, AgentRunError, ClaudeCodeHarness,
                                     prepare_workspace, run_claude_code)
from meta_harness.core import TraceRecorder
from meta_harness.datasets import agent_tasks
from meta_harness.metrics import agent_metric

TASK = {"instruction": "fix it", "test_command": "exit 0", "files": {"m.py": "x = 1\n"}}


def _stream(*rows):
    return "\n".join(json.dumps(r) for r in rows)


def _result_row(turns=3, is_error=False, subtype="success"):
    return {"type": "result", "subtype": subtype, "is_error": is_error, "num_turns": turns,
            "usage": {"input_tokens": 120, "cache_read_input_tokens": 4000},
            "total_cost_usd": 0.02, "result": "done"}


def _fake_run(monkeypatch, stdout, capture=None, returncode=0):
    def run(argv, **kwargs):
        if capture is not None:
            capture.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, returncode, stdout, "")

    monkeypatch.setattr(subprocess, "run", run)


# --- workspace -------------------------------------------------------------

def test_workspace_is_seeded_from_task_files(tmp_path: Path):
    ws = prepare_workspace(TASK, tmp_path)
    assert (ws / "m.py").read_text() == "x = 1\n"
    assert ws.parent == tmp_path


def test_workspace_rejects_path_escape(tmp_path: Path):
    with pytest.raises(AgentRunError, match="escapes"):
        prepare_workspace({"files": {"../evil.py": "x"}}, tmp_path)


def test_each_run_gets_a_fresh_workspace(tmp_path: Path):
    assert prepare_workspace(TASK, tmp_path) != prepare_workspace(TASK, tmp_path)


# --- the tool ceiling ------------------------------------------------------

def test_candidate_cannot_widen_the_tool_ceiling(monkeypatch):
    monkeypatch.delenv("META_HARNESS_AGENT_BASH", raising=False)
    config = AgentConfig(allowed_tools=("Read", "Bash", "WebFetch", "Task"))
    assert config.tools() == ["Read"]


def test_bash_is_available_only_when_enabled(monkeypatch):
    monkeypatch.setenv("META_HARNESS_AGENT_BASH", "1")
    assert "Bash" in AgentConfig(allowed_tools=("Read", "Bash")).tools()
    monkeypatch.delenv("META_HARNESS_AGENT_BASH")
    assert "Bash" not in AgentConfig(allowed_tools=("Read", "Bash")).tools()


def test_empty_tool_selection_still_yields_one_tool():
    assert AgentConfig(allowed_tools=()).tools() == ["Read"]


# --- invocation ------------------------------------------------------------

def test_invocation_carries_the_candidates_knobs(monkeypatch, tmp_path: Path):
    calls = []
    _fake_run(monkeypatch, _stream(_result_row()), capture=calls)
    ws = prepare_workspace(TASK, tmp_path)
    config = AgentConfig(append_system_prompt="BE CAREFUL", max_turns=7, model="sonnet",
                         prompt_template="TASK: {instruction}")
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        run_claude_code(ws, config, TASK, trace)
    argv, kwargs = calls[0]
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert argv[argv.index("--max-turns") + 1] == "7"
    assert argv[argv.index("--append-system-prompt") + 1] == "BE CAREFUL"
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in argv
    assert kwargs["input"] == "TASK: fix it"
    assert kwargs["cwd"] == ws


def test_gateway_env_is_stripped_from_the_agent(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
    _fake_run(monkeypatch, _stream(_result_row()), capture=calls)
    ws = prepare_workspace(TASK, tmp_path)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        run_claude_code(ws, AgentConfig(), TASK, trace)
    env = calls[0][1]["env"]
    assert "ANTHROPIC_BASE_URL" not in env and "ANTHROPIC_AUTH_TOKEN" not in env


# --- traces ----------------------------------------------------------------

def test_transcript_becomes_trace_events(monkeypatch, tmp_path: Path):
    stdout = _stream(
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "reading the file"},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "m.py"}}]}},
        {"type": "user", "message": {"content": [{"content": "1  x = 1"}]}},
        _result_row(),
    )
    _fake_run(monkeypatch, stdout)
    ws = prepare_workspace(TASK, tmp_path)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        run = run_claude_code(ws, AgentConfig(), TASK, trace)
    events = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    roles = [e["payload"].get("role") for e in events if e["event"] == "agent_step"]
    assert roles == ["assistant", "tool_use", "tool_result", "result"]
    assert run.turns == 3
    # Context cost is the session's real input, cache reads included.
    assert run.input_tokens == 4120


def test_context_cost_is_recorded_on_the_trace(monkeypatch, tmp_path: Path):
    _fake_run(monkeypatch, _stream(_result_row()))
    ws = prepare_workspace(TASK, tmp_path)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        run_claude_code(ws, AgentConfig(), TASK, trace)
        assert trace.context_cost == 4120


def test_agent_error_is_recorded_not_raised(monkeypatch, tmp_path: Path):
    _fake_run(monkeypatch, _stream(_result_row(is_error=True, subtype="error_max_turns")))
    ws = prepare_workspace(TASK, tmp_path)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        run = run_claude_code(ws, AgentConfig(), TASK, trace)
    assert run.completed is False
    assert "error_max_turns" in run.error


# --- scoring ---------------------------------------------------------------

def test_metric_runs_the_task_test_command(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("META_HARNESS_WORKSPACE_ROOT", raising=False)
    ws = prepare_workspace(TASK, tmp_path)
    assert agent_metric(str(ws), {"test_command": "exit 0"}) == 1.0
    assert agent_metric(str(ws), {"test_command": "exit 1"}) == 0.0


def test_metric_refuses_a_workspace_outside_the_run_root(tmp_path: Path, monkeypatch):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.setenv("META_HARNESS_WORKSPACE_ROOT", str(tmp_path / "root"))
    (tmp_path / "root").mkdir()
    # A candidate cannot point scoring at a directory that already passes.
    assert agent_metric(str(outside), {"test_command": "exit 0"}) == 0.0


def test_metric_without_a_workspace_is_zero():
    assert agent_metric("", {"test_command": "exit 0"}) == 0.0


# --- harness plumbing ------------------------------------------------------

def test_validation_probe_does_not_launch_an_agent(tmp_path: Path, monkeypatch):
    def boom(*_, **__):
        raise AssertionError("should not run the agent for a validation probe")

    monkeypatch.setattr(subprocess, "run", boom)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        assert ClaudeCodeHarness().run({"input": "validation", "label": "ok"}, None, trace) == ""


def test_harness_returns_the_workspace_path(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("META_HARNESS_WORKSPACE_ROOT", str(tmp_path))
    _fake_run(monkeypatch, _stream(_result_row()))
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        out = ClaudeCodeHarness().run(TASK, None, trace)
    assert Path(out).is_dir() and Path(out).parent == tmp_path


# --- dataset ---------------------------------------------------------------

def test_agent_tasks_requires_a_test_command():
    with pytest.raises(ValueError, match="test_command"):
        agent_tasks([{"instruction": "do it", "files": {"a.py": ""}}])


def test_agent_tasks_requires_a_workspace_source():
    with pytest.raises(ValueError, match="files.*repo"):
        agent_tasks([{"instruction": "do it", "test_command": "exit 0"}])


def test_agent_tasks_defaults_timeout():
    task = agent_tasks([{"instruction": "x", "test_command": "exit 0", "files": {"a.py": ""}}])[0]
    assert task["timeout"] == 900.0
