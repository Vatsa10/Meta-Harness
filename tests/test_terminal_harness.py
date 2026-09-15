import json
from pathlib import Path

import pytest

from meta_harness.core import TraceRecorder
from meta_harness.datasets import terminal_tasks
from meta_harness.metrics import METRICS
from meta_harness.terminal_harness import DockerPolicy, TerminalHarness


def test_terminal_tasks_adapter_defaults():
    tasks = terminal_tasks([{"instruction": "make a file", "test_command": "test -f out.txt"}])
    assert tasks[0]["instruction"] == "make a file"
    assert tasks[0]["image"] is None
    assert tasks[0]["timeout"] == 300.0


def test_terminal_tasks_requires_an_instruction():
    with pytest.raises(ValueError):
        terminal_tasks([{"test_command": "true"}])


def test_docker_policy_wraps_commands():
    policy = DockerPolicy(container="c1", workdir="/app")
    wrapped = policy.wrap("ls -la")
    assert wrapped[:4] == ["docker", "exec", "-w", "/app"]
    assert wrapped[4] == "c1"
    assert wrapped[-1] == "ls -la"


def test_terminal_harness_refuses_local_shell_by_default(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("META_HARNESS_ALLOW_LOCAL_SHELL", raising=False)
    harness = TerminalHarness()
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        with pytest.raises(PermissionError, match="allow_local_shell"):
            harness.run({"instruction": "echo hi", "image": None}, lambda p, **_: "{}", trace)


def test_validation_probe_does_not_touch_the_shell(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("META_HARNESS_ALLOW_LOCAL_SHELL", raising=False)
    harness = TerminalHarness()
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        assert harness.run({"input": "validation", "label": "ok"}, lambda p, **_: "{}", trace) == ""


def test_terminal_harness_runs_locally_when_opted_in(tmp_path: Path):
    replies = iter(['{"command": "echo ready", "done": false}', '{"done": true, "answer": "finished"}'])
    harness = TerminalHarness(allow_local_shell=True, max_steps=4)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        answer = harness.run({"instruction": "say ready", "workdir": str(tmp_path)},
                             lambda prompt, **_: next(replies), trace)
    assert answer == "finished"
    events = [json.loads(line)["event"] for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert "terminal_step" in events


def test_terminal_metric_runs_the_test_command(tmp_path: Path):
    metric = METRICS["terminal"]
    assert metric("done", {"test_command": "exit 0", "workdir": str(tmp_path)}) == 1.0
    assert metric("done", {"test_command": "exit 1", "workdir": str(tmp_path)}) == 0.0


def test_terminal_metric_without_command_is_zero():
    assert METRICS["terminal"]("done", {"instruction": "x"}) == 0.0
