"""Candidate isolation: a child process with no keys, resource limits and preemptive timeouts."""
import json
import subprocess
import time

import pytest

from meta_harness.core import CandidateEvaluator
from meta_harness.isolation import IsolatedHarness, IsolationError, IsolationTimeout, scrubbed_env
from meta_harness.sandboxes import preflight


class Recorder:
    def __init__(self, reply="pong"):
        self.prompts = []
        self.reply = reply

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.reply


class Trace:
    def __init__(self):
        self.events = []
        self.context_cost = 0.0

    def event(self, name, payload=None, **fields):
        self.events.append((name, payload, fields))


def _harness(tmp_path, body, name="harness.py"):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


SLOW = """
import time
class Harness:
    def run(self, task, model, trace):
        if task["input"] == "slow":
            time.sleep(30)
        return {"output": task["input"]}
"""


def test_slow_task_is_killed_preemptively_and_scored_timeout(tmp_path):
    source = _harness(tmp_path, SLOW)
    evaluator = CandidateEvaluator(Recorder(), task_timeout=2.0, isolation="process")
    started = time.monotonic()
    result = evaluator.evaluate(source, [{"input": "slow", "expected": "slow"},
                                         {"input": "fast", "expected": "fast"}],
                                "c1", tmp_path / "trace.jsonl")
    elapsed = time.monotonic() - started
    assert elapsed < 10.0
    assert result.stop_reasons.get("timeout") == 1
    assert result.score == 0.5  # the child was restarted and the next task still ran
    ends = [json.loads(line) for line in (tmp_path / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    slow_end = [e for e in ends if e["event"] == "task_end"][0]
    assert slow_end["payload"]["stop_reason"] == "timeout" and "timeout" in slow_end["error"]


def test_isolated_run_raises_timeout_and_kills_child(tmp_path):
    with IsolatedHarness(_harness(tmp_path, SLOW)) as child:
        started = time.monotonic()
        with pytest.raises(IsolationTimeout, match="timeout"):
            child.run({"input": "slow"}, Recorder(), Trace(), timeout=2.0)
        assert time.monotonic() - started < 10.0
        assert not child.alive()


ENV = """
import os
class Harness:
    def run(self, task, model, trace):
        return {"output": os.environ.get("UNIKEY_API_KEY", "") + "|" + os.environ.get("GH_TOKEN", "")}
"""


def test_child_sees_no_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIKEY_API_KEY", "uk-test-not-a-real-key")
    monkeypatch.setenv("GH_TOKEN", "gh-test-not-a-real-token")
    with IsolatedHarness(_harness(tmp_path, ENV)) as child:
        result = child.run({"input": "x"}, Recorder(), Trace(), timeout=30.0)
    assert result["output"] == "|"
    assert "UNIKEY_API_KEY" not in scrubbed_env({"UNIKEY_API_KEY": "v", "PATH": "p"})


HOG = """
class Harness:
    def run(self, task, model, trace):
        block = bytearray(4 * 1024 ** 3)
        return {"output": len(block)}
"""


def test_memory_limit_stops_a_4gb_allocation(tmp_path):
    child = IsolatedHarness(_harness(tmp_path, HOG), memory_mb=256)
    try:
        child.start()
        if not child.limits_applied:
            pytest.skip(f"memory limit cannot be applied on this platform: {child.warnings}")
        with pytest.raises(IsolationError):
            child.run({"input": "x"}, Recorder(), Trace(), timeout=60.0)
    finally:
        child.close()


ECHO = """
class Harness:
    def run(self, task, model, trace):
        trace.event("thinking", {"step": 1})
        return {"output": model("question: " + task["input"])}
"""


def test_model_calls_round_trip_through_the_parent(tmp_path):
    model = Recorder(reply="answer-42")
    trace = Trace()
    with IsolatedHarness(_harness(tmp_path, ECHO)) as child:
        result = child.run({"input": "life"}, model, trace, timeout=30.0)
    assert model.prompts == ["question: life"]
    assert result["output"] == "answer-42"
    assert ("thinking", {"step": 1}, {}) in trace.events


def test_evaluator_meters_isolated_model_calls(tmp_path):
    model = Recorder(reply="answer-42")
    result = CandidateEvaluator(model, isolation="process").evaluate(
        _harness(tmp_path, ECHO), [{"input": "a", "expected": "answer-42"}], "c", tmp_path / "t.jsonl")
    assert result.score == 1.0 and result.metrics["model_calls"] == 1.0
    assert model.prompts == ["question: a"]


LEARNER = """
class Harness:
    def __init__(self):
        self.seen = []
    def run(self, task, model, trace):
        return {"output": ",".join(self.seen)}
    def learn(self, task, result, model):
        self.seen.append(task["input"])
"""


def test_state_persists_between_run_and_learn(tmp_path):
    with IsolatedHarness(_harness(tmp_path, LEARNER)) as child:
        child.run({"input": "a"}, Recorder(), Trace(), learn=True, timeout=30.0)
        child.run({"input": "b"}, Recorder(), Trace(), learn=True, timeout=30.0)
        assert child.run({"input": "q"}, Recorder(), Trace(), timeout=30.0)["output"] == "a,b"
        child.snapshot()
        child.run({"input": "c"}, Recorder(), Trace(), learn=True, timeout=30.0)
        child.restore()
        assert child.run({"input": "q"}, Recorder(), Trace(), timeout=30.0)["output"] == "a,b"


def test_online_protocol_runs_isolated(tmp_path):
    evaluator = CandidateEvaluator(Recorder(), protocol="online", isolation="process",
                                   train_tasks=[{"input": "a", "expected": "x"}, {"input": "b", "expected": "x"}])
    result = evaluator.evaluate(_harness(tmp_path, LEARNER),
                                [{"input": "t1", "expected": "a,b"}, {"input": "t2", "expected": "a,b"}],
                                "c", tmp_path / "t.jsonl")
    assert result.score == 1.0  # trained state visible, test tasks never learned


def test_unknown_isolation_mode_rejected():
    with pytest.raises(ValueError):
        CandidateEvaluator(Recorder(), isolation="vm")


def _docker_ready() -> str | None:
    check = preflight()
    if not check.ok:
        return f"docker unavailable: {check.problem}"
    image = subprocess.run(["docker", "image", "inspect", "python:3.12-slim"], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=60)
    if image.returncode != 0:
        return "image python:3.12-slim not present locally (tests never pull)"
    return None


def test_docker_mode_runs_one_task(tmp_path):
    reason = _docker_ready()
    if reason:
        pytest.skip(reason)
    model = Recorder(reply="from-parent")
    with IsolatedHarness(_harness(tmp_path, ECHO), mode="docker", memory_mb=256) as child:
        result = child.run({"input": "d"}, model, Trace(), timeout=120.0)
    assert result["output"] == "from-parent" and model.prompts == ["question: d"]
