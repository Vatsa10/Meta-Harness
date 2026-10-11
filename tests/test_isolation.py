"""Candidate isolation: a child process with no keys, resource limits and preemptive timeouts."""
import json
import subprocess
import time

import pytest

from meta_harness.core import CandidateEvaluator
from meta_harness import isolation
from meta_harness.isolation import (PACKAGE_ROOT, IsolatedHarness, IsolationError, IsolationTimeout,
                                    docker_argv, scrubbed_env)
from meta_harness.sandboxes import preflight


class Recorder:
    def __init__(self, reply="pong"):
        self.prompts = []
        self.kwargs = []
        self.reply = reply

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
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
        names = ("UNIKEY_API_KEY", "GH_TOKEN", "AWS_SECRET_ACCESS_KEY", "DATABASE_URL", "FOO_PASSWORD")
        return {"output": "|".join(os.environ.get(n, "") for n in names)}
"""


def test_child_sees_no_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIKEY_API_KEY", "uk-test-not-a-real-key")
    monkeypatch.setenv("GH_TOKEN", "gh-test-not-a-real-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-test-not-real")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pw@host/db")
    monkeypatch.setenv("FOO_PASSWORD", "hunter2")
    with IsolatedHarness(_harness(tmp_path, ENV)) as child:
        result = child.run({"input": "x"}, Recorder(), Trace(), timeout=30.0)
    assert result["output"] == "||||"
    assert scrubbed_env({"DATABASE_URL": "v", "PATH": "p", "LC_ALL": "C"}) == {"PATH": "p", "LC_ALL": "C"}


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
    assert ("harness.thinking", {"step": 1}, {}) in trace.events


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


GRANDCHILD = """
import subprocess, sys, time
class Harness:
    def run(self, task, model, trace):
        beat = task["beat"]
        code = ("import time, pathlib\\n"
                "p = pathlib.Path(%r)\\n"
                "end = time.time() + 15\\n"
                "while time.time() < end:\\n"
                "    p.write_text(str(time.time()))\\n"
                "    time.sleep(0.1)\\n") % beat
        subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(30)
        return {"output": "never"}
"""


def test_timeout_kills_grandchildren(tmp_path):
    beat = tmp_path / "beat.txt"
    with IsolatedHarness(_harness(tmp_path, GRANDCHILD)) as child:
        child.start()
        if not child.limits_applied:
            pytest.skip("no job object: grandchildren cannot be reaped on this host")
        with pytest.raises(IsolationTimeout):
            child.run({"input": "x", "beat": str(beat)}, Recorder(), Trace(), timeout=3.0)
    assert beat.exists(), "the grandchild never started"
    time.sleep(1.0)
    first = beat.read_text(encoding="utf-8")
    time.sleep(1.0)
    assert beat.read_text(encoding="utf-8") == first, "a grandchild survived the kill"


GUARDS = """
import socket
class Harness:
    def run(self, task, model, trace):
        out = []
        try:
            open(task["secret"], "r")
            out.append("opened")
        except PermissionError:
            out.append("refused")
        except OSError:
            out.append("missing-but-allowed")
        try:
            socket.create_connection(("127.0.0.1", 9), timeout=1)
            out.append("connected")
        except PermissionError:
            out.append("blocked")
        except OSError:
            out.append("unblocked")
        with open("scratch.txt", "w") as f:
            f.write("ok")
        out.append("cwd-ok")
        return {"output": ",".join(out)}
"""


def test_guards_refuse_repo_env_and_sockets(tmp_path):
    secret = PACKAGE_ROOT / ".env"
    with IsolatedHarness(_harness(tmp_path, GUARDS)) as child:
        result = child.run({"input": "x", "secret": str(secret)}, Recorder(), Trace(), timeout=30.0)
    assert result["output"] == "refused,blocked,cwd-ok"
    assert not (tmp_path / "scratch.txt").exists()  # cwd is a fresh temp dir, not the candidate dir


def test_docker_argv_is_hardened(tmp_path):
    argv = docker_argv("n", tmp_path / "harness.py", 256, 30, "python:3.12-slim")
    joined = " ".join(argv)
    for flag in ("--network none", "--cap-drop ALL", "--read-only", "--tmpfs /tmp",
                 "--user 65534:65534", "--memory 256m", "--cpus 1"):
        assert flag in joined


def test_oversized_frame_kills_child(tmp_path, monkeypatch):
    monkeypatch.setattr(isolation, "MAX_FRAME_BYTES", 1024)
    body = """
import sys
class Harness:
    def run(self, task, model, trace):
        sys.__stdout__.write("x" * 4096)
        sys.__stdout__.flush()
        return {"output": "x"}
"""
    with IsolatedHarness(_harness(tmp_path, body)) as child:
        with pytest.raises(IsolationError, match="larger than"):
            child.run({"input": "x"}, Recorder(), Trace(), timeout=30.0)
        assert not child.alive()


def test_non_object_frame_rejected(tmp_path):
    body = """
import sys
class Harness:
    def run(self, task, model, trace):
        sys.__stdout__.write("[1, 2]\\n")
        sys.__stdout__.flush()
        return {"output": "x"}
"""
    with IsolatedHarness(_harness(tmp_path, body)) as child:
        with pytest.raises(IsolationError, match="not a JSON object"):
            child.run({"input": "x"}, Recorder(), Trace(), timeout=30.0)


def test_trace_events_prefixed_and_reserved_fields_dropped(tmp_path):
    body = """
class Harness:
    def run(self, task, model, trace):
        trace.event("task_end", {"reward": 1.0}, time=0, event="forged", error="e")
        return {"output": "x"}
"""
    trace = Trace()
    with IsolatedHarness(_harness(tmp_path, body)) as child:
        child.run({"input": "x"}, Recorder(), trace, timeout=30.0)
    assert trace.events == [("harness.task_end", {"reward": 1.0}, {"error": "e"})]


def test_model_kwargs_whitelisted_and_capped(tmp_path):
    body = """
class Harness:
    def run(self, task, model, trace):
        return {"output": model("p", temperature=0.1, max_tokens=10**9, api_key="x", base_url="y")}
"""
    model = Recorder()
    with IsolatedHarness(_harness(tmp_path, body), max_tokens=4096) as child:
        child.run({"input": "x"}, model, Trace(), timeout=30.0)
    assert model.kwargs == [{"temperature": 0.1, "max_tokens": 4096}]


def test_chat_in_child_gives_clear_error(tmp_path):
    body = """
class Harness:
    def run(self, task, model, trace):
        return {"output": model.chat("s", [])}
"""
    with IsolatedHarness(_harness(tmp_path, body)) as child:
        with pytest.raises(IsolationError, match=r"chat\(\) is unavailable under --isolation process"):
            child.run({"input": "x"}, Recorder(), Trace(), timeout=30.0)


PROBE_FILES = """
import importlib.util
from pathlib import Path
class Harness:
    def run(self, task, model, trace):
        out = []
        for name in task["paths"]:
            try:
                with open(name, encoding="utf-8") as handle:
                    out.append("read:" + handle.read().strip())
            except PermissionError:
                out.append("refused")
        here = Path(__file__).resolve().parent / "helper.py"
        out.append(here.parent.name)
        if not here.is_file():
            return {"output": ",".join(out)}
        spec = importlib.util.spec_from_file_location("helper", here)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        out.append(module.VALUE)
        return {"output": ",".join(out)}
"""


def test_seed_beside_dotenv_runs_from_run_dir_copy_and_cannot_read_it(tmp_path):
    from meta_harness.core import SearchConfig, SearchRunner

    user_dir = tmp_path / "repo"
    user_dir.mkdir()
    (user_dir / ".env").write_text("FAKE_KEY=not-a-real-value\n", encoding="utf-8")
    (user_dir / "notes.txt").write_text("repo file\n", encoding="utf-8")
    seed = _harness(user_dir, PROBE_FILES, "my_harness.py")
    seen = []

    def metric(task, result):
        seen.append(result["output"])
        return 1.0

    task = {"input": "x", "paths": [str(user_dir / ".env"), str(user_dir / "notes.txt")]}
    evaluator = CandidateEvaluator(Recorder(), isolation="process", metric=metric)
    runner = SearchRunner(SearchConfig(tmp_path / "store", iterations=0), evaluator, lambda *_: [])
    runner.run([seed], [task])
    # the repo dir is not an allowed root: neither its .env nor any other file opens
    assert seen and set(seen) == {"refused,refused,initial-0000"}


def test_dotenv_variants_refused_inside_candidate_dir_and_parent_not_allowed(tmp_path):
    cand = tmp_path / "cand"
    cand.mkdir()
    (tmp_path / "outside.txt").write_text("parent file\n", encoding="utf-8")
    names = (".env", ".env.local", ".Env.production")
    for name in names:
        (cand / name).write_text("FAKE=1\n", encoding="utf-8")
    (cand / "helper.py").write_text("VALUE = 'h'\n", encoding="utf-8")
    paths = [str(cand / n) for n in names] + [str(tmp_path / "outside.txt"), str(cand / "helper.py")]
    with IsolatedHarness(_harness(cand, PROBE_FILES)) as child:
        result = child.run({"input": "x", "paths": paths}, Recorder(), Trace(), timeout=30.0)
    assert result["output"] == "refused,refused,refused,refused,read:VALUE = 'h',cand,h"
