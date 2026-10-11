import json
import subprocess
from pathlib import Path

import pytest

import meta_harness.terminal as terminal_mod
from meta_harness.domains import get_domain
from meta_harness.loaders import terminalbench
from meta_harness.metrics import terminal_metric
from meta_harness.terminal import Container

domain = get_domain("terminal")


class FakeContainer:
    """Stands in for terminal.Container: records start / exec_test / put / remove."""

    log: list = []

    def __init__(self, image, workdir="/app", network=False):
        self.image, self.workdir, self.network = image, workdir, network
        self.id = None

    def start(self):
        self.id = "cid-1"
        FakeContainer.log.append(("start", self.image, self.network))
        return self.id

    @classmethod
    def attach(cls, container_id, workdir="/app"):
        c = cls("", workdir)
        c.id = container_id
        return c

    # A tiny shared filesystem: the agent, the wipe, the copy-in and the tests all act on it.
    fs: dict = {}
    solved = True
    rewriter = False      # an agent background loop that rewrites the reward file until killed
    changes: list = []    # what `docker diff` reports
    symlink = False       # /logs or /logs/verifier is a symlink

    def restart(self, timeout=120):
        FakeContainer.log.append(("restart", self.id))
        FakeContainer.rewriter = False

    def diff(self, timeout=120):
        FakeContainer.log.append(("diff", self.id))
        return list(FakeContainer.changes)

    def exec(self, command, timeout):
        FakeContainer.log.append(("exec", self.id, command))
        if FakeContainer.rewriter:
            FakeContainer.fs["/logs/verifier/reward.txt"] = "1"
        if command == domain.SYMLINK_CHECK:
            return (0, "") if FakeContainer.symlink else (1, "")
        if command == "rm -rf /tests /logs/verifier":
            for key in [k for k in FakeContainer.fs if k.startswith(("/tests", "/logs/verifier"))]:
                del FakeContainer.fs[key]
            if FakeContainer.rewriter:
                FakeContainer.fs["/logs/verifier/reward.txt"] = "1"
        return 0, ""

    def exec_test(self, command, timeout):
        """Like TB2: run whatever is at /tests; the real test writes reward 1 only when solved;
        the verdict is the reward file."""
        FakeContainer.log.append(("test", self.id, self.workdir, command))
        if FakeContainer.rewriter:
            FakeContainer.fs["/logs/verifier/reward.txt"] = "1"
        if FakeContainer.fs.get("/tests") == "REAL" and FakeContainer.solved:
            FakeContainer.fs["/logs/verifier/reward.txt"] = "1"
        elif FakeContainer.fs.get("/tests") == "PLANTED":
            FakeContainer.fs["/logs/verifier/reward.txt"] = "1"
        return (0, "passed") if FakeContainer.fs.get("/logs/verifier/reward.txt") == "1" else (1, "failed")

    def put(self, local, remote):
        FakeContainer.log.append(("put", str(local), remote))
        if remote not in FakeContainer.fs:  # docker cp into an existing dir nests instead
            FakeContainer.fs[remote] = "REAL"

    def remove(self):
        FakeContainer.log.append(("remove", self.id))


class SeenHarness:
    def __init__(self, fail=False):
        self.seen, self.fail = [], fail

    def run(self, task, model, trace):
        self.seen.append(dict(task))
        if self.fail:
            raise RuntimeError("harness blew up")
        return {"output": "done", "stop_reason": "done"}


class PlantingHarness:
    """A cheating agent: plants a passing reward file and its own /tests, solves nothing."""

    def run(self, task, model, trace):
        FakeContainer.fs["/logs/verifier/reward.txt"] = "1"
        FakeContainer.fs["/tests"] = "PLANTED"
        return {"output": "done"}


@pytest.fixture(autouse=True)
def fake_container(monkeypatch):
    FakeContainer.log = []
    FakeContainer.fs = {}
    FakeContainer.solved = True
    FakeContainer.rewriter = False
    FakeContainer.changes = []
    FakeContainer.symlink = False
    monkeypatch.setattr(terminal_mod, "Container", FakeContainer)


TASK = {"id": "fix-the-bug", "instruction": "fix it", "image": "img", "workdir": "/app",
        "test_command": "pytest /tests", "test_files": [["tests_dir", "/tests"]]}


def test_metric_runs_in_the_container_the_harness_used():
    harness = SeenHarness()
    score, result = domain.evaluate_task(harness, TASK, None, None, container_factory=FakeContainer)
    assert score == 1.0 and result["output"] == "done"
    assert harness.seen[0]["container"] == "cid-1"
    assert "test_command" not in harness.seen[0] and "test_files" not in harness.seen[0]
    assert ("test", "cid-1", "/app", "pytest /tests") in FakeContainer.log
    kinds = [e[0] for e in FakeContainer.log]
    assert kinds.index("put") < kinds.index("test") < kinds.index("remove")


def test_metric_without_a_container_scores_zero_and_runs_nothing(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    task = {"instruction": "x", "test_command": "exit 0", "workdir": "."}
    assert terminal_metric(task, {"output": "done"}) == 0.0
    assert terminal_metric("done", task) == 0.0  # legacy argument order
    assert domain.metric(task, {"output": "done"}) == 0.0
    assert FakeContainer.log == []


def test_metric_without_test_command_is_zero():
    assert terminal_metric({"instruction": "x", "container": "c"}, {"output": "x"}) == 0.0


def test_domain_removes_the_container_when_the_harness_raises():
    with pytest.raises(RuntimeError, match="blew up"):
        domain.evaluate_task(SeenHarness(fail=True), TASK, None, None, container_factory=FakeContainer)
    assert FakeContainer.log[-1] == ("remove", "cid-1")
    assert not any(e[0] == "test" for e in FakeContainer.log)


def test_domain_refuses_a_task_without_an_image():
    with pytest.raises(ValueError, match="no image"):
        domain.evaluate_task(SeenHarness(), {**TASK, "image": None}, None, None, FakeContainer)


def test_in_container_wrapper_feeds_the_domain_metric():
    wrapped = domain.HARNESS_WRAPPER(SeenHarness(), container_factory=FakeContainer)
    result = wrapped.run(TASK, None, None)
    assert domain.METRICS["terminal"](TASK, result) == 1.0
    forged = {"output": "x", "_terminal_score_token": "made-up"}
    assert domain.METRICS["terminal"](TASK, forged) == 0.0


def test_domain_contract():
    assert domain.PROTOCOL == "plain" and domain.SEARCH_EQUALS_TEST
    assert (domain.SEED_DIR / "terminus2.py").is_file() and (domain.SEED_DIR / "terminus_kira.py").is_file()
    assert "PLANNING" in domain.PROMPT and "BASH_ONLY" in domain.PROMPT
    assert domain.adapt([{"instruction": "a"}])[0]["workdir"] == "/app"


# --------------------------------------------------------------------------- loader + audit
def _tb_checkout(root: Path) -> Path:
    yaml_task = root / "tasks" / "hello-world"
    (yaml_task / "tests").mkdir(parents=True)
    (yaml_task / "task.yaml").write_text(
        "instruction: |-\n  Create /app/hello.txt\n  containing Hello.\ndifficulty: easy\n"
        "max_agent_timeout_sec: 300\n", encoding="utf-8")
    (yaml_task / "Dockerfile").write_text("FROM alpine\n", encoding="utf-8")
    (yaml_task / "run-tests.sh").write_text("pytest\n", encoding="utf-8")
    (yaml_task / "tests" / "test_outputs.py").write_text(
        "def test_it():\n    assert open('/app/hello.txt').read() == 'Hello, world from TB!'\n", encoding="utf-8")
    toml_task = root / "tasks" / "sort-numbers"
    (toml_task / "tests").mkdir(parents=True)
    (toml_task / "environment").mkdir()
    (toml_task / "task.toml").write_text(
        "[agent]\ntimeout_sec = 600\n[verifier]\ntimeout_sec = 120\n[environment]\nallow_internet = true\n",
        encoding="utf-8")
    (toml_task / "instruction.md").write_text("Sort the numbers in /app/in.txt.\n", encoding="utf-8")
    (toml_task / "tests" / "test.sh").write_text("echo 1 > /logs/verifier/reward.txt\n", encoding="utf-8")
    return root


def test_loader_reads_both_task_formats(tmp_path: Path):
    tasks = terminalbench.load(_tb_checkout(tmp_path), version="2.0")
    by_id = {t["id"]: t for t in tasks}
    hello, sort = by_id["hello-world"], by_id["sort-numbers"]
    assert hello["instruction"] == "Create /app/hello.txt\ncontaining Hello."
    assert hello["timeout"] == 300.0 and hello["test_command"] == terminalbench.TB1_TEST
    assert "Hello, world from TB!" in hello["expected_outputs"]
    assert sort["network"] is True and sort["test_timeout"] == 120.0 and "/tests/test.sh" in sort["test_command"]
    assert all(t["dataset_version"] == "2.0" and t["workdir"] == "/app" for t in tasks)
    assert set(tasks[0]) >= {"instruction", "image", "test_command", "workdir", "timeout", "id"}


def test_fetch_writes_one_file_for_search_and_test(tmp_path: Path):
    splits = terminalbench.fetch(tmp_path / "cache", tb_dir=_tb_checkout(tmp_path / "tb"), version="2.0-rc")
    assert splits["search"] == splits["test"]
    rows = [json.loads(l) for l in splits["search"].read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and rows[0]["dataset_version"] == "2.0-rc"
    assert terminalbench.run_meta("2.0-rc") == {"dataset": "terminalbench", "dataset_version": "2.0-rc"}


def test_leakage_audit_flags_ids_test_files_and_expected_outputs(tmp_path: Path):
    tasks = terminalbench.load(_tb_checkout(tmp_path))
    assert terminalbench.audit_leakage("def run(task):\n    return 'generic'\n", tasks) == []
    findings = terminalbench.audit_leakage("if 'hello-world' in task['id']: pass", tasks)
    assert findings == [{"kind": "task_id", "value": "hello-world", "task": "hello-world"}]
    kinds = {f["kind"] for f in terminalbench.audit_leakage(
        "open('test_outputs.py'); x = 'Hello, world from TB!'", tasks)}
    assert kinds == {"test_file", "expected_output"}


def test_seeds_pass_the_leakage_audit(tmp_path: Path):
    tasks = terminalbench.load(_tb_checkout(tmp_path))
    for name in ("terminus2.py", "terminus_kira.py"):
        assert terminalbench.audit_leakage((domain.SEED_DIR / name).read_text(encoding="utf-8"), tasks) == []


# --------------------------------------------------------------------------- real docker
@pytest.mark.skipif(not Container.available(), reason="docker not installed or daemon not running")
def test_real_container_scores_inside_itself(monkeypatch):
    monkeypatch.setattr(terminal_mod, "Container", Container)
    box = Container("alpine", workdir="/app")
    try:
        cid = box.start()
        code, _ = box.exec("echo ok > /app/x", 60)
        assert code == 0
        task = {"container": cid, "workdir": "/app", "test_command": "test -f /app/x"}
        assert terminal_metric(task, {"output": ""}) == 1.0
        assert terminal_metric({**task, "test_command": "test -f /app/missing"}, {"output": ""}) == 0.0
    finally:
        box.remove()


# --------------------------------------------------------------------------- review fixes
def test_agent_planted_test_state_is_wiped_before_scoring():
    FakeContainer.solved = False
    score, _ = domain.evaluate_task(PlantingHarness(), TASK, None, None, container_factory=FakeContainer)
    assert score == 0.0
    kinds = [e[0] for e in FakeContainer.log]
    assert kinds.index("exec") < kinds.index("put") < kinds.index("test")


def test_leakage_audit_rejects_a_leaked_candidate(tmp_path: Path):
    tasks = terminalbench.load(_tb_checkout(tmp_path / "tb"))
    leaked = tmp_path / "leaked.py"
    leaked.write_text("SPECIAL = {'hello-world': 'echo Hello'}\n", encoding="utf-8")
    with pytest.raises(domain.LeakageError, match="leakage audit.*hello-world"):
        domain.validate_candidate(leaked, tasks)
    with pytest.raises(domain.LeakageError):
        domain.HARNESS_WRAPPER(SeenHarness(), FakeContainer, source=leaked, tasks=tasks)
    domain.validate_candidate(domain.SEED_DIR / "terminus2.py", tasks)


class Runner:
    def __init__(self, has_bash=True):
        self.calls, self.has_bash = [], has_bash

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        assert kwargs.get("encoding") == "utf-8" and kwargs.get("timeout")
        out = "/bin/bash\n" if ("command -v bash" in argv and self.has_bash) else ""
        code = 1 if ("command -v bash" in argv and not self.has_bash) else 0
        return subprocess.CompletedProcess(argv, code, out, "")


@pytest.mark.parametrize("has_bash,shell", [(True, "bash"), (False, "sh")])
def test_scoring_goes_through_the_sandbox_with_a_kill_deadline(has_bash, shell):
    runner = Runner(has_bash)
    box = Container.attach("cid-9", "/app", runner=runner)
    box.exec_test("pytest", 30)
    box.exec_test("pytest -x", 30)
    probes = [c for c in runner.calls if "command -v bash" in c]
    assert len(probes) == 1
    test = runner.calls[-1]
    assert test[:5] == ["docker", "exec", "-w", "/app", "cid-9"]
    assert test[5:8] == ["timeout", "-s", "KILL"] and test[-3:] == [f"/bin/{shell}", "-c", "pytest -x"]


@pytest.mark.parametrize("network", [False, True])
def test_task_container_argv(network):
    from meta_harness.terminal import TaskSandbox
    args = TaskSandbox("img", network=network).run_args("n1")
    assert "--cap-drop" not in args and "no-new-privileges" in args
    assert "--pids-limit" in args and "--memory" in args and "--cpus" in args
    assert ("--network" in args) is (not network)


def test_toml_task_network_defaults_to_allowed(tmp_path: Path):
    task = tmp_path / "t1"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text("[agent]\ntimeout_sec = 60\n", encoding="utf-8")
    (task / "instruction.md").write_text("do it\n", encoding="utf-8")
    assert terminalbench.read_task(task)["network"] is True
    (task / "task.toml").write_text("[environment]\nallow_internet = false\n", encoding="utf-8")
    assert terminalbench.read_task(task)["network"] is False


# --------------------------------------------------------------------------- tamper checks
class Events:
    def __init__(self):
        self.events = []

    def event(self, name, payload=None, **fields):
        self.events.append((name, payload))


class BackgroundRewriter:
    """Starts a loop that keeps writing a passing reward file (until the container restarts)."""

    def run(self, task, model, trace):
        FakeContainer.rewriter = True
        return {"output": "done"}


def test_background_rewriter_is_killed_before_the_wipe():
    FakeContainer.solved = False
    score, _ = domain.evaluate_task(BackgroundRewriter(), TASK, None, None, container_factory=FakeContainer)
    assert score == 0.0
    kinds = [e[0] for e in FakeContainer.log]
    wipe = FakeContainer.log.index(("exec", "cid-1", domain.WIPE_COMMAND))
    assert kinds.index("restart") < wipe


@pytest.mark.parametrize("changes", [[("C", "/bin/bash")], [("C", "/usr/bin")], [("A", "/usr/local/bin/bash")],
                                     [("A", "/tests/test.sh")], [("D", "/logs")]])
def test_replaced_scorer_binary_or_verifier_path_scores_zero_tampered(changes):
    FakeContainer.changes = changes
    trace = Events()
    score, result = domain.evaluate_task(SeenHarness(), TASK, None, trace, container_factory=FakeContainer)
    assert score == 0.0 and result["score_reason"] == "tampered"
    assert not any(e[0] == "test" for e in FakeContainer.log)
    payload = dict(trace.events)["terminal_score"]
    assert payload["reason"] == "tampered" and payload["tampered"]


def test_unrelated_changes_still_score():
    FakeContainer.changes = [("A", "/app/env"), ("C", "/app"), ("C", "/logs"), ("A", "/logs/agent.log")]
    score, result = domain.evaluate_task(SeenHarness(), TASK, None, None, container_factory=FakeContainer)
    assert score == 1.0 and "score_reason" not in result


def test_logs_symlink_scores_zero_tampered():
    FakeContainer.symlink = True
    trace = Events()
    score, result = domain.evaluate_task(SeenHarness(), TASK, None, trace, container_factory=FakeContainer)
    assert score == 0.0 and result["score_reason"] == "tampered"
    assert dict(trace.events)["terminal_score"]["reason"] == "tampered"


def test_container_restart_and_diff_use_the_docker_cli():
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        assert kwargs.get("encoding") == "utf-8" and kwargs.get("timeout")
        out = "C /bin\nC /bin/bash\nA /app/x\n" if argv[1] == "diff" else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    box = Container.attach("cid-7", "/app", runner=runner)
    box.restart()
    assert box.diff() == [("C", "/bin"), ("C", "/bin/bash"), ("A", "/app/x")]
    assert calls == [["docker", "restart", "-t", "0", "cid-7"], ["docker", "diff", "cid-7"]]
    assert domain.tampered_paths(box.diff()) == ["C /bin", "C /bin/bash"]
