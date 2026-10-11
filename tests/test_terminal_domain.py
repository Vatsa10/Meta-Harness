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

    def exec_test(self, command, timeout):
        FakeContainer.log.append(("test", self.id, self.workdir, command))
        return 0, "passed"

    def put(self, local, remote):
        FakeContainer.log.append(("put", str(local), remote))

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


@pytest.fixture(autouse=True)
def fake_container(monkeypatch):
    FakeContainer.log = []
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
