"""Wave F integration: cross-module wiring between search, sweep, domains, core and providers."""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness.chat_types import AssistantTurn, ToolCall, Usage
from meta_harness.core import CandidateEvaluator, MeteredModel, SearchConfig, SearchRunner, TraceRecorder

REPO = Path(__file__).resolve().parents[1]
SMOKE = Path(__file__).parent / "fixtures" / "smoke"
sys.path.insert(0, str(Path(__file__).parent))


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


# -- item 1: one shared adapter / metric map for search and sweep ----------------------------------

def test_sweep_and_search_share_the_domain_maps():
    from meta_harness.commands import search, sweep
    from meta_harness.domains import adapter_for, metric_for
    from meta_harness.protocol import exact, numeric

    assert sweep.adapter_for is adapter_for and sweep.metric_for is metric_for
    assert search.metric_for is metric_for
    assert metric_for("classification") is exact and metric_for("math") is numeric
    assert adapter_for("terminal")([{"instruction": "x"}])[0]["workdir"] == "/app"
    with pytest.raises(KeyError):
        adapter_for("nope")


def test_sweep_runs_offline(tmp_path, capsys, monkeypatch):
    harness = tmp_path / "h.py"
    harness.write_text("import os\nclass Harness:\n    def run(self, task, model, trace):\n"
                       "        return {'output': os.environ.get('MODE')}\n", encoding="utf-8")
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps({"input": f"q{i}", "label": "a"}) + "\n" for i in range(4)),
                    encoding="utf-8")
    from meta_harness.commands import sweep
    monkeypatch.setattr(sweep, "model_from_environment", lambda *a, **k: (lambda prompt, **kw: "x"))
    code = cli.main(["sweep", "--harness", str(harness), "--env", "MODE=a,b", "--tasks", str(data),
                     "--model", "m", "--root", str(tmp_path / "sw"), "--no-cache"])
    assert code == 0
    rows = json.loads((tmp_path / "sw" / "sweep.json").read_text(encoding="utf-8"))["rows"]
    assert [r["score"] for r in rows] == [1.0, 0.0]


# -- items 3, 6, 13: terminal domain wired into search ---------------------------------------------

class FakeBox:
    log: list = []

    def __init__(self, image, workdir="/app", network=False):
        self.image, self.workdir, self.id = image, workdir, None

    @classmethod
    def attach(cls, container_id, workdir="/app"):
        box = cls("", workdir)
        box.id = container_id
        return box

    def start(self):
        self.id = f"cid-{len(FakeBox.log)}"
        FakeBox.log.append(("start", self.image))
        return self.id

    def exec(self, command, timeout):
        FakeBox.log.append(("exec", command))
        return 0, ""

    def exec_test(self, command, timeout):
        FakeBox.log.append(("test", command))
        return 0, "passed"

    def put(self, local, remote):
        FakeBox.log.append(("put", remote))

    def remove(self):
        FakeBox.log.append(("remove", self.id))


TERMINAL_SEED = '''class Harness:
    def run(self, task, model, trace):
        wrote = "write" in task["instruction"]
        return {"output": "done", "stop_reason": "done", "ended_without_writing_code": not wrote,
                "ended_in_understand": not wrote}
'''


@pytest.fixture
def terminal_search(tmp_path, monkeypatch):
    import meta_harness.terminal as terminal_mod
    from meta_harness.domains import terminal

    FakeBox.log = []
    monkeypatch.setattr(terminal_mod, "Container", FakeBox)
    monkeypatch.setattr(terminal, "HARNESS_WRAPPER",
                        lambda harness: terminal.InContainer(harness, container_factory=FakeBox))
    tasks = tmp_path / "tb.jsonl"
    tasks.write_text("".join(json.dumps(r) + "\n" for r in [
        {"id": "make-hello", "instruction": "write hello", "image": "img", "test_command": "bash /tests/t.sh"},
        {"id": "read-config", "instruction": "inspect it", "image": "img", "test_command": "bash /tests/t.sh"},
    ]), encoding="utf-8")

    def run(seed_text, *extra):
        seed = tmp_path / "seed.py"
        seed.write_text(seed_text, encoding="utf-8")
        return cli.main(["search", "--domain", "terminal", "--harness", str(seed), "--search", str(tasks),
                         "--test", str(tasks), "--provider", "scripted", "--model", str(SMOKE / "base.jsonl"),
                         "--iterations", "0", "--root", str(tmp_path / "s"), "--no-cache", *extra])
    return run, tmp_path / "s"


def test_terminal_search_scores_in_the_container_and_records_the_dataset(terminal_search, capsys):
    run, root = terminal_search
    assert run(TERMINAL_SEED, "--dataset-version", "2.0") == 0
    scores = json.loads((root / "candidates" / "initial-0000" / "scores.json").read_text(encoding="utf-8"))
    assert scores["score"] == 1.0, "METRICS['terminal'] reads the in-container verdict"
    assert [e[0] for e in FakeBox.log].count("start") >= 2
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert manifest["run_meta"]["dataset"] == "terminalbench"
    assert manifest["run_meta"]["dataset_version"] == "2.0"
    # item 6: phase labels aggregated into scores.json
    assert scores["metrics"]["ended_without_writing_code_rate"] == pytest.approx(0.5)
    assert scores["metrics"]["ended_in_understand_rate"] == pytest.approx(0.5)


def test_terminal_seeds_validate_without_running_on_the_host(terminal_search, capsys):
    run, root = terminal_search
    seed = (REPO / "baselines" / "terminal" / "terminus2.py").read_text(encoding="utf-8")
    assert run(seed, "--task-timeout", "5") == 0
    receipt = json.loads((root / "candidates" / "initial-0000" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["validation"] == {"ok": True}


def test_terminal_search_rejects_a_leaked_candidate(terminal_search, capsys):
    run, root = terminal_search
    leaked = TERMINAL_SEED + "\nSPECIAL = {'make-hello': 'echo hello'}\n"
    assert run(leaked) == 0
    receipt = json.loads((root / "candidates" / "initial-0000" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["verdict"] == "rejected_invalid_source"
    assert "leakage audit" in receipt["fallback"] and "make-hello" in receipt["fallback"]
    assert not any(e[0] == "start" for e in FakeBox.log), "a leaked candidate never runs"


def test_phase_rates_absent_when_no_task_reports_phases(tmp_path):
    harness = tmp_path / "h.py"
    harness.write_text("class Harness:\n    def run(self, task, model, trace):\n        return {'output': 'a'}\n",
                       encoding="utf-8")
    result = CandidateEvaluator(lambda p, **k: "x").evaluate(harness, [{"input": "q", "label": "a"}], "c",
                                                             tmp_path / "t.jsonl")
    assert "ended_without_writing_code_rate" not in result.metrics


def test_experience_reads_shows_phase_rates(tmp_path):
    from meta_harness.experience_query import read_stats

    directory = tmp_path / "candidates" / "c1"
    directory.mkdir(parents=True)
    (directory / "scores.json").write_text(json.dumps({
        "candidate_id": "c1", "score": 1.0,
        "metrics": {"ended_without_writing_code_rate": 0.25, "ended_in_understand_rate": 0.5}}),
        encoding="utf-8")
    stats = read_stats(tmp_path)
    assert stats["phase_rates"] == [{"candidate_id": "c1", "ended_without_writing_code_rate": 0.25,
                                     "ended_in_understand_rate": 0.5}]


# -- item 4: loader-provided train split for classification ----------------------------------------

def test_classification_search_uses_the_loader_train_split(tmp_path, capsys, monkeypatch):
    from test_loaders_classification import law_http

    from meta_harness.commands import search as search_cmd
    from meta_harness.loaders import lawbench

    splits = lawbench.fetch(tmp_path / "law", law_http())
    captured = {}
    real = search_cmd.CandidateEvaluator

    def spy(*args, **kwargs):
        captured["train"] = kwargs.get("train_tasks")
        return real(*args, **kwargs)

    monkeypatch.setattr(search_cmd, "CandidateEvaluator", spy)
    code = cli.main(["search", "--domain", "classification",
                     "--harness", str(REPO / "baselines" / "classification" / "zero_shot.py"),
                     "--search", str(splits["val"]), "--test", str(splits["test"]),
                     "--provider", "scripted", "--model", str(SMOKE / "base.jsonl"),
                     "--iterations", "0", "--root", str(tmp_path / "s"), "--no-cache"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    train_rows = _rows(splits["train"])
    assert out["train_tasks"] == len(train_rows) > 0
    train = {t["input"] for t in captured["train"]}
    assert train == {r["input"] for r in train_rows}
    assert not train & {r["input"] for r in _rows(splits["val"])}
    assert not train & {r["input"] for r in _rows(splits["test"])}


def test_classification_train_tasks_without_a_loader_dir_is_empty(tmp_path):
    from meta_harness.domains import classification

    lone = tmp_path / "data.jsonl"
    lone.write_text(json.dumps({"input": "a", "label": "x"}) + "\n", encoding="utf-8")
    assert classification.train_tasks(lone) == [] and classification.train_tasks(None) == []


# -- item 5: native tool calls are metered --------------------------------------------------------

class CachingProvider:
    def chat(self, system, messages, tools, max_tokens=4096):
        return AssistantTurn("ok", [ToolCall("t1", "bash", {"command": "ls"})],
                             Usage(input=100, output=7, cache_read=40, cache_write=5, reasoning=11), "tool_use")


def test_metered_chat_meters_usage_and_traces(tmp_path):
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        metered = MeteredModel(CachingProvider(), trace)
        turn = metered.chat("sys", [{"role": "user", "content": "go"}], [])
    assert turn.tool_calls[0].name == "bash"
    assert metered.calls == 1 and metered.prompt_tokens == 100 and metered.completion_tokens == 7
    assert metered.usage == {"input": 100, "output": 7, "cache_read": 40, "cache_write": 5, "reasoning": 11}
    events = _rows(tmp_path / "t.jsonl")
    call = next(e for e in events if e["event"] == "model_call")
    assert call["payload"]["tool_calls"][0]["name"] == "bash"
    assert call["payload"]["usage"]["reasoning"] == 11


def test_metered_chat_with_the_scripted_provider_through_the_cache(tmp_path):
    from meta_harness.cache import CachedModel
    from meta_harness.scripted import ScriptedModel
    from meta_harness.terminal_harness import KiraHarness

    script = tmp_path / "s.jsonl"
    script.write_text(json.dumps({"text": "", "tool_calls": [{"name": "bash", "arguments": {"command": "ls"}}]})
                      + "\n", encoding="utf-8")
    model = CachedModel(ScriptedModel(script), tmp_path / "cache", enabled=False)
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        metered = MeteredModel(model, trace)
        assert KiraHarness._uses_chat(metered)
        turn = metered.chat("sys", [{"role": "user", "content": "go"}], [])
    assert turn.tool_calls[0].arguments == {"command": "ls"}
    assert metered.calls == 1 and metered.usage["input"] > 0


def test_metered_model_without_chat_has_no_chat(tmp_path):
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        metered = MeteredModel(lambda prompt, **kw: "x", trace)
        assert getattr(metered, "chat", None) is None


# -- item 7: redact_obj redacts dict keys ----------------------------------------------------------

def test_redact_obj_redacts_keys():
    from meta_harness.redact import redact_obj

    secret = "sk-ant-" + "a" * 40
    out = redact_obj({secret: {secret: "v"}})
    assert secret not in json.dumps(out)


# -- item 8: ProviderError.status ------------------------------------------------------------------

def test_provider_error_carries_the_http_status():
    import io
    import urllib.error

    from meta_harness.judge import http_status
    from meta_harness.providers import ProviderError, RetryPolicy, _HttpClient

    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(b"denied"))

    client = _HttpClient(urlopen=fake_urlopen, retry=RetryPolicy(attempts=1), sleep=lambda s: None)
    with pytest.raises(ProviderError) as info:
        client.post("https://example.invalid/x", {}, {})
    assert info.value.status == 401

    def rate_limited(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 429, "slow", {}, io.BytesIO(b"later"))

    client = _HttpClient(urlopen=rate_limited, retry=RetryPolicy(attempts=1), sleep=lambda s: None)
    with pytest.raises(ProviderError) as info:
        client.post("https://example.invalid/x", {}, {})
    assert info.value.status == 429
    assert http_status(ProviderError("no status in this text", status=503)) == 503


# -- item 9: --price reaches Budget.price as an argument -------------------------------------------

def test_price_override_does_not_touch_the_environment(tmp_path, capsys, monkeypatch):
    from meta_harness.commands import search as search_cmd

    monkeypatch.delenv("META_HARNESS_PRICES", raising=False)
    seen = []
    real_run = search_cmd.SearchRunner.run

    def spy(self, *args, **kwargs):
        seen.append(os.environ.get("META_HARNESS_PRICES"))
        return real_run(self, *args, **kwargs)

    monkeypatch.setattr(search_cmd.SearchRunner, "run", spy)
    base = str(SMOKE / "base.jsonl")
    code = cli.main(["search", "--harness", str(SMOKE / "seed.py"), "--data", str(SMOKE / "data.jsonl"),
                     "--metric", "exact", "--provider", "scripted", "--model", base,
                     "--iterations", "0", "--root", str(tmp_path / "s"), "--no-cache",
                     "--price", f"scripted:{base}=1000000,0"])
    assert code == 0 and seen == [None]
    run = json.loads((tmp_path / "s" / "run.json").read_text(encoding="utf-8"))
    assert run["budget"]["cost_usd"] == pytest.approx(run["budget"]["model_usage"]["input"])


def test_budget_price_takes_overrides():
    from meta_harness.core import Budget
    from meta_harness.providers import Price

    budget = Budget(model_calls=1, model_usage={"input": 1_000_000, "output": 0, "cache_read": 0,
                                                "cache_write": 0, "reasoning": 0})
    assert budget.price("unpriced-model", None) is None
    assert budget.price("unpriced-model", None, {"unpriced-model": Price(2.0, 0.0)}) == pytest.approx(2.0)


# -- item 10: ood --train-from end to end ----------------------------------------------------------

LEARNER = '''class Harness:
    def __init__(self):
        self.learned = 0

    def run(self, task, model, trace):
        return {"output": "Sci/Tech" if self.learned > 0 else "nothing"}

    def learn(self, task, result, model=None):
        self.learned += 1
'''


def test_ood_train_from_streams_the_loader_train_split(tmp_path, capsys):
    from test_loaders_classification import law_http
    from test_ood import FakeHttp, fixture

    from meta_harness.loaders import lawbench
    from meta_harness.loaders.ood import fetch_one

    cache = tmp_path / "cache"
    train_rows = _rows(lawbench.fetch(cache / "lawbench", law_http())["train"])
    fetch_one("agnews", cache / "ood", FakeHttp(), limit=20)
    harness = tmp_path / "learner.py"
    harness.write_text(LEARNER, encoding="utf-8")
    script = tmp_path / "script.jsonl"
    script.write_text(json.dumps({"default": "x"}) + "\n", encoding="utf-8")

    def accuracy(*extra):
        out = tmp_path / ("out" + "".join(extra))
        code = cli.main(["ood", "--harness", str(harness), "--datasets", "agnews", "--provider", "scripted",
                         "--model", str(script), "--cache-dir", str(cache), "--out", str(out), *extra])
        assert code == 0
        row = json.loads((out / "ood.json").read_text(encoding="utf-8"))["rows"][0]
        return row["accuracy"], row["train_n"]

    rows = fixture("agnews")["rows"]
    sci = sum(r["row"]["label"] == 3 for r in rows) / len(rows)
    assert accuracy() == (0.0, 0), "zero-shot: nothing was learned"
    assert accuracy("--train-from", "lawbench", "--train-limit", "3") == (pytest.approx(sci),
                                                                             min(3, len(train_rows)))
    assert sci > 0 and train_rows


# -- item 11: small fixes -------------------------------------------------------------------------

def test_torn_scores_json_is_not_finished(tmp_path):
    runner = SearchRunner(SearchConfig(root=tmp_path / "r"), CandidateEvaluator(lambda p, **k: "x"),
                          lambda *a: [])
    runner._resuming = True
    directory = runner.experience.candidate_dir("c1")
    directory.mkdir(parents=True)
    (directory / "receipt.json").write_text(json.dumps({"verdict": "accepted"}), encoding="utf-8")
    (directory / "scores.json").write_text("{torn", encoding="utf-8")
    assert runner._is_finished("c1") is False
    (directory / "scores.json").write_text(json.dumps({"candidate_id": "c1", "score": 1.0}), encoding="utf-8")
    assert runner._is_finished("c1") is True


def test_gitignore_keeps_plugin_types_tracked():
    assert ".claude-plugin/types/" not in (REPO / ".gitignore").read_text(encoding="utf-8")


def test_no_network_guard_blocks_connect_and_connect_ex():
    for method in ("connect", "connect_ex"):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(AssertionError, match="network"):
                getattr(sock, method)(("127.0.0.1", 9))
        finally:
            sock.close()
