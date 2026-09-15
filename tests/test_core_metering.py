import json
from pathlib import Path

from meta_harness.core import CandidateEvaluator, MeteredModel, TraceRecorder

HARNESS = '''
class Harness:
    def run(self, task, model, trace):
        return model("classify: " + task["input"])
'''


def test_metered_model_records_provider_usage(tmp_path: Path):
    class Upstream:
        model = "m"

        def call(self, prompt, **kwargs):
            return "ok", {"prompt_tokens": 123, "completion_tokens": 4}

    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        metered = MeteredModel(Upstream(), trace)
        assert metered("hello") == "ok"
        assert trace.context_cost == 123
    events = [json.loads(line) for line in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "model_call"
    assert events[0]["payload"]["prompt"] == "hello"
    assert events[0]["payload"]["response"] == "ok"


def test_metered_model_estimates_when_usage_absent(tmp_path: Path):
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        metered = MeteredModel(lambda prompt, **_: "ok", trace)
        metered("a" * 400)
        assert trace.context_cost == 100


def test_evaluator_context_cost_is_tokens_not_events(tmp_path: Path):
    source = tmp_path / "harness.py"
    source.write_text(HARNESS)
    evaluator = CandidateEvaluator(lambda prompt, **_: "fruit")
    result = evaluator.evaluate(source, [{"input": "apple", "label": "fruit"}], "c0", tmp_path / "trace.jsonl")
    assert result.score == 1.0
    # "classify: apple" is 15 chars -> ceil(15/4) == 4 tokens.
    assert result.context_cost == 4.0
    assert result.metrics["model_calls"] == 1.0


def test_evaluator_enforces_task_timeout(tmp_path: Path):
    source = tmp_path / "slow.py"
    source.write_text('''
import time


class Harness:
    def run(self, task, model, trace):
        time.sleep(1.0)
        return "late"
''')
    evaluator = CandidateEvaluator(lambda prompt, **_: "x", task_timeout=0.2)
    result = evaluator.evaluate(source, [{"input": "a", "label": "a"}], "c1", tmp_path / "trace.jsonl")
    assert result.valid is False
    assert "timeout" in (result.error or "").lower()
