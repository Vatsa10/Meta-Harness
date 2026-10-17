"""Budget ledger: one labelled entry per evaluation and proposer call, checked against run.json."""
import json
from pathlib import Path

import pytest

from meta_harness.core import Budget, BudgetError, CandidateEvaluator, Ledger, SearchConfig, SearchRunner

HARNESS = '''
VERSION = {n}

class Harness:
    def run(self, task, model, trace):
        model("hello")
        return task["input"]
'''


class Proposer:
    """Writes one distinct candidate per call and reports cumulative usage like the real ones."""

    def __init__(self, scratch: Path):
        self.scratch = scratch
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.invocations = 0

    def __call__(self, experience, iteration, count):
        self.invocations += 1
        self.usage = {"calls": self.usage["calls"] + 2, "input_tokens": self.usage["input_tokens"] + 100,
                      "output_tokens": self.usage["output_tokens"] + 10}
        path = self.scratch / f"proposal-{iteration}.py"
        path.write_text(HARNESS.format(n=iteration), encoding="utf-8")
        return [path]


def _runner(tmp_path: Path, iterations: int = 2, cap: int | None = None, repeats: int = 1):
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    seed = scratch / "seed.py"
    seed.write_text(HARNESS.format(n=0), encoding="utf-8")
    config = SearchConfig(tmp_path / "store", iterations=iterations, max_evaluations=cap, repeats=repeats)
    proposer = Proposer(scratch)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), proposer)
    return runner, proposer, seed


TASKS = [{"input": "a", "label": "a"}, {"input": "b", "label": "a"}]


def _run_json(runner):
    return json.loads((runner.experience.root / "run.json").read_text(encoding="utf-8"))


def test_duplicate_label_raises():
    ledger = Ledger()
    ledger.evaluation("initial-0000", "search", 0, {"model_calls": 1})
    with pytest.raises(BudgetError, match="duplicate"):
        ledger.evaluation("initial-0000", "search", 0, {"model_calls": 1})
    ledger.proposer("iteration-0001", {"calls": 1})
    with pytest.raises(BudgetError, match="duplicate"):
        ledger.proposer("iteration-0001", {"calls": 1})
    with pytest.raises(BudgetError, match="duplicate"):
        Ledger([{"kind": "proposer", "label": "x"}, {"kind": "proposer", "label": "x"}])


def test_run_json_records_ledger_totals_matching_budget(tmp_path):
    runner, proposer, seed = _runner(tmp_path, iterations=2, repeats=2)
    runner.run([seed], TASKS)
    data = _run_json(runner)
    totals = data["ledger"]["totals"]
    assert totals["harness_evaluations"] == data["budget"]["harness_evaluations"] == 6
    assert totals["proposer_calls"] == data["budget"]["proposer_calls"] == 4
    assert totals["proposer_invocations"] == proposer.invocations == 2
    assert totals["model_calls"] == data["budget"]["model_calls"] == 12
    labels = [e["label"] for e in data["ledger"]["entries"]]
    assert "eval:initial-0000:search:r1" in labels and "proposer:iteration-0002" in labels
    assert "expected" not in data


def test_expected_block_and_cap_with_max_evaluations(tmp_path):
    runner, _, seed = _runner(tmp_path, iterations=5, cap=3)
    runner.run([seed], TASKS)
    data = _run_json(runner)
    assert data["expected"]["max_evaluations"] == 3
    assert data["budget"]["harness_evaluations"] == data["ledger"]["totals"]["harness_evaluations"] == 3
    over = Budget.from_dict(data["budget"])
    with pytest.raises(BudgetError, match="cap 2"):
        Ledger(data["ledger"]["entries"]).verify(over, cap=2)


def test_mismatch_between_run_json_and_ledger_is_detected(tmp_path):
    runner, _, seed = _runner(tmp_path, iterations=1)
    runner.run([seed], TASKS)
    runner.verify_ledger()   # consistent
    path = runner.experience.root / "run.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["budget"]["harness_evaluations"] += 1
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BudgetError, match="harness_evaluations: run.json 3 != ledger 2"):
        runner.verify_ledger()


def test_in_memory_drift_fails_the_run(tmp_path):
    runner, _, seed = _runner(tmp_path, iterations=1)
    original = runner.budget.add_run

    def double(metrics, split):   # a counting bug: the budget counts a run the ledger never saw
        original(metrics, split)
        original(metrics, split)

    runner.budget.add_run = double
    with pytest.raises(BudgetError, match="mismatch"):
        runner.run([seed], TASKS)


def test_resume_does_not_double_count(tmp_path):
    runner, _, seed = _runner(tmp_path, iterations=2)
    runner.run([seed], TASKS, test_tasks=[{"input": "c", "label": "c"}])
    before = _run_json(runner)

    again, proposer, _ = _runner(tmp_path, iterations=2)
    again.run([seed], TASKS, test_tasks=[{"input": "c", "label": "c"}], resume=True)
    after = _run_json(again)
    assert proposer.invocations == 0
    assert after["budget"]["harness_evaluations"] == before["budget"]["harness_evaluations"] == 3
    assert after["ledger"]["totals"] == before["ledger"]["totals"]
    assert after["budget"]["test_evaluations"] == before["budget"]["test_evaluations"] >= 1
    # held-out entries stay outside the experience root
    assert all(e["split"] != "test" for e in after["ledger"]["entries"] if e["kind"] == "evaluation")
    held = json.loads((Path(again.config.test_root) / "ledger.json").read_text(encoding="utf-8"))
    assert len(held) == after["budget"]["test_evaluations"]
    assert all(e["label"] == f"eval:{e['candidate']}:test1:r0" for e in held)


def test_resume_of_a_torn_candidate_replaces_its_entries(tmp_path):
    runner, _, seed = _runner(tmp_path, iterations=2)
    runner.run([seed], TASKS)
    torn = runner.experience.candidate_dir("initial-0000")
    receipt = json.loads((torn / "receipt.json").read_text(encoding="utf-8"))
    receipt["verdict"] = "incomplete"   # killed after counting the evaluation, before finishing it
    (torn / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")

    again, _, _ = _runner(tmp_path, iterations=2)
    again.run([seed], TASKS, resume=True)
    data = _run_json(again)
    assert data["budget"]["harness_evaluations"] == data["ledger"]["totals"]["harness_evaluations"] == 3
    assert json.loads((torn / "receipt.json").read_text(encoding="utf-8"))["verdict"] == "accepted"
