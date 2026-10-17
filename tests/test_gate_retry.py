"""Gate retry (an invalid proposal is asked for once more, then the parent is kept) and no-op
rejection (a candidate equal to an existing one after normalisation is never evaluated)."""
import json
from pathlib import Path

from meta_harness.core import VERDICTS, CandidateEvaluator, SearchConfig, SearchRunner, normalised_source

SEED = '''class Harness:
    def run(self, task, model, trace):
        return model(task["input"])
'''
REFORMATTED = '''# a comment the proposer added
class Harness:

    def run(self, task, model, trace):   # same body, new spacing
        return model( task["input"] )
'''
DIFFERENT = '''class Harness:
    def run(self, task, model, trace):
        return model(task["input"]).strip()
'''
BROKEN = "class Harness:\n    def run(self, task, model, trace)\n        return 1\n"


class Model:
    def __init__(self):
        self.n = 0

    def __call__(self, prompt, **_):
        self.n += 1
        return "x"


class Proposer:
    def __init__(self, scratch: Path, sources, revised=None, hook=True):
        self.scratch, self.sources, self.revised = scratch, list(sources), revised
        self.revisions: list[tuple[int, str, str]] = []
        self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        if not hook:
            self.revise = None

    def __call__(self, experience, iteration, count):
        self.usage = {**self.usage, "calls": self.usage["calls"] + 1}
        paths = []
        for index, text in enumerate(self.sources[:count]):
            path = self.scratch / f"p{iteration}-{index}.py"
            path.write_text(text, encoding="utf-8")
            paths.append(path)
        return paths

    def revise(self, experience, iteration, source, error):
        self.usage = {**self.usage, "calls": self.usage["calls"] + 1}
        self.revisions.append((iteration, Path(source).name, error))
        if self.revised is None:
            return []
        path = self.scratch / f"revised-{iteration}.py"
        path.write_text(self.revised, encoding="utf-8")
        return [path]


TASKS = [{"input": "a", "label": "x"}]


def _run(tmp_path, proposer_args, iterations=1, per_iteration=1, resume=False):
    scratch = tmp_path / "scratch"
    scratch.mkdir(exist_ok=True)
    seed = scratch / "seed.py"
    seed.write_text(SEED, encoding="utf-8")
    proposer = Proposer(scratch, **proposer_args)
    model = Model()
    config = SearchConfig(tmp_path / "store", iterations=iterations, candidates_per_iteration=per_iteration)
    runner = SearchRunner(config, CandidateEvaluator(model), proposer)
    frontier = runner.run([seed], TASKS, resume=resume)
    return runner, proposer, model, frontier


def _receipt(runner, cid):
    return json.loads((runner.experience.candidate_dir(cid) / "receipt.json").read_text(encoding="utf-8"))


def _budget(runner):
    return json.loads((runner.experience.root / "run.json").read_text(encoding="utf-8"))["budget"]


def test_normalised_source_ignores_comments_and_formatting():
    assert "rejected_noop" in VERDICTS
    assert normalised_source(SEED) == normalised_source(REFORMATTED)
    assert normalised_source(SEED) != normalised_source(DIFFERENT)
    assert normalised_source(BROKEN) is None


def test_candidate_equal_to_its_parent_is_rejected_noop_and_not_evaluated(tmp_path):
    runner, _, model, frontier = _run(tmp_path, {"sources": [REFORMATTED]})
    receipt = _receipt(runner, "iteration-0001-00")
    assert receipt["verdict"] == "rejected_noop"
    assert "initial-0000" in receipt["fallback"] and receipt["evaluation"] is None
    assert model.n == 1, "only the seed was evaluated"
    assert _budget(runner)["harness_evaluations"] == 1
    assert _budget(runner)["invalid_proposals"] == 0
    assert [r.candidate_id for r in frontier] == ["initial-0000"]


def test_candidate_equal_to_an_earlier_proposal_is_rejected_noop(tmp_path):
    runner, _, model, _ = _run(tmp_path, {"sources": [DIFFERENT, DIFFERENT + "\n# again\n"]}, per_iteration=2)
    assert _receipt(runner, "iteration-0001-00")["verdict"] == "accepted"
    second = _receipt(runner, "iteration-0001-01")
    assert second["verdict"] == "rejected_noop" and "iteration-0001-00" in second["fallback"]
    assert model.n == 2


def test_changed_candidate_is_evaluated(tmp_path):
    runner, _, model, _ = _run(tmp_path, {"sources": [DIFFERENT]})
    assert _receipt(runner, "iteration-0001-00")["verdict"] == "accepted"
    assert model.n == 2


def test_invalid_proposal_is_retried_once_with_the_error(tmp_path):
    runner, proposer, _, frontier = _run(tmp_path, {"sources": [BROKEN], "revised": DIFFERENT})
    assert len(proposer.revisions) == 1
    iteration, name, error = proposer.revisions[0]
    assert iteration == 1 and name == "p1-0.py" and "does not compile" in error
    original = _receipt(runner, "iteration-0001-00")
    assert original["verdict"] == "rejected_invalid_source" and original["retry_requested"] is True
    assert original["retry"] == {"candidate_id": "iteration-0001-00-retry", "verdict": "accepted"}
    assert _receipt(runner, "iteration-0001-00-retry")["verdict"] == "accepted"
    labels = [e["label"] for e in json.loads((runner.experience.root / "run.json").read_text(
        encoding="utf-8"))["ledger"]["entries"]]
    assert "proposer:iteration-0001-00:retry" in labels
    assert _budget(runner)["proposer_calls"] == 2


def test_failed_retry_keeps_the_parent(tmp_path):
    runner, proposer, _, frontier = _run(tmp_path, {"sources": [BROKEN], "revised": BROKEN + "\n"})
    assert len(proposer.revisions) == 1, "asked once more, not again for the retry"
    original = _receipt(runner, "iteration-0001-00")
    assert original["retry"]["verdict"] == "rejected_invalid_source"
    assert "parent kept" in original["fallback"]
    assert [r.candidate_id for r in frontier] == ["initial-0000"]


def test_retry_that_returns_the_parent_is_rejected_noop(tmp_path):
    runner, _, model, _ = _run(tmp_path, {"sources": [BROKEN], "revised": SEED})
    assert _receipt(runner, "iteration-0001-00-retry")["verdict"] == "rejected_noop"
    assert model.n == 1


def test_without_a_hook_the_error_is_left_for_the_next_iteration(tmp_path):
    runner, _, _, _ = _run(tmp_path, {"sources": [BROKEN], "hook": False})
    receipt = _receipt(runner, "iteration-0001-00")
    assert receipt["retry_requested"] is True and "retry" not in receipt
    assert "parent kept" in receipt["fallback"]
    note = runner.experience.root / receipt["validation_error_file"]
    assert note.is_file() and "does not compile" in note.read_text(encoding="utf-8")


def test_resume_does_not_retry_again(tmp_path):
    _run(tmp_path, {"sources": [BROKEN], "revised": DIFFERENT})
    runner, proposer, model, _ = _run(tmp_path, {"sources": [BROKEN], "revised": DIFFERENT}, resume=True)
    assert proposer.revisions == [] and model.n == 0
    assert _receipt(runner, "iteration-0001-00-retry")["verdict"] == "accepted"
