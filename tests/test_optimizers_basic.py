import json

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner
from meta_harness.domains import generic
from meta_harness.optimizers import OptimizerContext, all_optimizers, build
from meta_harness.providers import model_from_environment

SEED = "class Harness:\n    def run(self, task, model, trace):\n        return model(task['input'])\n"
NEW = "class Harness:\n    def run(self, task, model, trace):\n        return 'fruit'\n"
WRITE = {"name": "write_candidate", "arguments": {"source": NEW, "rationale": "r"}}
READ_ITER1 = {"name": "read", "arguments": {"path": "candidates/iteration-0001-00/scores.json"}}


def scripted(tmp_path, rows):
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return model_from_environment("scripted", str(path))


def search(tmp_path, optimizer, rows, iterations, view="full"):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED, encoding="utf-8")
    ctx = OptimizerContext(scripted(tmp_path, rows), generic, view=view, max_tool_calls=5)
    config = SearchConfig(root=tmp_path / "store", iterations=iterations)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: "vehicle"), build(optimizer, ctx))
    runner.run([seed], [{"input": "apple", "label": "fruit"}])
    return config.root


def tool_log(root, iteration):
    path = root / "proposals" / f"iteration-{iteration:04d}" / "candidate-00.proposer_log.jsonl"
    records = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return [e for e in records if e.get("role") == "tool"]


def test_both_optimizers_are_registered():
    found = all_optimizers()
    assert "metaharness" in found and "best-of-n" in found


def test_best_of_n_iteration_two_cannot_see_iteration_one(tmp_path):
    root = search(tmp_path, "best-of-n", [{"tool_calls": [WRITE]},
                                          {"tool_calls": [READ_ITER1]}, {"tool_calls": [WRITE]}], 2)
    assert (root / "candidates" / "iteration-0001-00" / "scores.json").is_file()
    assert tool_log(root, 2)[0]["ok"] is False


def test_metaharness_iteration_two_sees_iteration_one(tmp_path):
    root = search(tmp_path, "metaharness", [{"tool_calls": [WRITE]},
                                            {"tool_calls": [READ_ITER1]}, {"tool_calls": [WRITE]}], 2)
    assert tool_log(root, 2)[0]["ok"] is True


def test_scores_view_hides_traces(tmp_path):
    read = {"name": "read", "arguments": {"path": "candidates/initial-0000/traces.jsonl"}}
    root = search(tmp_path, "metaharness", [{"tool_calls": [read]}, {"tool_calls": [WRITE]}], 1,
                  view="scores")
    assert (root / "candidates" / "initial-0000" / "traces.jsonl").is_file()
    assert tool_log(root, 1)[0]["ok"] is False
