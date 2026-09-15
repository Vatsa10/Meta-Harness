from pathlib import Path

from meta_harness.core import CandidateEvaluator, ParetoFrontier, SearchConfig, SearchRunner


def test_end_to_end_search(tmp_path: Path):
    source = tmp_path / "harness.py"
    source.write_text("class Harness:\n    def run(self, task, model, trace):\n        return model(task['input'])\n")
    tasks = [{"input": "yes", "label": "yes"}]
    runner = SearchRunner(SearchConfig(tmp_path / "store", iterations=0), CandidateEvaluator(lambda p: p), lambda *_: [])
    frontier = runner.run([source], tasks)
    assert frontier[0].score == 1.0
    assert (tmp_path / "store" / "candidates" / "initial-0000" / "traces.jsonl").exists()


def test_pareto_frontier():
    from meta_harness.core import EvaluationResult
    values = [EvaluationResult("a", .8, context_cost=10), EvaluationResult("b", .7, context_cost=2), EvaluationResult("c", .6, context_cost=10)]
    assert {x.candidate_id for x in ParetoFrontier.select(values)} == {"a", "b"}
