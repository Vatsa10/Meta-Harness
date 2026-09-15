from pathlib import Path

from meta_harness.core import CandidateEvaluator, ParetoFrontier, SearchConfig, SearchRunner
from meta_harness.harnesses import LabelPrimedQueryHarness, MathRetrievalHarness
from meta_harness.retrieval import BM25Index, TfidfIndex
from meta_harness.core import TraceRecorder


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


def test_retrieval_and_classification_harness(tmp_path: Path):
    examples = [{"input": "apple pie", "label": "fruit"}, {"input": "fast car", "label": "vehicle"}]
    harness = LabelPrimedQueryHarness(["fruit", "vehicle"])
    with TraceRecorder(tmp_path / "trace.jsonl") as trace:
        for task in examples:
            prediction = harness.run(task, lambda prompt: "vehicle" if "Input: fast car" in prompt else "fruit", trace)
            assert prediction == task["label"]
    assert TfidfIndex(examples, [x["input"] for x in examples]).search("apple", 1)[0].item["label"] == "fruit"
    assert BM25Index(examples, [x["input"] for x in examples], math_mode=False).search("car", 1)[0].item["label"] == "vehicle"


def test_math_router_and_retrieval(tmp_path: Path):
    corpus = [
        {"problem": "Prove a triangle has an angle", "solution": "Use angles", "difficulty": 7},
        {"problem": "Show every prime is odd", "solution": "By contradiction", "difficulty": 5},
    ]
    harness = MathRetrievalHarness(corpus)
    assert harness.route("Find the angle of a triangle") == "geometry"
    assert harness.route("Prove a statement about prime divisibility") == "number_theory"
    with TraceRecorder(tmp_path / "math.jsonl") as trace:
        answer = harness.run({"problem": "Find the angle of a triangle"}, lambda prompt: "answer", trace)
    assert answer == "answer"
