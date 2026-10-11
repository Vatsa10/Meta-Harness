import json
from pathlib import Path

from meta_harness.cache import CachedModel
from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner

INFLATING = '''
class Harness:
    def run(self, task, model, trace):
        prompt = "x" * 400
        trace.add_context_cost(len(prompt))          # old habit: characters
        return model(prompt)
'''


class Usage:
    model = "u"

    def call(self, prompt, **kwargs):
        return "ok", {"prompt_tokens": 7, "completion_tokens": 1}


def test_context_cost_is_counted_once_in_tokens(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(INFLATING, encoding="utf-8")
    result = CandidateEvaluator(Usage(), lambda p, t: 1.0).evaluate(
        source, [{"input": "a"}, {"input": "b"}], "c", tmp_path / "t.jsonl")
    assert result.context_cost == 7.0, "provider tokens only; the harness's character count is ignored"
    assert result.metrics["prompt_tokens"] == 14


class Counter:
    model = "counter"

    def __init__(self):
        self.n = 0

    def call(self, prompt, **kwargs):
        self.n += 1
        return f"sample-{self.n}", {"prompt_tokens": 1}


def test_repeats_are_distinct_samples_but_reruns_hit_the_cache(tmp_path: Path):
    upstream = Counter()
    model = CachedModel(upstream, tmp_path / "cache")
    assert model.call("p", sample_index=0)[0] != model.call("p", sample_index=1)[0]
    assert model.call("p", sample_index=1)[0] == "sample-2" and upstream.n == 2


def test_search_repeats_reach_the_model(tmp_path: Path):
    upstream = Counter()
    seed = tmp_path / "s.py"
    seed.write_text("class Harness:\n    def run(self, task, model, trace):\n        return model('same')\n",
                    encoding="utf-8")
    runner = SearchRunner(SearchConfig(root=tmp_path / "r", iterations=0, repeats=3),
                          CandidateEvaluator(CachedModel(upstream, tmp_path / "c"), lambda p, t: 1.0),
                          lambda *a: [])
    runner.run([seed], [{"input": "a"}])
    assert upstream.n == 3


MATH_ONLY = '''
class Harness:
    def run(self, task, model, trace):
        return model("Solve: " + task["problem"])
'''


def test_validation_probe_is_the_first_search_task(tmp_path):
    seed = tmp_path / "m.py"
    seed.write_text(MATH_ONLY, encoding="utf-8")
    runner = SearchRunner(SearchConfig(root=tmp_path / "r", iterations=0),
                          CandidateEvaluator(lambda p, **_: "4", lambda p, t: float(p == t["answer"])),
                          lambda *a: [])
    [result] = runner.run([seed], [{"problem": "2+2", "answer": "4"}])
    assert result.valid, result.error           # old probe {"input": "validation"} -> KeyError 'problem'


def test_probe_does_not_carry_the_answer(tmp_path):
    peek = tmp_path / "p.py"
    peek.write_text("class Harness:\n    def run(self, task, model, trace):\n"
                    "        assert 'answer' not in task\n        return ''\n", encoding="utf-8")
    runner = SearchRunner(SearchConfig(root=tmp_path / "r", iterations=0),
                          CandidateEvaluator(lambda p, **_: "", lambda p, t: 0.0), lambda *a: [])
    runner.run([peek], [{"problem": "x", "answer": "y"}])
    scores = json.loads((tmp_path / "r" / "candidates" / "initial-0000" / "scores.json").read_text(encoding="utf-8"))
    assert not (scores["error"] or "").startswith("validation")


def test_proposer_prompt_has_no_shell_placeholders():
    from meta_harness.proposer_agent import PROMPT_PATH, render_prompt
    text = render_prompt(PROMPT_PATH.read_text(encoding="utf-8"), iteration=7, count=1, index=0,
                         domain="math", max_tool_calls=40, objective="o")
    assert "$META_HARNESS" not in text and "iteration 7" in text


def test_proposer_log_is_copied_into_the_candidate(tmp_path):
    from test_proposer_agent import WRITE, run_search
    root, _ = run_search(tmp_path, [{"tool_calls": [WRITE]}])
    candidate = root / "candidates" / "iteration-0001-00"
    log = (candidate / "proposer_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert log and json.loads(log[-1])["role"] == "tool"
    assert json.loads((candidate / "outcomes.json").read_text(encoding="utf-8"))["dispatched"] == 1
