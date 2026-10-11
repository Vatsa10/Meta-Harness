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
