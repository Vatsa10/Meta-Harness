import pytest
from meta_harness import protocol


class Old:
    def run(self, task, model, trace):
        trace.event("x")
        return "Fruit"


class New:
    def __init__(self): self.seen = []
    def run(self, task, model): return {"output": "fruit", "why": "w"}
    def learn(self, task, result): self.seen.append((task["label"], result["output"]))


class Trace:
    def event(self, *a, **k): pass


def test_both_signatures_normalise_to_output_dict():
    assert protocol.call_run(Old(), {}, None, Trace()) == {"output": "Fruit"}
    assert protocol.call_run(New(), {}, None, Trace())["why"] == "w"


def test_learn_is_optional_and_receives_result():
    h = New()
    protocol.call_learn(h, {"label": "fruit"}, {"output": "fruit"}, None)
    protocol.call_learn(Old(), {"label": "x"}, {"output": "y"}, None)
    assert h.seen == [("fruit", "fruit")]


def test_learn_receives_model_when_it_takes_three_params():
    class L:
        got = None
        def learn(self, task, result, model): L.got = model
    protocol.call_learn(L(), {}, {"output": ""}, "M")
    assert L.got == "M"


def test_expected_value_and_hide_target():
    assert protocol.expected_value({"answer": 2, "label": "l"}) == "l"
    assert protocol.expected_value({"input": 1}) is None
    assert protocol.hide_target({"input": 1, "label": "l", "answer": 2}) == {"input": 1}


@pytest.mark.parametrize("name,task,output,score", [
    ("exact", {"label": "Fruit "}, "fruit", 1.0), ("exact", {"expected": "a"}, "b", 0.0),
    ("contains", {"answer": "Paris"}, "It is paris.", 1.0), ("numeric", {"answer": "1/2"}, "0.5", 1.0),
])
def test_builtin_metrics(name, task, output, score):
    assert protocol.resolve_metric(name)(task, {"output": output}) == score


def test_module_and_file_metrics(tmp_path, monkeypatch):
    (tmp_path / "mymetric.py").write_text("def score(task, result):\n    return 0.25\n", encoding="utf-8")
    assert protocol.resolve_metric("mymetric:score", base=tmp_path)({}, {}) == 0.25
    assert protocol.resolve_metric(f"{tmp_path / 'mymetric.py'}:score")({}, {}) == 0.25


def test_extra_metrics_after_builtins():
    f = lambda t, r: 0.5
    assert protocol.resolve_metric("custom", extra={"custom": f}) is f
    assert protocol.resolve_metric("exact", extra={"exact": f}) is protocol.exact


@pytest.mark.parametrize("bad", ["nope", "C:score", "mod:", ":fn"])
def test_bad_metric_specs(bad):
    with pytest.raises((ValueError, ModuleNotFoundError)):
        protocol.resolve_metric(bad)
