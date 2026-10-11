from meta_harness import protocol


class Memorizer:
    def __init__(self): self.memory = {}
    def run(self, task, model): return {"output": self.memory.get(task["input"], "unknown")}
    def learn(self, task, result): self.memory[task["input"]] = task["label"]


class Trace:
    def event(self, *a, **k): pass


def test_test_items_are_hidden_and_never_learned():
    h = Memorizer()
    protocol.stream_train(h, [{"input": "a", "label": "x"}], None, Trace())
    tests = [{"input": "b", "label": "y"}, {"input": "b", "label": "y"}, {"input": "a", "label": "x"}]
    scores = [s for s, _ in protocol.evaluate_frozen(h, tests, None, Trace(), protocol.resolve_metric("exact"))]
    assert scores == [0.0, 0.0, 1.0]


def test_label_is_removed_before_run():
    class Peeker:
        def run(self, task, model):
            assert "label" not in task and "answer" not in task
            return {"output": ""}
    protocol.evaluate_frozen(Peeker(), [{"input": "q", "label": "l"}], None, Trace(),
                             protocol.resolve_metric("exact"))
