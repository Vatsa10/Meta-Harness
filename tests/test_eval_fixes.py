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


# --- Task 10 part 2: online protocol, budget, equal-budget cap ---------------------------------

import pytest

from meta_harness import protocol
from meta_harness.httpdata import read_jsonl

ECHO = "class Harness:\n    def run(self, task, model):\n        return model(task['input'])\n"


def test_online_evaluator_learns_from_train_before_scoring(tmp_path):
    source = tmp_path / "m.py"
    source.write_text(
        "class Harness:\n"
        "    def __init__(self): self.memory = {}\n"
        "    def run(self, task, model): return {'output': self.memory.get(task['input'], 'unknown')}\n"
        "    def learn(self, task, result): self.memory[task['input']] = task['label']\n", encoding="utf-8")
    exact = protocol.resolve_metric("exact")
    tasks = [{"input": "a", "label": "x"}]
    online = CandidateEvaluator(lambda p, **_: "", exact, protocol="online", train_tasks=tasks)
    plain = CandidateEvaluator(lambda p, **_: "", exact)
    assert online.evaluate(source, tasks, "c", tmp_path / "o.jsonl").score == 1.0
    assert plain.evaluate(source, tasks, "c", tmp_path / "p.jsonl").score == 0.0


def _copying_proposer(tmp_path, seed, made):
    def proposer(experience, iteration, count):
        paths = []
        for index in range(count):
            path = tmp_path / f"p{iteration}-{index}.py"
            path.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
            paths.append(path)
        made.append(iteration)
        return paths
    return proposer


def test_budget_and_equal_budget_cap(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text("class Harness:\n    def run(self, task, model):\n        return model(task['input'])\n",
                    encoding="utf-8")
    made = []
    def proposer(experience, iteration, count):
        path = tmp_path / f"p{iteration}.py"
        path.write_text(seed.read_text(encoding="utf-8"), encoding="utf-8")
        made.append(iteration)
        return [path]
    proposer.usage = {"calls": 2, "input_tokens": 10, "output_tokens": 3}
    config = SearchConfig(root=tmp_path / "r", iterations=10, repeats=2, max_evaluations=6)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: "x", protocol.resolve_metric("exact")), proposer)
    runner.run([seed], [{"input": "a", "label": "x"}, {"input": "b", "label": "x"}])
    budget = json.loads((config.root / "run.json").read_text(encoding="utf-8"))["budget"]
    assert budget["harness_evaluations"] == 6 and made == [1, 2]      # seed + 2 proposals, x2 repeats
    assert budget["model_calls"] == 12 and budget["proposer_calls"] == 2


def test_equal_budget_cap_truncates_the_in_flight_iteration(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    made = []
    config = SearchConfig(root=tmp_path / "r", iterations=10, repeats=2, candidates_per_iteration=2,
                          max_evaluations=4)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), _copying_proposer(tmp_path, seed, made))
    runner.run([seed], [{"input": "a", "label": "x"}])
    run = json.loads((config.root / "run.json").read_text(encoding="utf-8"))
    assert run["budget"]["harness_evaluations"] == 4 and made == [1]
    receipt = json.loads((config.root / "candidates" / "iteration-0001-01" / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["verdict"] == "budget_expired" and "iteration-0001-01" not in run["frontier"]


def test_test_root_inside_root_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="test_root"):
        SearchConfig(root=tmp_path / "r", test_root=tmp_path / "r" / "test")
    with pytest.raises(ValueError, match="test_root"):
        SearchConfig(root=tmp_path / "r", test_root=tmp_path / "r")


def test_an_odd_test_root_is_denied_to_the_optimizer_as_the_split(tmp_path):
    from meta_harness.domains import generic
    from meta_harness.optimizers import OptimizerContext, build
    from test_proposer_agent import WRITE, scripted

    held_out = tmp_path / "elsewhere" / "held-out"
    config = SearchConfig(root=tmp_path / "r", iterations=1, test_root=held_out)
    rows = [{"tool_calls": [{"name": "read", "arguments": {"path": str(held_out / "test_results.json")}}]},
            {"tool_calls": [WRITE]}]
    ctx = OptimizerContext.for_config(config, scripted(tmp_path, rows), generic, max_tool_calls=5)
    assert held_out in ctx.forbidden
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), build("metaharness", ctx)).run(
        [seed], [{"input": "a", "label": "x"}])
    outcomes = json.loads((config.root / "candidates" / "iteration-0001-00" / "outcomes.json").read_text(encoding="utf-8"))
    assert outcomes["denied_split"] == 1


class Usage7:
    model = "u"

    def call(self, prompt, **kwargs):
        return "ok", {"prompt_tokens": 7, "completion_tokens": 1}


def test_trace_context_cost_reports_metered_tokens(tmp_path):
    source = tmp_path / "c.py"
    source.write_text("class Harness:\n    def run(self, task, model, trace):\n"
                      "        model('hello')\n        return str(trace.context_cost)\n", encoding="utf-8")
    result = CandidateEvaluator(Usage7()).evaluate(source, [{"input": "a", "label": "7.0"}], "c", tmp_path / "t.jsonl")
    assert result.score == 1.0


def test_stop_reason_histogram_is_in_scores(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text("class Harness:\n    def run(self, task, model):\n"
                    "        if task['input'] == 'boom': raise RuntimeError('boom')\n"
                    "        return {'output': 'x', 'stop_reason': task['input']}\n", encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=0)
    tasks = [{"input": "length", "label": "x"}, {"input": "boom", "label": "x"}, {"input": "stuck", "label": "x"}]
    SearchRunner(config, CandidateEvaluator(lambda p, **_: ""), lambda *a: []).run([seed], tasks)
    scores = json.loads((config.root / "candidates" / "initial-0000" / "scores.json").read_text(encoding="utf-8"))
    assert scores["stop_reasons"] == {"overflow": 1, "error": 1, "stuck": 1}


# --- Task 10 part 3: resume, per-task failures, evaluation timeout ----------------------------

class CountingModel:
    model = "counting"

    def __init__(self):
        self.n = 0

    def __call__(self, prompt, **kwargs):
        self.n += 1
        return "x"


def test_kill_and_resume_skips_finished_work(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    tasks = [{"input": "a", "label": "x"}]
    made = []
    copy = _copying_proposer(tmp_path, seed, made)

    def dies_at_two(experience, iteration, count):
        if iteration == 2:
            raise KeyboardInterrupt
        return copy(experience, iteration, count)

    config = SearchConfig(root=tmp_path / "r", iterations=3)
    first = CountingModel()
    with pytest.raises(KeyboardInterrupt):
        SearchRunner(config, CandidateEvaluator(first), dies_at_two).run([seed], tasks)
    run = json.loads((config.root / "run.json").read_text(encoding="utf-8"))
    assert run["cursor"]["iteration"] == 1 and first.n == 2
    second = CountingModel()
    frontier = SearchRunner(config, CandidateEvaluator(second), copy).run([seed], tasks, resume=True)
    assert second.n == 2, "no seed and no iteration-1 candidate is evaluated again"
    assert made == [1, 2, 3]
    assert (config.root / "candidates" / "iteration-0003-00" / "scores.json").is_file()
    run = json.loads((config.root / "run.json").read_text(encoding="utf-8"))
    assert run["cursor"]["iteration"] == 3 and run["budget"]["harness_evaluations"] == 4
    assert "initial-0000" in {r.candidate_id for r in frontier}


def test_a_task_exception_scores_zero_and_the_candidate_stays_valid(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text("class Harness:\n    def run(self, task, model):\n"
                    "        if task['input'] == 'b': raise ValueError('bad task')\n"
                    "        return model(task['input'])\n", encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=0)
    tasks = [{"input": "a", "label": "x"}, {"input": "b", "label": "x"}, {"input": "c", "label": "x"}]
    [result] = SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), lambda *a: []).run([seed], tasks)
    assert result.valid and result.score == pytest.approx(2 / 3)
    assert result.metrics["task_failures"] == 1
    trace = config.root / "candidates" / "initial-0000" / "traces.jsonl"
    ends = [e for e in read_jsonl(trace) if e["event"] == "task_end"]
    assert "bad task" in ends[1]["error"] and ends[1]["payload"]["reward"] == 0.0


def test_evaluation_timeout_fails_tasks_without_retry(tmp_path):
    import time
    source = tmp_path / "slow.py"
    source.write_text("import time\nclass Harness:\n    def run(self, task, model):\n"
                      "        time.sleep(0.6)\n        return model('late')\n", encoding="utf-8")
    model = CountingModel()
    evaluator = CandidateEvaluator(model, task_timeout=5.0, evaluation_timeout=0.3)
    tasks = [{"input": str(i), "label": "x"} for i in range(3)]
    started = time.monotonic()
    result = evaluator.evaluate(source, tasks, "c", tmp_path / "t.jsonl")
    assert time.monotonic() - started < 2.0
    assert result.valid and result.score == 0.0 and result.metrics["task_failures"] == 3
    assert result.abort_reason == "deadline"
    time.sleep(0.6)
    assert model.n == 0, "the abandoned harness thread cannot spend model calls"


# --- Task 10 part 4: receipts, acceptance gate, crash repair, redaction ----------------------

def _receipt(root, cid):
    return json.loads((root / "candidates" / cid / "receipt.json").read_text(encoding="utf-8"))


def test_receipt_is_incomplete_when_evaluation_dies_then_accepted_on_resume(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    made = []
    config = SearchConfig(root=tmp_path / "r", iterations=1)

    class Dying(CandidateEvaluator):
        def evaluate(self, source, tasks, candidate_id, *args, **kwargs):
            if candidate_id.startswith("iteration-"):
                raise KeyboardInterrupt
            return super().evaluate(source, tasks, candidate_id, *args, **kwargs)

    tasks = [{"input": "a", "label": "x"}]
    with pytest.raises(KeyboardInterrupt):
        SearchRunner(config, Dying(lambda p, **_: "x"), _copying_proposer(tmp_path, seed, made)).run([seed], tasks)
    assert _receipt(config.root, "iteration-0001-00")["verdict"] == "incomplete"
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), _copying_proposer(tmp_path, seed, made)).run(
        [seed], tasks, resume=True)
    receipt = _receipt(config.root, "iteration-0001-00")
    assert receipt["verdict"] == "accepted" and receipt["validation"]["ok"] is True
    assert made == [1], "the proposal on disk is reused, not proposed again"


def test_source_naming_the_test_root_is_a_split_violation(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=1)
    cheat = tmp_path / "cheat.py"
    cheat.write_text(f"PATH = {str(tmp_path / 'r-test')!r}\n" + ECHO, encoding="utf-8")
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), lambda *a: [cheat]).run(
        [seed], [{"input": "a", "label": "x"}])
    assert _receipt(config.root, "iteration-0001-00")["verdict"] == "rejected_split_violation"
    frontier = json.loads((config.root / "frontier.json").read_text(encoding="utf-8"))
    assert [f["candidate_id"] for f in frontier] == ["initial-0000"]


def test_changed_experience_between_proposal_and_evaluation_is_stale(tmp_path):
    from meta_harness.proposer_agent import fingerprint
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=1)

    def proposer(experience, iteration, count):
        path = tmp_path / "cand.py"
        path.write_text(ECHO, encoding="utf-8")
        (tmp_path / "cand.proposal.json").write_text(
            json.dumps({"candidate": "cand", "fingerprint": fingerprint(experience.root), "parents_read": []}),
            encoding="utf-8")
        scores = experience.root / "candidates" / "initial-0000" / "scores.json"
        scores.write_text(scores.read_text(encoding="utf-8").replace('"score": 1.0', '"score": 0.5'),
                          encoding="utf-8")
        return [path]

    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), proposer).run([seed], [{"input": "a", "label": "x"}])
    receipt = _receipt(config.root, "iteration-0001-00")
    assert receipt["verdict"] == "stale_parent"
    assert receipt["fingerprint"].startswith("v1:")


def test_torn_trace_line_does_not_break_resume(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    tasks = [{"input": "a", "label": "x"}]
    SearchRunner(SearchConfig(root=tmp_path / "r", iterations=0), CandidateEvaluator(lambda p, **_: "x"),
                 lambda *a: []).run([seed], tasks)
    trace = tmp_path / "r" / "candidates" / "initial-0000" / "traces.jsonl"
    with trace.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "model_ca')
    made = []
    frontier = SearchRunner(SearchConfig(root=tmp_path / "r", iterations=1), CandidateEvaluator(lambda p, **_: "x"),
                            _copying_proposer(tmp_path, seed, made)).run([seed], tasks, resume=True)
    assert made == [1] and frontier
    assert read_jsonl(trace, tolerate_torn=False)


def test_a_fourth_automatic_resume_raises(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=0)
    tasks = [{"input": "a", "label": "x"}]

    def runner():
        return SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), lambda *a: [])

    runner().run([seed], tasks)
    for _ in range(3):
        runner().run([seed], tasks, resume=True)
    with pytest.raises(RuntimeError, match="force-resume"):
        runner().run([seed], tasks, resume=True)
    assert runner().run([seed], tasks, resume=True, force_resume=True)


def test_secrets_in_model_responses_are_redacted_in_traces(tmp_path):
    source = tmp_path / "s.py"
    source.write_text(ECHO, encoding="utf-8")
    trace = tmp_path / "t.jsonl"
    CandidateEvaluator(lambda p, **_: "token: Bearer abc123def").evaluate(source, [{"input": "a"}], "c", trace)
    text = trace.read_text(encoding="utf-8")
    assert "[redacted]" in text and "abc123def" not in text


# --- Task 10 review fixes ---------------------------------------------------------------------

def test_held_out_split_is_read_once_across_resumes(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    tasks, test = [{"input": "a", "label": "x"}], [{"input": "t", "label": "x"}]
    better = tmp_path / "better.py"
    better.write_text("class Harness:\n    def run(self, task, model):\n        return 'x'\n", encoding="utf-8")
    model = CountingModel()
    config = SearchConfig(root=tmp_path / "r", iterations=0)
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "y"), lambda *a: []).run([seed], tasks, test)
    results = tmp_path / "r-test" / "test_results.json"
    first = results.read_bytes()
    budget = json.loads((config.root / "run.json").read_text(encoding="utf-8"))["budget"]["test_evaluations"]
    SearchRunner(config, CandidateEvaluator(model), lambda *a: []).run([seed], tasks, test, resume=True)
    assert model.n == 0 and results.read_bytes() == first
    assert json.loads((config.root / "run.json").read_text(encoding="utf-8"))["budget"]["test_evaluations"] == budget
    more = SearchConfig(root=tmp_path / "r", iterations=1)
    SearchRunner(more, CandidateEvaluator(lambda p, **_: "y"), lambda *a: [better]).run(
        [seed], tasks, test, resume=True)
    assert (tmp_path / "r-test" / "test_results.1.json").read_bytes() == first
    ids = [r["candidate_id"] for r in json.loads(results.read_text(encoding="utf-8"))["results"]]
    assert ids == ["iteration-0001-00"]
    assert json.loads((config.root / "run.json").read_text(encoding="utf-8"))["budget"]["test_evaluations"] == budget + 1


def test_budget_cap_covers_baselines(tmp_path):
    seeds = []
    for i in range(3):
        seeds.append(tmp_path / f"s{i}.py")
        seeds[-1].write_text(ECHO, encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=2, repeats=2, max_evaluations=3)
    made = []
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), _copying_proposer(tmp_path, seeds[0], made)).run(
        seeds, [{"input": "a", "label": "x"}])
    run = json.loads((config.root / "run.json").read_text(encoding="utf-8"))
    assert run["budget"]["harness_evaluations"] == 2 and made == []
    assert _receipt(config.root, "initial-0002")["verdict"] == "budget_expired"


def test_proposals_without_a_fingerprint_get_one_from_the_runner(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=1)

    def command_like(experience, iteration, count):
        path = tmp_path / "plain.py"
        path.write_text(ECHO, encoding="utf-8")
        scores = experience.root / "candidates" / "initial-0000" / "scores.json"
        scores.write_text(scores.read_text(encoding="utf-8").replace('"score": 1.0', '"score": 0.5'),
                          encoding="utf-8")
        return [path]

    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), command_like).run([seed], [{"input": "a", "label": "x"}])
    assert _receipt(config.root, "iteration-0001-00")["verdict"] == "stale_parent"


def test_torn_test_trace_is_repaired_on_resume(tmp_path):
    seed = tmp_path / "s.py"
    seed.write_text(ECHO, encoding="utf-8")
    config = SearchConfig(root=tmp_path / "r", iterations=0)
    tasks = [{"input": "a", "label": "x"}]
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), lambda *a: []).run([seed], tasks, tasks)
    trace = tmp_path / "r-test" / "traces" / "initial-0000.jsonl"
    with trace.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "tor')
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "x"), lambda *a: []).run([seed], tasks, tasks, resume=True)
    assert read_jsonl(trace, tolerate_torn=False)
