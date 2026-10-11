import json

import pytest

from meta_harness.core import CandidateEvaluator, FilesystemExperience, SearchConfig, SearchRunner
from meta_harness.proposer_agent import (AgentProposer, PROMPT_PATH, ProposerStuck,
                                         ProposerTruncated, fingerprint, render_prompt)
from meta_harness.proposer_tools import ToolSandbox
from meta_harness.providers import model_from_environment
from meta_harness.timeouts import Aborted

SEED = "class Harness:\n    def run(self, task, model, trace):\n        return model(task['input'])\n"
NEW = "class Harness:\n    def run(self, task, model, trace):\n        return 'fruit'\n"
WRITE = {"name": "write_candidate", "arguments": {"source": NEW, "rationale": "always fruit"}}


def scripted(tmp_path, rows, name="p.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return model_from_environment("scripted", str(path))


def run_search(tmp_path, proposer_rows, iterations=1, max_tool_calls=5, **kwargs):
    tmp_path.mkdir(parents=True, exist_ok=True)
    seed = tmp_path / "seed.py"
    seed.write_text(SEED, encoding="utf-8")
    proposer = AgentProposer(scripted(tmp_path, proposer_rows), max_tool_calls=max_tool_calls, **kwargs)
    config = SearchConfig(root=tmp_path / "store", iterations=iterations)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: "vehicle"), proposer)
    tasks = [{"input": "apple", "label": "fruit"}, {"input": "car", "label": "vehicle"}]
    runner.run([seed], tasks, [{"input": "pear", "label": "fruit"}])
    return config.root, proposer


def proposals(root, iteration=1):
    return root / "proposals" / f"iteration-{iteration:04d}"


def read_log(root, iteration=1):
    path = proposals(root, iteration) / "candidate-00.proposer_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def tool_records(log):
    return [e for e in log if e.get("role") == "tool"]


def outcomes(root):
    return json.loads((proposals(root) / "candidate-00.outcomes.json").read_text(encoding="utf-8"))


def error_text(root):
    return (proposals(root) / "proposer.error").read_text(encoding="utf-8")


def test_agent_reads_then_writes_and_logs_every_call(tmp_path):
    root, proposer = run_search(tmp_path, [
        {"tool_calls": [{"name": "list", "arguments": {"path": "candidates"}}]},
        {"tool_calls": [{"name": "read", "arguments": {"path": "candidates/initial-0000/scores.json"}}]},
        {"tool_calls": [WRITE]},
    ])
    candidate = root / "candidates" / "iteration-0001-00"
    assert (candidate / "harness.py").read_text(encoding="utf-8") == NEW
    assert (candidate / "proposer_reasoning.md").read_text(encoding="utf-8") == "always fruit"
    log = tool_records(read_log(root))
    assert [e["tool"] for e in log] == ["list", "read", "write_candidate"]
    assert log[1]["path"] == "candidates/initial-0000/scores.json"
    assert log[2]["arguments"]["source"] == len(NEW)
    assert proposer.usage["calls"] == 3
    proposal = json.loads((proposals(root) / "candidate-00.proposal.json").read_text(encoding="utf-8"))
    assert proposal["parents_read"] == ["initial-0000"]
    assert proposal["fingerprint"].startswith("v1:")


def test_agent_cannot_read_the_test_split(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the sandbox read must never run for a denied path")
    monkeypatch.setattr(ToolSandbox, "read", boom)
    root, _ = run_search(tmp_path, [
        {"tool_calls": [{"name": "read", "arguments": {"path": "../store-test/test_results.json"}}]},
        {"tool_calls": [WRITE]},
    ])
    log = tool_records(read_log(root))
    assert log[0]["ok"] is False and log[0]["denied"] is True


def test_budget_exhaustion_is_recorded_not_fatal(tmp_path):
    root, _ = run_search(tmp_path, [{"tool_calls": [{"name": "list", "arguments": {}}]}] * 5)
    assert "ProposerBudgetExceeded" in error_text(root)


def test_turns_without_tool_calls_count_toward_the_budget(tmp_path):
    root, _ = run_search(tmp_path, [{"text": "thinking"}] * 6)
    assert "ProposerBudgetExceeded" in error_text(root)


def test_write_candidate_ends_the_turn(tmp_path):
    root, _ = run_search(tmp_path, [
        {"tool_calls": [WRITE, {"name": "read", "arguments": {"path": "run.json"}}]},
    ])
    log = tool_records(read_log(root))
    assert [e["tool"] for e in log] == ["write_candidate", "read"]
    assert log[1]["executed"] is False and log[1]["ok"] is False
    assert outcomes(root)["aborted"] == 1 and outcomes(root)["dispatched"] == 1


def test_unknown_tool_is_an_error_result_and_outcomes_are_counted(tmp_path):
    root, _ = run_search(tmp_path, [
        {"tool_calls": [{"name": "read", "arguments": {"path": "../store-test/x"}}]},
        {"tool_calls": [{"name": "rm", "arguments": {"path": "run.json"}}]},
        {"tool_calls": [WRITE]},
    ])
    log = tool_records(read_log(root))
    assert [e["tool"] for e in log] == ["read", "rm", "write_candidate"]
    assert log[1]["ok"] is False
    assert outcomes(root) == {"dispatched": 2, "denied": 1, "denied_split": 1, "failed": 1, "aborted": 0}


def test_three_length_stops_raise_truncated(tmp_path):
    model = scripted(tmp_path, [{"stop_reason": "length", "text": "cut",
                                 "tool_calls": [{"name": "list", "arguments": {}}]}] * 3)
    experience = FilesystemExperience(tmp_path / "store")
    with pytest.raises(ProposerTruncated):
        AgentProposer(model, max_tool_calls=10)(experience, 1, 1)
    log = read_log(experience.root)
    assert sum(1 for e in log if e.get("event") == "length") == 3
    assert all(e["executed"] is False for e in tool_records(log))


def test_log_records_chain_by_parent_and_redact_secrets(tmp_path):
    root, _ = run_search(tmp_path, [
        {"text": "my key is sk-abcdefgh12345678", "tool_calls": [{"name": "list", "arguments": {}}]},
        {"tool_calls": [WRITE]},
    ])
    raw = (proposals(root) / "candidate-00.proposer_log.jsonl").read_text(encoding="utf-8")
    assert "sk-abcdefgh12345678" not in raw and "[redacted]" in raw
    log = read_log(root)
    assert log[0]["parent"] is None
    assert all(b["parent"] == a["id"] for a, b in zip(log, log[1:]))
    assert all("usage" in e for e in log if e["role"] == "assistant")


def test_parallel_read_only_calls_each_count(tmp_path):
    lists = [{"name": "list", "arguments": {"path": "."}} for _ in range(3)]
    root, proposer = run_search(tmp_path / "a", [{"tool_calls": lists}, {"tool_calls": [WRITE]}],
                                max_tool_calls=4)
    log = tool_records(read_log(root))
    assert [e["step"] for e in log] == [1, 1, 1, 2]
    assert all(e["executed"] for e in log)
    assert proposer.usage["calls"] == 2
    # 3 calls cost 3 of the 4: a fourth read in the first turn leaves nothing for write_candidate
    root, _ = run_search(tmp_path / "b", [{"tool_calls": lists * 2}, {"tool_calls": [WRITE]}],
                         max_tool_calls=4)
    assert "ProposerBudgetExceeded" in error_text(root)
    log = tool_records(read_log(root))
    assert [e["executed"] for e in log] == [True] * 4 + [False] * 2


def test_stuck_identical_failures_remind_once_then_stop(tmp_path):
    bad = {"tool_calls": [{"name": "read", "arguments": {"path": "nope.txt"}}]}
    model = scripted(tmp_path, [bad] * 8)
    experience = FilesystemExperience(tmp_path / "store")
    with pytest.raises(ProposerStuck):
        AgentProposer(model, max_tool_calls=20)(experience, 1, 1)
    log = read_log(experience.root)
    reminders = [e for e in log if e.get("event") == "reminder"]
    assert len(reminders) == 1 and reminders[0]["step"] == 5
    assert [e for e in log if e.get("event") == "stop"][0]["step"] == 8


def test_a_different_argument_breaks_the_streak(tmp_path):
    rows = [{"tool_calls": [{"name": "read", "arguments": {"path": "nope.txt", "offset": 3 if i == 3 else 0}}]}
            for i in range(8)] + [{"tool_calls": [WRITE]}]
    experience = FilesystemExperience(tmp_path / "store")
    assert len(AgentProposer(scripted(tmp_path, rows), max_tool_calls=20)(experience, 1, 1)) == 1
    assert not any(e.get("event") == "stop" for e in read_log(experience.root))


def test_fingerprint_changes_when_scores_change(tmp_path):
    root = tmp_path / "store"
    (root / "candidates" / "a").mkdir(parents=True)
    (root / "candidates" / "a" / "scores.json").write_text('{"score": 1}', encoding="utf-8")
    first = fingerprint(root)
    (root / "candidates" / "a" / "scores.json").write_text('{"score": 2}', encoding="utf-8")
    assert fingerprint(root) != first


def test_idle_watchdog_aborts(tmp_path):
    now = [0.0]

    def clock():
        now[0] += 6.0
        return now[0]

    model = scripted(tmp_path, [{"tool_calls": [{"name": "list", "arguments": {}}]}] * 3)
    experience = FilesystemExperience(tmp_path / "store")
    proposer = AgentProposer(model, max_tool_calls=10, idle=10, deadline=10_000, clock=clock)
    with pytest.raises(Aborted) as info:
        proposer(experience, 1, 1)
    assert info.value.reason == "idle"
    assert any(e.get("event") == "aborted" and e["reason"] == "idle" for e in read_log(experience.root))


def test_model_without_chat_is_rejected():
    with pytest.raises(TypeError, match="tool calling"):
        AgentProposer(lambda prompt: prompt)


def test_prompt_template_renders_without_leftovers():
    text = render_prompt(PROMPT_PATH.read_text(encoding="utf-8"), iteration=3, count=2, index=0,
                         domain="classification", max_tool_calls=40,
                         objective="maximize score, then minimize context tokens")
    assert "{" not in text.replace("{{", "") and "$META_HARNESS" not in text
    assert "iteration 3" in text
    assert "test split is not available" in text and "write_candidate" in text
    with pytest.raises(KeyError):
        render_prompt("iteration {iteration} {missing}", iteration=1)


def test_grep_on_a_missing_path_is_an_error_not_no_matches(tmp_path):
    root, _ = run_search(tmp_path, [
        {"tool_calls": [{"name": "grep", "arguments": {"pattern": "reward",
                                                       "path": "candidates/initial-0000/nope.jsonl"}}]},
        {"tool_calls": [WRITE]},
    ])
    log = tool_records(read_log(root))
    assert log[0]["ok"] is False


def test_seed_view_prompt_says_traces_are_absent(tmp_path):
    seen = []

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def chat(self, system, messages, tools, **kwargs):
            seen.append(messages[0]["content"])
            return self.inner.chat(system, messages, tools, **kwargs)

    seed = tmp_path / "seed.py"
    seed.write_text(SEED, encoding="utf-8")
    proposer = AgentProposer(Spy(scripted(tmp_path, [{"tool_calls": [WRITE]}])), view="seed",
                             max_tool_calls=5)
    config = SearchConfig(root=tmp_path / "store", iterations=1)
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "vehicle"), proposer).run(
        [seed], [{"input": "apple", "label": "fruit"}], [{"input": "pear", "label": "fruit"}])
    assert "traces.jsonl, proposer_reasoning.md and summary.md are absent" in seen[0]
    assert (config.root / "candidates" / "iteration-0001-00" / "harness.py").is_file()


def test_proposer_is_told_to_write_before_the_budget_runs_out(tmp_path):
    reads = [{"tool_calls": [{"name": "list", "arguments": {"path": f"candidates{'/' * i}"}}]}
             for i in range(1, 8)]
    root, _ = run_search(tmp_path, [*reads, {"tool_calls": [WRITE]}], max_tool_calls=10)
    log = read_log(root)
    events = [e for e in log if e.get("event") == "budget_reminder"]
    assert len(events) == 1 and events[0]["left"] == 3
    assert (root / "candidates" / "iteration-0001-00" / "harness.py").is_file()
