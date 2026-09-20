import json
from pathlib import Path

from meta_harness.cc_history import FailureEpisode
from meta_harness.replay import expectation_for, load_agent_steps, verify_expectation


def _episode(kind="tool_error", tools=("Bash",), text="UnicodeDecodeError: charmap"):
    return FailureEpisode(session_id="s", project="p", kind=kind, turn_index=1,
                          tools=list(tools), assistant_text=text)


def _steps(*items):
    return list(items)


def test_expectation_for_a_tool_error_names_tool_and_cause():
    assert expectation_for(_episode()) == {
        "no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}}


def test_expectation_for_thrash_carries_window_and_threshold():
    assert expectation_for(_episode(kind="thrash", tools=("Edit",))) == {
        "no_thrash": {"tool": "Edit", "window": 6, "threshold": 4}}


def test_tool_error_expectation_fails_when_the_error_recurs():
    steps = _steps({"role": "tool_result", "content": "UnicodeDecodeError: charmap codec"})
    assert verify_expectation({"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}},
                              steps) is False


def test_tool_error_expectation_passes_when_it_does_not():
    steps = _steps({"role": "tool_result", "content": "ok"})
    assert verify_expectation({"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}},
                              steps) is True


def test_an_unrelated_error_does_not_fail_the_expectation():
    steps = _steps({"role": "tool_result", "content": "Permission denied"})
    # The replay asks whether THIS fault recurred, not whether the run was flawless.
    assert verify_expectation({"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}},
                              steps) is True


def test_thrash_expectation_fails_when_the_tool_is_hammered():
    steps = _steps(*[{"role": "tool_use", "name": "Edit"} for _ in range(5)])
    assert verify_expectation({"no_thrash": {"tool": "Edit", "window": 6, "threshold": 4}},
                              steps) is False


def test_thrash_expectation_passes_below_threshold():
    steps = _steps(*[{"role": "tool_use", "name": "Edit"} for _ in range(3)])
    assert verify_expectation({"no_thrash": {"tool": "Edit", "window": 6, "threshold": 4}},
                              steps) is True


def test_load_agent_steps_reads_only_agent_step_payloads(tmp_path: Path):
    trace = tmp_path / "t.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in [
        {"event": "agent_start", "payload": {"prompt": "x"}},
        {"event": "agent_step", "payload": {"role": "tool_use", "name": "Read"}},
        {"event": "agent_end", "payload": {"turns": 3}},
    ]), encoding="utf-8")
    assert load_agent_steps(trace) == [{"role": "tool_use", "name": "Read"}]


def test_empty_expectation_is_vacuously_true():
    assert verify_expectation({}, []) is True
