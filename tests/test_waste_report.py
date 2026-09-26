import json
from pathlib import Path

import pytest

from meta_harness.waste import waste_report


def make_home(tmp_path: Path, sessions: dict[str, list[dict]]) -> Path:
    home = tmp_path / "projects"
    for name, records in sessions.items():
        directory = home / "proj"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return home


def assistant(*names):
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": f"t{i}", "name": n, "input": {"file_path": "a.py"}}
        for i, n in enumerate(names)]}}


def user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def test_empty_history_reports_zeros_not_a_crash(tmp_path):
    report = waste_report(home=tmp_path / "nothing")
    assert report["sessions"] == 0
    assert report["corrections"]["count"] == 0
    assert report["corrections"]["median"] == 0
    assert report["by_project"] == []


def test_correction_lag_is_the_number_of_calls_in_the_stretch(tmp_path):
    home = make_home(tmp_path, {"s1": [
        user("go"), assistant("Read", "Edit", "Bash", "Read"), user("no, wrong"),
    ]})
    report = waste_report(home=home)
    assert report["corrections"]["count"] == 1
    assert report["corrections"]["median"] == 4
    assert report["corrections"]["calls_burned"] == 4


def test_percentiles_use_nearest_rank(tmp_path):
    # lags 3,4,5,6,100 -> median 5, max 100
    sessions = {}
    for i, extra in enumerate([0, 1, 2, 3, 97]):
        sessions[f"s{i}"] = [user("go"), assistant(*(["Read"] * (3 + extra))), user("no, wrong")]
    report = waste_report(home=make_home(tmp_path, sessions))
    assert report["corrections"]["median"] == 5
    assert report["corrections"]["max"] == 100


def test_a_stretch_that_ends_normally_is_not_counted_as_a_correction(tmp_path):
    home = make_home(tmp_path, {"s1": [user("go"), assistant("Read", "Edit", "Bash"), user("thanks")]})
    assert waste_report(home=home)["corrections"]["count"] == 0
    assert waste_report(home=home)["stretches"] == 1


def test_the_report_states_that_its_correction_count_is_a_heuristic(tmp_path):
    report = waste_report(home=tmp_path / "nothing")
    assert "heuristic" in report["caveat"].lower()


# --- fix round 1: tool_calls must count every call, not only calls inside qualifying stretches -

def test_tool_calls_counts_every_call_even_outside_a_qualifying_stretch(tmp_path):
    # A 2-call stretch (below the default min_calls=3) is not a "stretch" but its calls still
    # happened and must still be in the headline count.
    home = make_home(tmp_path, {"s1": [user("go"), assistant("Read", "Edit"), user("thanks")]})
    report = waste_report(home=home)
    assert report["tool_calls"] == 2
    assert report["stretches"] == 0


def test_tool_calls_matches_the_sum_across_a_qualifying_and_a_short_stretch(tmp_path):
    home = make_home(tmp_path, {"s1": [
        user("go"), assistant("Read", "Edit", "Bash"), user("ok"),
        assistant("Read"), user("thanks"),
    ]})
    report = waste_report(home=home)
    assert report["tool_calls"] == 4
    assert report["stretches"] == 1


# --- fix round 1: "identical" failures must actually be identical, not just both "other" ------

def tool_use(call_id, name):
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": call_id, "name": name, "input": {}}]}}


def error_result(call_id, text):
    return {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": call_id, "is_error": True, "content": text}]}}


def test_two_different_unclassified_errors_in_one_session_are_not_counted_as_a_repeat(tmp_path):
    home = make_home(tmp_path, {"s1": [
        user("go"),
        tool_use("a", "Bash"), error_result("a", "some completely unrelated failure happened"),
        tool_use("b", "Bash"), error_result("b", "a totally different problem occurred here"),
        user("thanks"),
    ]})
    report = waste_report(home=home)
    assert report["repeats"]["wasted_retries"] == 0


def test_the_same_unclassified_error_twice_is_counted_as_a_repeat(tmp_path):
    home = make_home(tmp_path, {"s1": [
        user("go"),
        tool_use("a", "Bash"), error_result("a", "connection refused at 10.0.0.1:5432"),
        tool_use("b", "Bash"), error_result("b", "connection refused at 10.0.0.7:5432"),
        user("thanks"),
    ]})
    report = waste_report(home=home)
    assert report["repeats"]["wasted_retries"] == 1
    assert report["repeats"]["sessions_affected"] == 1
