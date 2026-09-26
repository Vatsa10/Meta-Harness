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
