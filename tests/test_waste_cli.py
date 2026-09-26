import json
import os

import pytest

from meta_harness.__main__ import build_parser, main


def test_waste_json_prints_the_report_verbatim(tmp_path, capsys):
    code = main(["waste", "--json", "--home", str(tmp_path / "nothing")])
    captured = json.loads(capsys.readouterr().out)
    assert code == 0
    assert captured["sessions"] == 0
    assert "corrections" in captured


def test_waste_human_output_states_the_caveat(tmp_path, capsys):
    main(["waste", "--home", str(tmp_path / "nothing")])
    assert "heuristic" in capsys.readouterr().out.lower()


def test_waste_accepts_the_documented_flags():
    parser = build_parser()
    args = parser.parse_args(["waste", "--this-project", "--limit", "5", "--json"])
    assert args.this_project and args.limit == 5 and args.json


# --- fix round 1: --this-project matching zero sessions must not fail silently ---------------

def test_this_project_with_zero_matches_prints_a_loud_warning_naming_the_slug(tmp_path, capsys):
    code = main(["waste", "--this-project", "--home", str(tmp_path / "nothing")])
    captured = capsys.readouterr()
    assert code == 0
    assert "no sessions found" in captured.err.lower()
    assert "--this-project" in captured.err


def test_this_project_with_zero_matches_still_prints_the_json_report_verbatim_on_stdout(tmp_path, capsys):
    code = main(["waste", "--this-project", "--json", "--home", str(tmp_path / "nothing")])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert code == 0
    assert report["sessions"] == 0


# --- fix round 2: `meta-harness waste` also writes <harness_home>/waste.json ------------------

def test_waste_writes_waste_json_under_harness_home(tmp_path, capsys, monkeypatch):
    harness_home = tmp_path / "harness"
    monkeypatch.setenv("META_HARNESS_HOME", str(harness_home))
    code = main(["waste", "--json", "--home", str(tmp_path / "nothing")])
    capsys.readouterr()
    assert code == 0
    written = json.loads((harness_home / "waste.json").read_text(encoding="utf-8"))
    assert written["sessions"] == 0
    assert "corrections" in written


def test_waste_writes_waste_json_in_human_mode_too(tmp_path, capsys, monkeypatch):
    harness_home = tmp_path / "harness"
    monkeypatch.setenv("META_HARNESS_HOME", str(harness_home))
    code = main(["waste", "--home", str(tmp_path / "nothing")])
    capsys.readouterr()
    assert code == 0
    assert (harness_home / "waste.json").is_file()


def test_waste_json_write_uses_a_temp_file_and_rename(tmp_path, capsys, monkeypatch):
    harness_home = tmp_path / "harness"
    monkeypatch.setenv("META_HARNESS_HOME", str(harness_home))
    main(["waste", "--json", "--home", str(tmp_path / "nothing")])
    capsys.readouterr()
    leftover_temp_files = list(harness_home.glob(".waste.json.*.tmp"))
    assert leftover_temp_files == []
    assert (harness_home / "waste.json").is_file()


def test_waste_json_write_failure_still_prints_the_report_and_exits_0(tmp_path, capsys, monkeypatch):
    # Point META_HARNESS_HOME at a path that can never be a directory (it is a file), so
    # home.mkdir(...) raises and the write is expected to fail open.
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("META_HARNESS_HOME", str(blocked))
    code = main(["waste", "--json", "--home", str(tmp_path / "nothing")])
    captured = capsys.readouterr()
    assert code == 0
    report = json.loads(captured.out)
    assert report["sessions"] == 0
    assert "waste.json" in captured.err
