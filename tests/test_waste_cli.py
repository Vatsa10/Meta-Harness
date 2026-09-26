import json

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
