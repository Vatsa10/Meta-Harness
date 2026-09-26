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
