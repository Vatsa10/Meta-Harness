import json
from pathlib import Path

from meta_harness import __main__ as cli
from meta_harness.receipts import K, MIN_PER_ARM


def _seed(home: Path) -> None:
    rows = [{"session": "s1", "call": 1, "decision": "acted", "artifact": "a1",
             "source": "learned-rule", "signature": "tool_error:Bash:x"}]
    (home / "receipts-s1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    obs = {"session": "s1", "call": 3, "kind": "tool_error", "tool": "Bash", "cause": "x"}
    (home / "observed-s1.jsonl").write_text(json.dumps(obs) + "\n", encoding="utf-8")


def test_report_json_prints_the_summary_table(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    _seed(tmp_path)
    assert cli.main(["receipts", "report", "--json"]) == 0
    table = json.loads(capsys.readouterr().out)
    assert table[0]["artifact"] == "a1"
    assert table[0]["verdict"] == "not enough data"
    assert table[0]["acted"] == 1 and table[0]["acted_rate"] == 1.0


def test_report_states_artifacts_verdicts_and_thresholds(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    _seed(tmp_path)
    assert cli.main(["receipts", "report"]) == 0
    out = capsys.readouterr().out
    assert "a1" in out and "not enough data" in out
    assert f"K={K}" in out and f"MIN_PER_ARM={MIN_PER_ARM}" in out


def test_report_on_an_empty_home_says_so(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    assert cli.main(["receipts", "report"]) == 0
    assert "no receipts yet" in capsys.readouterr().out


def test_export_writes_one_line_per_receipt_with_recurred(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    _seed(tmp_path)
    out = tmp_path / "case.jsonl"
    assert cli.main(["receipts", "export", "--out", str(out)]) == 0
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1 and lines[0]["recurred"] is True
