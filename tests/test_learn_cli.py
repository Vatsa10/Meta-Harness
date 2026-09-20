import json
from pathlib import Path

from meta_harness import __main__ as cli
from meta_harness.harness_store import Artifact, HarnessStore


def test_status_reports_staged_and_installed(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    store = HarnessStore(tmp_path)
    store.stage(Artifact(id="a1", type="rule", origin={"signature": "s1"}, payload="p",
                         replay={}, scores={"kept": True}))
    assert cli.main(["learn", "--status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["staged"][0]["id"] == "a1"
    assert payload["installed"] == []


def test_accept_moves_a_staged_artifact(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    HarnessStore(tmp_path).stage(Artifact(id="a1", type="doctrine",
                                          origin={"signature": "s1"}, payload="p", replay={}))
    assert cli.main(["learn", "--accept", "a1"]) == 0
    assert [a.id for a in HarnessStore(tmp_path).list_installed()] == ["a1"]


def test_reject_wrong_tombstones(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    HarnessStore(tmp_path).stage(Artifact(id="a1", type="rule",
                                          origin={"signature": "sig1"}, payload="p", replay={}))
    assert cli.main(["learn", "--reject", "a1", "--wrong"]) == 0
    assert HarnessStore(tmp_path).is_tombstoned("sig1") is True


def test_dry_run_selects_without_proposing(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "load_sessions", lambda **kw: [])
    assert cli.main(["learn", "--dry-run"]) == 0
    assert "no uncovered failure" in capsys.readouterr().out


def test_accepting_an_unknown_id_reports_an_error(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    assert cli.main(["learn", "--accept", "nope"]) == 1
    assert "nope" in capsys.readouterr().out
