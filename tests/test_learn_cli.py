import json
import os
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


def test_learn_does_not_set_workspace_root_env_var(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "load_sessions", lambda **kw: [])
    monkeypatch.delenv("META_HARNESS_WORKSPACE_ROOT", raising=False)
    assert cli.main(["learn", "--dry-run"]) == 0
    assert "META_HARNESS_WORKSPACE_ROOT" not in os.environ


def _stub_learn_cycle(monkeypatch, origin_fixed):
    """Drive `learn` up to the retention decision without running an agent."""
    from meta_harness.cc_history import FailureEpisode, Session
    from meta_harness.learn import FailureClass

    episode = FailureEpisode(session_id="s1", project="p", kind="thrash", turn_index=3,
                             tools=["Bash"])
    failure = FailureClass(signature="thrash:Bash", kind="thrash", tool="Bash", count=7,
                           episodes=[episode])
    monkeypatch.setattr(cli, "load_sessions",
                        lambda **kw: [Session(session_id="s1", project="p", path="x")])
    monkeypatch.setattr(cli, "select_target", lambda *a, **kw: failure)
    monkeypatch.setattr(cli, "build_replay", lambda *a, **kw: {
        "instruction": "go", "files": {}, "expect": {},
        "_origin": {"kind": "thrash", "session": "s1", "turn": 3, "signature": "thrash:Bash"}})
    monkeypatch.setattr(cli, "model_from_environment", lambda *a, **kw: (lambda prompt: ""))
    monkeypatch.setattr(cli, "propose_artifact", lambda *a, **kw: Artifact(
        id="thrash-Bash-deadbeef", type="rule",
        origin={"signature": "thrash:Bash", "tools": ["Bash"], "rule_kind": "repeat-call"},
        payload="do not repeat", replay={}))
    monkeypatch.setattr(cli, "run_replay", lambda *a, **kw: (origin_fixed, {"scored": True}))


def test_an_artifact_that_does_not_fix_its_origin_is_not_staged(tmp_path: Path, monkeypatch,
                                                                capsys):
    """The retention gate must actually gate: README and SKILL.md both promise it does."""
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    _stub_learn_cycle(monkeypatch, origin_fixed=False)
    assert cli.main(["learn"]) == 0
    out = json.loads(capsys.readouterr().out.split("target:", 1)[1].split("\n", 1)[1])
    assert "rejected" in out and "staged" not in out
    assert out["verdict"]["kept"] is False
    assert HarnessStore(tmp_path).list_staged() == []


def test_an_artifact_that_fixes_its_origin_is_staged_with_its_verdict(tmp_path: Path, monkeypatch,
                                                                     capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    _stub_learn_cycle(monkeypatch, origin_fixed=True)
    assert cli.main(["learn"]) == 0
    out = json.loads(capsys.readouterr().out.split("target:", 1)[1].split("\n", 1)[1])
    assert out["staged"] == "thrash-Bash-deadbeef"
    staged = HarnessStore(tmp_path).list_staged()
    assert [a.id for a in staged] == ["thrash-Bash-deadbeef"]
    # No task set was supplied, so the verdict must say the regression half did not run rather
    # than implying it passed.
    assert "not checked" in staged[0].scores["retention"]["task_set"]
    assert staged[0].scores["task_set_score"] is None


def test_an_unscored_rule_is_not_reported_as_a_score(tmp_path: Path, monkeypatch, capsys):
    """run_replay returns None when the artifact could not be applied; that is not `False`."""
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    _stub_learn_cycle(monkeypatch, origin_fixed=None)
    monkeypatch.setattr(cli, "run_replay",
                        lambda *a, **kw: (None, {"scored": False, "reason": "plugin not found"}))
    assert cli.main(["learn"]) == 0
    out = json.loads(capsys.readouterr().out.split("target:", 1)[1].split("\n", 1)[1])
    assert out["origin_fixed"] is None
    assert out["verdict"]["reason"] == "origin replay was not scored"
