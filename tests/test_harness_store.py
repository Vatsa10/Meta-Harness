import json
from pathlib import Path

import pytest

from meta_harness.harness_store import Artifact, HarnessStore, harness_home


def _artifact(id="a1", type="rule", signature="tool_error:Bash:cp1252"):
    return Artifact(id=id, type=type,
                    origin={"kind": "tool_error", "signature": signature,
                            "session": "s1", "turn": 12},
                    payload="deny Edit on unread files",
                    replay={"instruction": "x", "files": {}, "expect": {}},
                    scores={"origin_fixed": True}, sources=["s1#12"], created="2026-09-20")


def test_home_respects_the_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path / "h"))
    assert harness_home() == tmp_path / "h"


def test_stage_then_accept_moves_and_registers(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    assert [a.id for a in store.list_staged()] == ["a1"]
    store.accept("a1")
    assert store.list_staged() == []
    assert [a.id for a in store.list_installed()] == ["a1"]
    assert "tool_error:Bash:cp1252" in store.installed_signatures()


def test_reject_archives_without_tombstoning(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    store.reject("a1")
    assert store.list_staged() == []
    assert (tmp_path / "archive" / "a1" / "artifact.json").is_file()
    assert store.is_tombstoned("tool_error:Bash:cp1252") is False


def test_reject_as_wrong_tombstones_the_signature(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    store.reject("a1", wrong=True)
    # A tombstoned signature must never be proposed again.
    assert store.is_tombstoned("tool_error:Bash:cp1252") is True


def test_payload_round_trips(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    store.accept("a1")
    assert store.list_installed()[0].payload == "deny Edit on unread files"


def test_accepting_an_unknown_id_raises(tmp_path: Path):
    with pytest.raises(KeyError):
        HarnessStore(tmp_path).accept("nope")


def test_rejects_an_unknown_artifact_type(tmp_path: Path):
    with pytest.raises(ValueError):
        HarnessStore(tmp_path).stage(_artifact(type="telepathy"))
