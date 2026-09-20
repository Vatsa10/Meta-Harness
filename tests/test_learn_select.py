import json
from pathlib import Path

from meta_harness.cc_history import parse_session
from meta_harness.harness_store import Artifact, HarnessStore
from meta_harness.learn import rank_failures, select_target


def _transcript(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _error_session(tmp_path: Path, name: str, tool: str, text: str, times: int):
    records = [{"type": "user", "message": {"content": "do the thing"}}]
    for index in range(times):
        records.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"t{index}", "name": tool, "input": {}}]},
            "cwd": "D:\\repo", "timestamp": "2026-09-20T10:00:00Z"})
        records.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{index}", "is_error": True,
             "content": text}]}})
    return parse_session(_transcript(tmp_path / "proj" / f"{name}.jsonl", records),
                         include_text=True)


def test_ranks_the_most_frequent_failure_first(tmp_path: Path):
    sessions = [
        _error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 3),
        _error_session(tmp_path, "b", "Read", "Permission denied", 1),
    ]
    ranked = rank_failures(sessions)
    assert ranked[0].signature == "tool_error:Bash:unicode-decode"
    assert ranked[0].count == 3


def test_select_skips_an_installed_signature(tmp_path: Path):
    store = HarnessStore(tmp_path / "store")
    store.stage(Artifact(id="a1", type="rule",
                         origin={"signature": "tool_error:Bash:unicode-decode"},
                         payload="p", replay={}))
    store.accept("a1")
    sessions = [
        _error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 3),
        _error_session(tmp_path, "b", "Read", "Permission denied", 1),
    ]
    assert select_target(sessions, store).signature == "tool_error:Read:permission"


def test_select_skips_a_tombstoned_signature(tmp_path: Path):
    store = HarnessStore(tmp_path / "store")
    store.stage(Artifact(id="a1", type="rule",
                         origin={"signature": "tool_error:Bash:unicode-decode"},
                         payload="p", replay={}))
    store.reject("a1", wrong=True)
    sessions = [_error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 3)]
    assert select_target(sessions, store) is None


def test_select_returns_none_when_nothing_is_left(tmp_path: Path):
    assert select_target([], HarnessStore(tmp_path / "store")) is None


def test_a_failure_class_keeps_its_episodes(tmp_path: Path):
    sessions = [_error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 2)]
    assert len(rank_failures(sessions)[0].episodes) == 2
