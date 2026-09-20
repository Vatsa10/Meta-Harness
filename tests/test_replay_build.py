import json
from pathlib import Path

from meta_harness.cc_history import FailureEpisode, load_sessions, parse_session
from meta_harness.replay import build_replay


def _transcript(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _assistant(tools=(), text=""):
    content = [{"type": "text", "text": text}] if text else []
    for index, name in enumerate(tools):
        content.append({"type": "tool_use", "id": f"t{index}", "name": name, "input": {}})
    return {"type": "assistant", "message": {"content": content},
            "cwd": "D:\\repo", "timestamp": "2026-09-20T10:00:00Z"}


def _human(text):
    return {"type": "user", "message": {"content": text}}


def _session(tmp_path: Path):
    path = _transcript(tmp_path / "proj" / "s1.jsonl", [
        _human("make the parser handle empty input"),
        _assistant(tools=["Bash"]),
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t0", "is_error": True,
             "content": "UnicodeDecodeError: charmap"}]}},
    ])
    return parse_session(path, include_text=True)


def _episode():
    return FailureEpisode(session_id="s1", project="proj", kind="tool_error", turn_index=2,
                          tools=["Bash"], assistant_text="UnicodeDecodeError: charmap")


def test_replay_carries_instruction_and_expectation(tmp_path: Path):
    replay = build_replay(_episode(), _session(tmp_path), home=tmp_path / "claude")
    assert replay is not None
    assert "parser" in replay["instruction"]
    assert replay["expect"] == {"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}}
    assert replay["_origin"]["session"] == "s1"


def test_replay_without_a_request_is_not_replayable(tmp_path: Path):
    path = _transcript(tmp_path / "proj" / "s2.jsonl", [_assistant(tools=["Bash"])])
    assert build_replay(_episode(), parse_session(path, include_text=True),
                        home=tmp_path / "claude") is None


def test_replay_seeds_files_from_file_history(tmp_path: Path):
    home = tmp_path / "claude"
    _transcript(home / "projects" / "proj" / "s1.jsonl", [
        {"type": "file-history-delta", "trackingPath": "src/a.py",
         "backup": repr({"backupFileName": "h1@v2", "version": 2}),
         "timestamp": "2026-09-20T10:00:00Z"},
    ])
    blob = home / "file-history" / "s1" / "h1@v2"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_text(json.dumps("original body"), encoding="utf-8")

    replay = build_replay(_episode(), _session(tmp_path), home=home)
    assert replay["files"]["src/a.py"] == "original body"


def test_replay_caps_the_number_of_seeded_files(tmp_path: Path):
    home = tmp_path / "claude"
    deltas = []
    for index in range(8):
        deltas.append({"type": "file-history-delta", "trackingPath": f"src/f{index}.py",
                       "backup": repr({"backupFileName": f"h{index}@v2", "version": 2}),
                       "timestamp": "2026-09-20T10:00:00Z"})
        blob = home / "file-history" / "s1" / f"h{index}@v2"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps("x" * (50 + index)), encoding="utf-8")
    _transcript(home / "projects" / "proj" / "s1.jsonl", deltas)

    replay = build_replay(_episode(), _session(tmp_path), home=home, max_files=3)
    assert len(replay["files"]) == 3


def test_correction_episodes_are_not_mechanically_replayable(tmp_path: Path):
    episode = FailureEpisode(session_id="s1", project="proj", kind="correction",
                             turn_index=2, tools=["Edit"], user_text="that's wrong")
    # No mechanical expectation exists, so there is nothing to verify without a human.
    assert build_replay(episode, _session(tmp_path), home=tmp_path / "claude") is None
