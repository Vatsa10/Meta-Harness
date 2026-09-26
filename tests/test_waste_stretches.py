import json
from pathlib import Path

from meta_harness.waste import Stretch, iter_stretches


def write_transcript(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "session.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def assistant(*tools):
    content = [{"type": "tool_use", "id": f"t{i}", "name": n, "input": inp}
               for i, (n, inp) in enumerate(tools)]
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


def user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def test_a_run_of_tool_calls_between_user_messages_is_one_stretch(tmp_path):
    path = write_transcript(tmp_path, [
        user("add a flag"),
        assistant(("Read", {"file_path": "a.py"}), ("Edit", {"file_path": "a.py"})),
        assistant(("Bash", {"command": "pytest"})),
        user("thanks"),
    ])
    stretches = list(iter_stretches(path, min_calls=3))
    assert len(stretches) == 1
    assert stretches[0].calls == ["Read", "Edit", "Bash"]
    assert stretches[0].ended_by == "user"


def test_a_correction_is_labelled_and_carries_its_text(tmp_path):
    path = write_transcript(tmp_path, [
        user("add a flag"),
        assistant(("Read", {"file_path": "a.py"}), ("Edit", {"file_path": "a.py"}),
                  ("Bash", {"command": "pytest"})),
        user("no, that's not what I asked for"),
    ])
    stretch = next(iter_stretches(path, min_calls=3))
    assert stretch.ended_by == "correction"
    assert "not what I asked" in stretch.correction_text


def test_short_stretches_are_not_yielded(tmp_path):
    path = write_transcript(tmp_path, [
        user("hi"), assistant(("Read", {"file_path": "a.py"})), user("no, wrong"),
    ])
    assert list(iter_stretches(path, min_calls=3)) == []


def test_paths_are_collected_from_tool_input(tmp_path):
    path = write_transcript(tmp_path, [
        user("go"),
        assistant(("Read", {"file_path": "src/a.py"}), ("Edit", {"file_path": "src/b.py"}),
                  ("Bash", {"command": "ls src"})),
        user("ok"),
    ])
    stretch = next(iter_stretches(path, min_calls=3))
    assert "src/a.py" in stretch.paths and "src/b.py" in stretch.paths


def test_one_malformed_line_does_not_abort_the_file(tmp_path):
    path = tmp_path / "session.jsonl"
    good = [user("go"),
            assistant(("Read", {"file_path": "a"}), ("Edit", {"file_path": "b"}),
                      ("Bash", {"command": "c"})),
            user("no, wrong")]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(good[0]) + "\n")
        handle.write('{"type": "assistant", "message": {"content": [trunc\n')  # truncated write
        handle.write(json.dumps(good[1]) + "\n")
        handle.write(json.dumps(good[2]) + "\n")
    stretch = next(iter_stretches(path, min_calls=3))
    assert stretch.ended_by == "correction"


def test_a_missing_file_yields_nothing_rather_than_raising(tmp_path):
    assert list(iter_stretches(tmp_path / "absent.jsonl")) == []
