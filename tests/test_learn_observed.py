import json
from pathlib import Path

from meta_harness.cc_history import Session, ToolCall, Turn
from meta_harness.harness_store import HarnessStore
from meta_harness.learn import FailureClass, merge_failures, observed_failures, select_target


def _write(home: Path, session_id: str, records):
    home.mkdir(parents=True, exist_ok=True)
    lines = "\n".join(json.dumps(r) for r in records)
    (home / f"observed-{session_id}.jsonl").write_text(lines + "\n", encoding="utf-8")


def _mined_session(session_id: str, tool: str, text: str, occurrences: int) -> Session:
    """A session whose `failure_episodes()` yields `occurrences` tool_error episodes of one
    signature, so `rank_failures` sees a mined failure with a controllable count."""
    turns = [
        Turn(role="assistant", index=i,
             tools=[ToolCall(name=tool, is_error=True, result_excerpt=text)])
        for i in range(occurrences)
    ]
    return Session(session_id=session_id, project="proj", path="", turns=turns)


def test_reads_observed_failures_across_per_session_files(tmp_path: Path):
    _write(tmp_path, "session-a", [
        {"kind": "tool_error", "tool": "Bash", "text": "UnicodeDecodeError: charmap"},
    ])
    _write(tmp_path, "session-b", [
        {"kind": "tool_error", "tool": "Bash", "text": "UnicodeDecodeError: charmap"},
    ])
    failures = observed_failures(tmp_path)
    assert failures[0].signature == "tool_error:Bash:unicode-decode"
    assert failures[0].count == 2


def test_uses_precomputed_cause_field_when_present(tmp_path: Path):
    # The real hook (hooks/harness.ts observe()) writes a "cause" field alongside "text"; the
    # reader must prefer it rather than recompute from raw text, since the hook's cause() and
    # Python's _cause() are only guaranteed to agree, not identical in every edge case.
    _write(tmp_path, "session-a", [
        {"kind": "tool_error", "tool": "Bash", "cause": "permission", "text": "something else entirely"},
    ])
    failures = observed_failures(tmp_path)
    assert failures[0].signature == "tool_error:Bash:permission"


def test_repeat_records_become_thrash_classes(tmp_path: Path):
    _write(tmp_path, "session-a", [{"kind": "repeat", "tool": "Edit"}])
    assert observed_failures(tmp_path)[0].signature == "thrash:Edit"


def test_a_missing_log_dir_is_not_an_error(tmp_path: Path):
    assert observed_failures(tmp_path / "nothing") == []


def test_no_observed_files_is_not_an_error(tmp_path: Path):
    tmp_path.mkdir(exist_ok=True)
    assert observed_failures(tmp_path) == []


def test_corrupt_line_is_skipped_not_fatal(tmp_path: Path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "observed-session-a.jsonl").write_text(
        '{"kind": "tool_error", "tool": "Bash", "text": "Permission denied"}\n'
        '{not json at all\n',
        encoding="utf-8",
    )
    failures = observed_failures(tmp_path)
    assert failures[0].signature == "tool_error:Bash:permission"
    assert failures[0].count == 1


def test_partial_trailing_line_is_tolerated(tmp_path: Path):
    # A session may still be mid-write when the reader runs; the last line can be truncated.
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "observed-session-a.jsonl").write_text(
        '{"kind": "tool_error", "tool": "Bash", "text": "Permission denied"}\n'
        '{"kind": "tool_error", "tool": "Edit", "text"',
        encoding="utf-8",
    )
    failures = observed_failures(tmp_path)
    assert len(failures) == 1
    assert failures[0].signature == "tool_error:Bash:permission"


def test_observed_failures_are_considered_when_there_is_no_mined_history(tmp_path: Path):
    home = tmp_path / "home"
    _write(home, "session-a", [{"kind": "tool_error", "tool": "Bash", "text": "Permission denied"}])
    store = HarnessStore(tmp_path / "store")
    # With no mined sessions at all, the only evidence is the live observation.
    assert select_target([], store, home=home).signature == "tool_error:Bash:permission"


def test_merge_sums_counts_for_a_signature_seen_in_both_sources():
    mined = [FailureClass(signature="tool_error:Bash:permission", kind="tool_error", tool="Bash",
                          count=3)]
    observed = [FailureClass(signature="tool_error:Bash:permission", kind="tool_error",
                             tool="Bash", count=1)]
    merged = merge_failures(observed, mined)
    assert merged[0].signature == "tool_error:Bash:permission"
    assert merged[0].count == 4


def test_a_high_count_mined_failure_beats_a_single_observed_failure(tmp_path: Path):
    # This is the regression the source-order concatenation shipped: a single live observation
    # of an unrelated signature must not preempt a mined signature seen 50 times.
    home = tmp_path / "home"
    _write(home, "session-a", [{"kind": "tool_error", "tool": "Edit", "text": "old_string"}])
    sessions = [_mined_session("s1", "Bash", "Permission denied", 50)]
    store = HarnessStore(tmp_path / "store")
    target = select_target(sessions, store, home=home)
    assert target.signature == "tool_error:Bash:permission"
    assert target.count == 50


def test_a_signature_seen_in_both_sources_outranks_either_alone(tmp_path: Path):
    home = tmp_path / "home"
    # Observed once, adding weight to the mined signature below.
    _write(home, "session-a", [{"kind": "tool_error", "tool": "Bash", "text": "Permission denied"}])
    sessions = [
        # X: mined 3 times, observed once -> merged count 4.
        _mined_session("s1", "Bash", "Permission denied", 3),
        # Z: mined 3 times only, never observed -> stays at 3.
        _mined_session("s2", "Edit", "old_string", 3),
    ]
    store = HarnessStore(tmp_path / "store")
    target = select_target(sessions, store, home=home)
    assert target.signature == "tool_error:Bash:permission"
    assert target.count == 4


def test_covered_signatures_are_still_skipped(tmp_path: Path):
    from meta_harness.harness_store import Artifact

    home = tmp_path / "home"
    _write(home, "session-a", [{"kind": "tool_error", "tool": "Bash", "text": "Permission denied"}])
    store = HarnessStore(tmp_path / "store")
    store.stage(Artifact(id="a1", type="rule",
                         origin={"signature": "tool_error:Bash:permission"},
                         payload="p", replay={}))
    store.accept("a1")
    assert select_target([], store, home=home) is None


def test_select_target_default_home_call_site_still_works(tmp_path: Path, monkeypatch):
    # meta_harness/__main__.py calls select_target(sessions, store) with only two positional
    # arguments; the third parameter must default so that call site keeps working.
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path / "empty-home"))
    store = HarnessStore(tmp_path / "store2")
    assert select_target([], store) is None
