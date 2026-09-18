import json
from pathlib import Path

from meta_harness.cc_history import (draft_tasks, failure_episodes, harness_report,
                                     is_artifact_project, load_sessions, parse_session,
                                     project_slug, write_history_view)


def _write_transcript(path: Path, records) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _assistant(tools=(), text="", sidechain=False):
    content = [{"type": "text", "text": text}] if text else []
    for index, name in enumerate(tools):
        content.append({"type": "tool_use", "id": f"t{index}", "name": name, "input": {"a": 1}})
    return {"type": "assistant", "message": {"content": content}, "isSidechain": sidechain,
            "cwd": "D:\\repo", "gitBranch": "main", "version": "2.1.0",
            "timestamp": "2026-09-18T10:00:00Z"}


def _results(ids_errors):
    content = [{"type": "tool_result", "tool_use_id": i, "is_error": e, "content": "out"}
               for i, e in ids_errors]
    return {"type": "user", "message": {"content": content}}


def _human(text):
    return {"type": "user", "message": {"content": text}}


# --- slug ------------------------------------------------------------------

def test_project_slug_uses_one_dash_per_non_alnum():
    # Claude Code writes D:\a\b as D--a-b: the colon and the separator each become a dash.
    assert project_slug("D:\\a\\b").endswith("D--a-b") or project_slug("/a/b").endswith("-a-b")


def test_artifact_projects_are_recognised():
    assert is_artifact_project("D--x--meta-harness-workspaces-mh-agent-abc")
    assert not is_artifact_project("D--Files-Projects-RealWork")


# --- parsing ---------------------------------------------------------------

def test_parses_tools_and_errors(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _human("do the thing"),
        _assistant(tools=["Read", "Edit"]),
        _results([("t0", False), ("t1", True)]),
    ])
    session = parse_session(path)
    assert session.tool_counts == {"Read": 1, "Edit": 1}
    assert session.error_count == 1
    assert session.cwd == "D:\\repo" and session.git_branch == "main"


def test_redaction_drops_assistant_text_by_default(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _assistant(text="secret internal reasoning", tools=["Read"]),
    ])
    assert parse_session(path).turns[0].text == ""
    assert "secret" in parse_session(path, include_text=True).turns[0].text


def test_subagent_and_workflow_counted_from_tool_calls(tmp_path: Path):
    # isSidechain is false in real transcripts; delegation shows up as a tool call.
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _assistant(tools=["Agent", "Agent", "Workflow", "Skill", "Read"]),
    ])
    session = parse_session(path)
    assert session.subagent_turns == 2
    assert session.workflow_calls == 1
    assert session.skill_calls == 1


# --- episodes --------------------------------------------------------------

def test_tool_error_episode(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _assistant(tools=["Bash"]), _results([("t0", True)]),
    ])
    episodes = failure_episodes(parse_session(path))
    assert [e.kind for e in episodes] == ["tool_error"]
    assert episodes[0].tools == ["Bash"]


def test_correction_detected_across_a_toolless_closing_turn(tmp_path: Path):
    # The human never replies to a tool call; their turn lands after a closing assistant
    # message with no tools. A one-turn lookback would miss every correction.
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _human("fix it"),
        _assistant(tools=["Edit"]),
        _results([("t0", False)]),
        _assistant(text="Done."),
        _human("no, that broke the build - revert"),
    ])
    episodes = failure_episodes(parse_session(path, include_text=True))
    kinds = [e.kind for e in episodes]
    assert "correction" in kinds
    correction = next(e for e in episodes if e.kind == "correction")
    assert "Edit" in correction.tools
    assert correction.matched


def test_correction_needs_preceding_work(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [_human("that's wrong")])
    assert failure_episodes(parse_session(path, include_text=True)) == []


def test_thrash_episode(tmp_path: Path):
    records = []
    for _ in range(8):
        records.append(_assistant(tools=["Bash"]))
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", records)
    assert any(e.kind == "thrash" for e in failure_episodes(parse_session(path)))


# --- report and view -------------------------------------------------------

def test_report_aggregates_read_write_ratio(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _assistant(tools=["Read", "Write", "Write", "Write", "Write"]),
    ])
    report = harness_report([parse_session(path)])
    assert report["read_to_write_ratio"] == 0.25
    assert report["tool_calls"] == 5


def test_view_redacts_user_text_but_keeps_the_signal(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _human("fix it"), _assistant(tools=["Edit"]), _results([("t0", False)]),
        _assistant(text="Done."), _human("no, that doesn't work - revert it"),
    ])
    sessions = [parse_session(path, include_text=True)]
    view = write_history_view(sessions, tmp_path / "view", include_text=False)
    episodes = [json.loads(l) for l in (view / "episodes.jsonl").read_text().splitlines()]
    correction = next(e for e in episodes if e["kind"] == "correction")
    assert correction["user_text"] == ""          # the private part is gone
    assert correction["matched"]                   # the signal is kept
    assert correction["tools"] == ["Edit"]
    assert not any("doesn't work" in l for l in (view / "episodes.jsonl").read_text().splitlines())


def test_view_keeps_transcript_when_asked(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [_human("hello there")])
    sessions = [parse_session(path, include_text=True)]
    view = write_history_view(sessions, tmp_path / "view", include_text=True)
    summary = json.loads((view / "sessions" / "s1.json").read_text())
    assert "transcript" in summary


def test_view_writes_report_and_readme(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [_assistant(tools=["Read"])])
    view = write_history_view([parse_session(path)], tmp_path / "view")
    assert json.loads((view / "report.json").read_text())["tool_calls"] == 1
    assert "episodes.jsonl" in (view / "README.md").read_text()


# --- drafts ----------------------------------------------------------------

def test_draft_tasks_leave_verification_to_a_human(tmp_path: Path):
    path = _write_transcript(tmp_path / "proj" / "s1.jsonl", [
        _human("fix it"), _assistant(tools=["Edit"]), _results([("t0", False)]),
        _assistant(text="Done."), _human("no, you broke the tests"),
    ])
    drafts = draft_tasks([parse_session(path, include_text=True)])
    assert drafts, "a correction episode should yield a draft"
    # History supplies the request that preceded the work...
    assert drafts[0]["instruction"] == "fix it"
    assert drafts[0]["_correction"]
    # ...but never the verification. That hole is the human's to fill.
    assert drafts[0]["test_files"] == {}
    assert "_todo" in drafts[0]


# --- loading ---------------------------------------------------------------

def test_load_sessions_skips_short_ones_and_artifacts(tmp_path: Path):
    home = tmp_path / "claude"
    _write_transcript(home / "projects" / "D--real" / "a.jsonl",
                      [_assistant(tools=["Read"]) for _ in range(6)])
    _write_transcript(home / "projects" / "D--x--mh-agent-zzz" / "b.jsonl",
                      [_assistant(tools=["Read"]) for _ in range(6)])
    _write_transcript(home / "projects" / "D--real" / "tiny.jsonl", [_assistant(tools=["Read"])])
    sessions = load_sessions(home=home, min_turns=4)
    assert [s.session_id for s in sessions] == ["a"]


# --- file history ----------------------------------------------------------

def _delta(tracking_path, backup_name, version=1):
    backup = {"backupFileName": backup_name, "version": version,
              "backupTime": "2026-09-18T10:00:00Z", "realParentDir": "D:\repo"}
    # Claude Code stores this field as a Python repr, not as JSON.
    return {"type": "file-history-delta", "trackingPath": tracking_path,
            "backup": repr(backup), "timestamp": "2026-09-18T10:00:00Z"}


def test_read_file_history_recovers_previous_contents(tmp_path: Path):
    home = tmp_path / "claude"
    _write_transcript(home / "projects" / "D--repo" / "sess.jsonl", [
        _delta("src/a.py", "hash1@v2", version=2),
        _delta("src/new.py", None, version=1),
    ])
    blob = home / "file-history" / "sess" / "hash1@v2"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_text(json.dumps("def a():\n    return 1\n"), encoding="utf-8")

    from meta_harness.cc_history import read_file_history

    versions = read_file_history("sess", home=home)
    by_path = {v.tracking_path: v for v in versions}
    assert by_path["src/a.py"].previous_content == "def a():\n    return 1\n"
    # No backup file means the agent created it: there was nothing there before.
    assert by_path["src/new.py"].previous_content is None


def test_guess_test_command_from_repo_markers(tmp_path: Path):
    from meta_harness.cc_history import guess_test_command

    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    assert guess_test_command(str(tmp_path)) == "python -m pytest -q"
    assert guess_test_command(str(tmp_path / "missing")) == ""


def test_drafts_seed_files_and_leave_verification_empty(tmp_path: Path):
    home = tmp_path / "claude"
    _write_transcript(home / "projects" / "D--repo" / "sess.jsonl", [
        _human("make the parser handle empty input"),
        _assistant(tools=["Edit"]),
        _results([("t0", False)]),
        _assistant(text="Done."),
        _human("no, that broke it - revert"),
        _delta("src/parser.py", "h1@v2", version=2),
    ])
    blob = home / "file-history" / "sess" / "h1@v2"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_text(json.dumps("old parser body"), encoding="utf-8")

    sessions = load_sessions(home=home, include_text=True, min_turns=3)
    drafts = draft_tasks(sessions, home=home)
    assert drafts
    draft = drafts[0]
    assert draft["files"]["src/parser.py"] == "old parser body"
    assert draft["instruction"]                 # the ask that preceded the work
    assert draft["test_files"] == {}            # history cannot supply the verification
    assert "_todo" in draft


def test_drafts_cap_the_number_of_seeded_files(tmp_path: Path):
    from meta_harness.cc_history import MAX_DRAFT_FILES

    home = tmp_path / "claude"
    records = [_human("do a thing"), _assistant(tools=["Edit"]), _results([("t0", False)]),
               _assistant(text="Done."), _human("that's wrong, revert")]
    for index in range(MAX_DRAFT_FILES + 4):
        records.append(_delta(f"src/f{index}.py", f"h{index}@v2", version=2))
    _write_transcript(home / "projects" / "D--repo" / "sess.jsonl", records)
    for index in range(MAX_DRAFT_FILES + 4):
        blob = home / "file-history" / "sess" / f"h{index}@v2"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps("x" * (100 + index)), encoding="utf-8")

    drafts = draft_tasks(load_sessions(home=home, include_text=True, min_turns=3), home=home)
    assert len(drafts[0]["files"]) == MAX_DRAFT_FILES
    assert drafts[0]["_files_touched_in_session"] == MAX_DRAFT_FILES + 4


# --- comparing windows -----------------------------------------------------

def test_sessions_between_filters_by_timestamp(tmp_path: Path):
    from meta_harness.cc_history import sessions_between

    def at(stamp):
        session = parse_session(_write_transcript(
            tmp_path / "proj" / f"{stamp[:10]}.jsonl",
            [{"type": "assistant", "message": {"content": []}, "timestamp": stamp}]))
        return session

    early, late = at("2026-09-01T00:00:00Z"), at("2026-09-20T00:00:00Z")
    both = [early, late]
    assert len(sessions_between(both, before="2026-09-10")) == 1
    assert len(sessions_between(both, since="2026-09-10")) == 1
    assert len(sessions_between(both)) == 2


def test_compare_flags_direction_only_where_it_is_defensible():
    from meta_harness.cc_history import compare_reports

    before = {"sessions": 10, "tool_calls": 1000, "error_rate": 0.05,
              "read_to_write_ratio": 0.4, "subagent_calls": 50,
              "episode_kinds": {"tool_error": 100, "thrash": 20, "correction": 10}}
    after = {"sessions": 10, "tool_calls": 900, "error_rate": 0.03,
             "read_to_write_ratio": 0.9, "subagent_calls": 40,
             "episode_kinds": {"tool_error": 60, "thrash": 8, "correction": 4}}
    rows = {r["metric"]: r for r in compare_reports(before, after)["metrics"]}

    assert rows["error_rate"]["improved"] is True
    assert rows["read_to_write_ratio"]["improved"] is True
    assert rows["correction_per_session"]["improved"] is True
    # Volume moves with whatever work happened in the window; claiming a direction there
    # would be inventing a verdict the data cannot support.
    assert rows["tool_calls"]["improved"] is None
    assert rows["subagent_calls"]["improved"] is None


def test_compare_normalises_episodes_per_session():
    from meta_harness.cc_history import compare_reports

    # Twice the sessions and twice the errors is the same rate, not a regression.
    before = {"sessions": 5, "episode_kinds": {"tool_error": 50}}
    after = {"sessions": 10, "episode_kinds": {"tool_error": 100}}
    rows = {r["metric"]: r for r in compare_reports(before, after)["metrics"]}
    assert rows["tool_error_per_session"]["delta"] == 0.0


def test_compare_carries_its_caveat():
    from meta_harness.cc_history import compare_reports

    assert "Observational" in compare_reports({"sessions": 1}, {"sessions": 1})["caveat"]
