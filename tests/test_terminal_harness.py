import json
from pathlib import Path

import pytest

from meta_harness.core import TraceRecorder
from meta_harness.sandboxes import ExecResult
from meta_harness.scripted import ScriptedModel
from meta_harness.terminal import (CONTINUE_LINE, ELIDED_STUB, FileTools, PLAN_REMINDER, EditError,
                                   StagedCompactor, StuckDetector, apply_edit, cap_output,
                                   label_command, phase_summary)
from meta_harness.terminal_harness import (DockerPolicy, KiraHarness, TerminalHarness,
                                           Terminus2Harness)


class FakeSandbox:
    """Records exec calls; files live in a dict and move through get/put like a real sandbox."""

    def __init__(self, files=None, respond=None):
        self.files = dict(files or {})
        self.commands = []
        self.respond = respond or (lambda command: (0, "ok"))

    def exec(self, command, timeout, cwd=None, idle=None):
        self.commands.append(command)
        code, out = self.respond(command)
        return ExecResult(code, out, "", False)

    def put(self, local, remote):
        self.files[remote] = Path(local).read_bytes().decode("utf-8")

    def get(self, remote, local):
        if remote not in self.files:
            raise FileNotFoundError(remote)
        Path(local).write_bytes(self.files[remote].encode("utf-8"))

    def close(self):
        pass


def script(tmp_path: Path, entries, name="s.jsonl") -> ScriptedModel:
    path = tmp_path / name
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    return ScriptedModel(path)


def turn(commands=(), complete=False, **extra):
    return {"text": json.dumps({"analysis": "a", "plan": "p", "commands": list(commands),
                                "task_complete": complete, **extra})}


class Recorder:
    def __init__(self, model):
        self.model, self.prompts = model, []

    def __call__(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.model(prompt)


# --------------------------------------------------------------------------- contract
def test_docker_policy_wraps_commands():
    wrapped = DockerPolicy(container="c1", workdir="/app").wrap("ls -la")
    assert wrapped == ["docker", "exec", "-w", "/app", "c1", "bash", "-lc", "ls -la"]


def test_harness_refuses_to_run_without_a_container(monkeypatch):
    monkeypatch.delenv("META_HARNESS_ALLOW_LOCAL_SHELL", raising=False)
    with pytest.raises(PermissionError, match="allow_local_shell"):
        TerminalHarness().run({"instruction": "echo hi"}, lambda p, **_: "{}", None)


def test_validation_probe_does_not_touch_a_shell():
    assert Terminus2Harness().run({"input": "validation"}, lambda p, **_: "{}", None)["output"] == ""


# --------------------------------------------------------------------------- terminus-2 loop
def test_terminus2_runs_keystrokes_and_needs_a_confirmation_turn(tmp_path: Path):
    box = FakeSandbox()
    model = Recorder(script(tmp_path, [turn([{"keystrokes": "echo ready\n", "duration": 0.1}]),
                                       turn(complete=True), turn(complete=True)]))
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        result = Terminus2Harness().run({"instruction": "say ready", "sandbox": box}, model, trace)
    assert box.commands == ["echo ready"]
    assert result["stop_reason"] == "done" and result["steps"] == 3
    assert "Are you sure" in model.prompts[2] and "Are you sure" not in model.prompts[1]
    events = [json.loads(l)["event"] for l in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()]
    assert "terminal_step" in events and "terminal_end" in events


def test_terminus2_caps_an_observation_at_24k_chars(tmp_path: Path):
    box = FakeSandbox(respond=lambda c: (0, "x" * 100_000))
    model = Recorder(script(tmp_path, [turn([{"keystrokes": "cat big\n"}]), turn(complete=True),
                                       turn(complete=True)]))
    Terminus2Harness().run({"instruction": "t", "sandbox": box}, model, None)
    observation = model.prompts[1].split("## user\n")[-1]
    assert len(observation) <= 24_000 + 400  # cap + the confirm text
    assert "output truncated" in observation
    assert len(cap_output("y" * 100_000)) <= 24_000
    assert Terminus2Harness().max_steps == 300


def test_overflow_without_compaction_is_a_counted_stop_reason(tmp_path: Path):
    box = FakeSandbox(respond=lambda c: (0, "z" * 4000))
    model = script(tmp_path, [turn([{"keystrokes": "cat a\n"}])] * 5)
    result = Terminus2Harness(compaction=False, context_window=1500).run(
        {"instruction": "t", "sandbox": box}, model, None)
    assert result["stop_reason"] == "overflow"


def test_stuck_loop_stops_with_stuck(tmp_path: Path):
    box = FakeSandbox(respond=lambda c: (1, "boom"))
    model = Recorder(script(tmp_path, [turn([{"keystrokes": "make\n"}])] * 10))
    result = Terminus2Harness().run({"instruction": "t", "sandbox": box}, model, None)
    assert result["stop_reason"] == "stuck" and len(box.commands) == 8
    assert sum("keeps failing" in p.split("## user")[-1] for p in model.prompts) == 1


# --------------------------------------------------------------------------- stuck detector
def test_stuck_detector_reminds_once_at_5_and_stops_at_8():
    detector = StuckDetector()
    notes = [detector.observe("bash", {"command": "make"}, failed=True) for _ in range(8)]
    assert [i for i, n in enumerate(notes, 1) if n] == [5] and "keeps failing" in notes[4]
    assert detector.stuck


def test_stuck_detector_streak_broken_by_a_different_argument():
    detector = StuckDetector()
    for i in range(1, 9):
        detector.observe("read", {"path": "f", "offset": 10 if i == 4 else 0}, failed=True)
    assert not detector.stuck


def test_stuck_detector_ignores_denials_and_says_repeating_for_successes():
    detector = StuckDetector()
    for _ in range(10):
        detector.observe("bash", {"command": "rm -rf /"}, failed=True, denied=True)
    assert not detector.stuck
    notes = [detector.observe("bash", {"command": "ls"}, failed=False) for _ in range(5)]
    assert "same call 5 times" in notes[4] or "5 times" in notes[4]


# --------------------------------------------------------------------------- edit tool
def test_exact_edit():
    assert apply_edit("a = 1\nb = 2\n", "b = 2", "b = 3") == ("a = 1\nb = 3\n", "exact")


def test_fuzzy_edit_across_changed_indentation():
    text = "def f():\n    if x:\n        return 1\n    return 2\n"
    new_text, mode = apply_edit(text, "if x:\n    return 1", "if x:\n    return 10")
    assert mode == "whitespace"
    assert new_text == "def f():\n    if x:\n        return 10\n    return 2\n"


def test_fuzzy_edit_by_similarity():
    text = "def total(items):\n    return sum(item.price for item in items)\n"
    new_text, mode = apply_edit(text, "def total(items):\n    return sum(item.price for item in item)",
                                "def total(items):\n    return 0")
    assert mode.startswith("fuzzy") and "return 0" in new_text


def test_ambiguous_fuzzy_match_is_refused():
    block = "for i in range(10):\n    print(i)\n"
    with pytest.raises(EditError, match="ambiguous|places"):
        apply_edit(block + "x = 1\n" + block, "for i in range(10):\n    print(i )", "pass")
    near = "value = compute(alpha, beta)\nother = 1\nvalue = compute(alpha, betA)\n"
    with pytest.raises(EditError, match="ambiguous"):
        apply_edit(near, "value = compute(alpha, betx)", "value = 0")


def test_edit_requires_a_full_read_and_an_unchanged_hash():
    box = FakeSandbox({"a.txt": "one\n"})
    tools = FileTools(box)
    with pytest.raises(EditError, match="read it first"):
        tools.edit("a.txt", "one", "two")
    tools.read("a.txt")
    box.files["a.txt"] = "one\nchanged elsewhere\n"
    with pytest.raises(EditError, match="changed since"):
        tools.edit("a.txt", "one", "two")
    tools.read("a.txt")
    assert "exact" in tools.edit("a.txt", "one", "two") and box.files["a.txt"].startswith("two")


def test_edit_errors_are_observations_and_py_compile_is_reported(tmp_path: Path):
    box = FakeSandbox({"s.py": "x = 1\n"})
    model = Recorder(script(tmp_path, [
        turn([{"edit": {"path": "s.py", "old": "x = 1", "new": "x = 2"}}]),
        turn([{"read": {"path": "s.py"}}, {"edit": {"path": "s.py", "old": "x = 1", "new": "x = (("}}]),
        turn(complete=True), turn(complete=True)]))
    result = Terminus2Harness().run({"instruction": "t", "sandbox": box}, model, None)
    assert result["stop_reason"] == "done"
    assert "read it first" in model.prompts[1]
    assert "[py_compile]" in model.prompts[2] and "[py_compile] OK" not in model.prompts[2]


# --------------------------------------------------------------------------- staged compaction
def _history(n_turns=6, bulky=3000):
    msgs = [{"role": "user", "content": "TASK: fix the bug"}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "text": f"step {i}", "tool_calls": []})
        msgs.append({"role": "user", "kind": "observation", "content": f"out {i}\n" + "o" * bulky})
    return msgs


def test_compaction_below_b1_does_nothing():
    calls = []
    out, info = StagedCompactor(100_000).compact(_history(), lambda p: calls.append(p) or "S")
    assert out == _history() and info.summary_calls == 0 and not calls


def test_compaction_crossing_b1_only_elides_middle_outputs():
    msgs = _history()
    tokens = sum(len(m.get("content", m.get("text", ""))) for m in msgs) // 4
    calls = []
    compactor = StagedCompactor(int(tokens / 0.7), recent_fraction=0.05)
    out, info = compactor.compact(msgs, lambda p: calls.append(p) or "S")
    assert not calls and info.elided > 0
    assert out[0] == msgs[0] and out[-4:] == msgs[-4:]  # preamble and last 2 turns byte-identical
    assert any(str(m.get("content", "")).startswith(ELIDED_STUB.split(":")[0]) for m in out)


def test_compaction_crossing_b2_summarises_once_with_previous_summary():
    msgs = _history(bulky=300)
    msgs.insert(1, {"role": "user", "kind": "summary",
                    "content": f"[Summary of earlier work]\nPREVIOUS-SUMMARY\n\n{CONTINUE_LINE}"})
    tokens = sum(len(m.get("content", m.get("text", ""))) for m in msgs) // 4
    calls = []
    compactor = StagedCompactor(int(tokens / 0.9), recent_fraction=0.05)
    out, info = compactor.compact(msgs, lambda p: calls.append(p) or "NEW-SUMMARY")
    assert len(calls) == 1 and "PREVIOUS-SUMMARY" in calls[0] and "## Errors & fixes" in calls[0]
    assert out[0] == msgs[0] and out[-4:] == msgs[-4:]
    assert out[1]["content"].endswith(CONTINUE_LINE) and "NEW-SUMMARY" in out[1]["content"]
    assert out[2]["role"] == "assistant"  # cut at a turn boundary


def test_scripted_model_receives_the_summary_request(tmp_path: Path):
    box = FakeSandbox(respond=lambda c: (0, "q" * 300))
    path = tmp_path / "m.jsonl"
    entries = [{"match": "compacting the history", "text": "SUMMARY-TEXT"}]
    entries += [turn([{"keystrokes": f"cat f{i}\n"}]) for i in range(30)] + [turn(complete=True)] * 2
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    model = Recorder(ScriptedModel(path))
    result = Terminus2Harness(context_window=3000).run({"instruction": "t", "sandbox": box}, model, None)
    asks = [p for p in model.prompts if "compacting the history" in p]
    assert asks and result["stop_reason"] == "done"
    assert any("SUMMARY-TEXT" in p and CONTINUE_LINE in p for p in model.prompts)


# --------------------------------------------------------------------------- phases / toggles
def test_phase_labels_on_a_fixture_trace():
    commands = ["ls -la /app", "cat solve.py", "cat > /app/solve.py << 'EOF'\nprint(1)\nEOF",
                "sed -i 's/a/b/' x.py", "python -m pytest -q tests", "cd /app && python solve.py"]
    labels = [label_command(c) for c in commands]
    assert labels == ["understand", "understand", "write", "write", "verify", "execute"]
    assert phase_summary(["understand", "execute"])["ended_without_writing_code"]
    assert phase_summary(["write", "understand"])["ended_in_understand"]


def test_planning_injects_the_plan_without_storing_it(tmp_path: Path):
    box = FakeSandbox()
    model = Recorder(script(tmp_path, [
        turn([{"keystrokes": "ls\n"}]),
        turn([], update_plan=[{"step": "inspect", "status": "completed"},
                              {"step": "UNIQUE-STEP", "status": "in_progress"}]),
        turn([{"keystrokes": "ls\n"}]), turn(complete=True), turn(complete=True)]))
    harness = Terminus2Harness(planning=True)
    result = harness.run({"instruction": "t", "sandbox": box}, model, None)
    assert result["stop_reason"] == "done"
    assert PLAN_REMINDER in model.prompts[0] and PLAN_REMINDER in model.prompts[1]
    assert PLAN_REMINDER not in model.prompts[2]
    assert all(p.count("UNIQUE-STEP") <= 2 for p in model.prompts)  # once in the reply, once injected
    assert model.prompts[3].count("[>] UNIQUE-STEP") == 1


def test_bash_only_rejects_file_tools(tmp_path: Path):
    box = FakeSandbox({"a": "x"})
    model = Recorder(script(tmp_path, [turn([{"read": {"path": "a"}}]), turn(complete=True),
                                       turn(complete=True)]))
    Terminus2Harness(bash_only=True).run({"instruction": "t", "sandbox": box}, model, None)
    assert "disabled (bash_only)" in model.prompts[1]


# --------------------------------------------------------------------------- KIRA
def test_kira_tool_calling_snapshot_and_checklist(tmp_path: Path):
    box = FakeSandbox(respond=lambda c: (0, "SNAP" if "Environment" in c or "pwd" in c else "ok"))
    model = script(tmp_path, [
        {"tool_calls": [{"name": "bash", "arguments": {"command": "echo hi"}}]},
        {"tool_calls": [{"name": "task_complete", "arguments": {"summary": "did it"}}]},
        {"tool_calls": [{"name": "task_complete", "arguments": {"summary": "did it"}}]},
    ])
    seen = []
    original = model.chat

    def chat(system, messages, tools, max_tokens=4096):
        seen.append((system, [dict(m) for m in messages], [t.name for t in tools]))
        return original(system, messages, tools, max_tokens)

    model.chat = chat
    result = KiraHarness().run({"instruction": "t", "sandbox": box}, model, None)
    assert result["stop_reason"] == "done" and result["output"] == "did it"
    assert "pwd" in box.commands[0] and "SNAP" in seen[0][1][0]["content"]
    assert "echo hi" in box.commands
    assert any("completion checklist" in str(m.get("content")) for m in seen[2][1])
    assert {"bash", "edit", "read_file", "task_complete"} <= set(seen[0][2])


def test_terminal_agent_refuses_without_a_container(monkeypatch):
    from meta_harness.terminal_harness import TerminalAgent
    monkeypatch.delenv("META_HARNESS_ALLOW_LOCAL_SHELL", raising=False)
    with pytest.raises(PermissionError, match="allow_local_shell"):
        TerminalAgent(lambda p: '{"command": "echo hi"}').run({"instruction": "x"})
    agent = TerminalAgent(lambda p: '{"done": true, "answer": "ok"}', allow_local_shell=True)
    assert agent.run({"instruction": "x"}).completed


def test_kira_bash_only_completion_is_not_reported_disabled(tmp_path: Path):
    box = FakeSandbox()
    model = script(tmp_path, [
        {"tool_calls": [{"name": "task_complete", "arguments": {"summary": "s"}}]},
        {"tool_calls": [{"name": "task_complete", "arguments": {"summary": "s"}}]},
    ])
    seen = []
    original = model.chat

    def chat(system, messages, tools, max_tokens=4096):
        seen.append([dict(m) for m in messages])
        return original(system, messages, tools, max_tokens)

    model.chat = chat
    result = KiraHarness(bash_only=True).run({"instruction": "t", "sandbox": box}, model, None)
    assert result["stop_reason"] == "done"
    assert not any("disabled" in str(m.get("content")) for m in seen[1])
