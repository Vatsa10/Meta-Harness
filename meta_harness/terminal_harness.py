"""Terminus-2 and Terminus-KIRA agent loops as searchable harnesses (paper section 4.3, App. B.3).

Harness contract for the terminal domain: `run(task, model, trace)` receives `task["container"]`,
the id of a container the DOMAIN started (never the harness), and returns
`{"output": summary, "stop_reason": ...}` plus phase statistics. Scoring happens afterwards,
inside the same container, in `metrics.terminal_metric`.

Terminus-2 (App. B.3): the model answers in JSON
    {"analysis": str, "plan": str,
     "commands": [{"keystrokes": "ls -la\\n", "duration": 0.1}, ...],
     "task_complete": bool}
Keystrokes are executed tmux-style, one batch per turn, through the sandbox (`Container.exec`);
every observation is capped at 30KB; the first `task_complete: true` triggers an explicit
confirmation turn and only a second consecutive `true` ends the episode. Our seeds add
`{"read": {"path"}}`, `{"edit": {"path", "old", "new"}}`, `{"write": {"path", "content"}}`
command items and (with `planning`) a top-level `"update_plan": [...]`.

Terminus-KIRA adds native tool calling through `model.chat` when the model offers it (JSON
protocol otherwise), a completion checklist before accepting `task_complete`, and an environment
snapshot collected from inside the container at the start.

Both seeds share: stuck detection (reminder at 5, stop at 8), staged compaction (B1/B2 as class
constants the proposer may tune), the fuzzy `edit` tool with read-before-write hashing and
`py_compile` diagnostics, rule-based phase labels, and normalised `stop_reason` in
{done, step_limit, overflow, stuck, timeout, error}. Toggles: `planning` (update_plan tool, plan
injected per turn and never stored in history) and `bash_only` (only bash + task_complete).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .chat_types import ToolSpec
from .sandboxes import TempDirSandbox
from .terminal import (B1, B2, MIN_RECENT_TURNS, OBSERVATION_CAP_BYTES, RECENT_FRACTION, Container,
                       DockerPolicy, FileTools, Plan, ShellPolicy, StagedCompactor, StuckDetector,
                       TerminalResult, cap_output, label_action, message_text, message_tokens,
                       phase_summary, run_in_sandbox)


def _emit(trace: Any, name: str, payload: Any) -> None:
    if trace is not None and hasattr(trace, "event"):
        trace.event(name, payload)


@dataclass
class Observation:
    ok: bool
    text: str
    denied: bool = False


class ToolBox:
    """Executes agent actions in a sandbox. Errors come back as observations, never exceptions."""

    def __init__(self, sandbox: Any, workdir: str | None, policy: ShellPolicy,
                 cap: int = OBSERVATION_CAP_BYTES):
        self.sandbox = sandbox
        self.workdir = workdir
        self.policy = policy
        self.cap = cap
        self.files = FileTools(sandbox, workdir if _is_docker(sandbox) else None)

    def bash(self, command: str, timeout: float | None = None) -> Observation:
        try:
            self.policy.check(command)
        except PermissionError as exc:
            return Observation(False, f"blocked: {exc}", denied=True)
        code, out = run_in_sandbox(self.sandbox, command, float(timeout or self.policy.timeout_seconds))
        return Observation(code == 0, cap_output(f"[exit {code}]\n{out}", self.cap))

    def keystrokes(self, keys: str, timeout: float | None = None) -> Observation:
        if re.fullmatch(r"\s*C-[a-z]\s*", keys or ""):
            return Observation(True, f"[sent {keys.strip()}: commands run to completion in exec mode; "
                                     "nothing is in the foreground]")
        command = keys.rstrip("\n")
        if not command.strip():
            return Observation(True, "[no keystrokes]")
        return self.bash(command, timeout)

    def dispatch(self, tool: str, args: Mapping[str, Any]) -> Observation:
        try:
            if tool == "bash":
                return self.bash(str(args.get("command", "")), args.get("timeout"))
            if tool == "keystrokes":
                return self.keystrokes(str(args.get("keystrokes", "")), args.get("duration"))
            if tool in ("read", "read_file"):
                return Observation(True, self.files.read(str(args["path"]), self.cap))
            if tool == "edit":
                text = self.files.edit(str(args["path"]), str(args["old"]), str(args["new"]))
                return Observation("[py_compile] OK" in text or "[py_compile]" not in text, text)
            if tool in ("write", "write_file"):
                text = self.files.write(str(args["path"]), str(args.get("content", "")))
                return Observation("[py_compile] OK" in text or "[py_compile]" not in text, text)
        except Exception as exc:  # errors are observations
            return Observation(False, f"error: {type(exc).__name__}: {exc}")
        return Observation(False, f"error: unknown tool {tool!r}")


def _is_docker(sandbox: Any) -> bool:
    return isinstance(sandbox, Container) or type(sandbox).__name__ == "DockerSandbox"


def environment_snapshot(sandbox: Any, timeout: float = 20.0) -> str:
    """Environment facts collected from INSIDE the task's sandbox (never the host)."""
    probes = ["pwd", "ls -la", "python3 --version || python --version", "node --version",
              "gcc --version | head -1", "java -version 2>&1 | head -1", "rustc --version",
              "go version", "pip --version", "free -h | head -2"]
    script = "; ".join(f"echo '$ {p}'; ({p}) 2>&1 | head -20 || echo '[unavailable]'" for p in probes)
    try:
        _, out = run_in_sandbox(sandbox, script, timeout)
    except Exception as exc:
        out = f"[snapshot failed: {type(exc).__name__}]"
    return "[Environment Snapshot]\n" + cap_output(out.strip(), 4096)


class TerminalBootstrap:
    """Initial prompt = base prompt + an environment snapshot taken inside `sandbox`."""

    def __init__(self, base_prompt: str, snapshot: str | None = None, sandbox: Any = None):
        if snapshot is None and sandbox is None:
            raise ValueError("TerminalBootstrap needs a snapshot or a sandbox to take it in")
        self.base_prompt = base_prompt
        self.snapshot = snapshot if snapshot is not None else environment_snapshot(sandbox)

    def initial_prompt(self) -> str:
        return f"{self.base_prompt}\n\n{self.snapshot}"


def _parse_json(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start:end + 1] if 0 <= start < end else None
    if candidate is None:
        return None
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


TERMINUS2_SYSTEM = """You are an AI assistant solving a command-line task in a Linux environment.
You are given the task and the output of your previous commands. Reply ONLY with one JSON object:

{
  "analysis": "what the last output shows and what is still missing",
  "plan": "what the next batch of commands will do and why",
  "commands": [
    {"keystrokes": "ls -la\\n", "duration": 0.1},
    {"read": {"path": "solve.py"}},
    {"edit": {"path": "solve.py", "old": "exact old lines", "new": "replacement lines"}},
    {"write": {"path": "new_file.py", "content": "full file content"}}
  ],
  "task_complete": false
}

Rules:
- Each keystrokes item is one shell command, run to completion; its output comes back to you.
- Read a file in full before you edit or overwrite it. edit replaces old with new.
- Inspect before you change; verify (run the tests or the program) before you finish.
- Set "task_complete": true only when the task is solved; you will be asked to confirm."""

CONFIRM_PROMPT = ("You set task_complete: true. Are you sure the task is solved? This ends the "
                  "episode and your solution is graded; no further changes are possible. If yes, "
                  "reply again with \"task_complete\": true. Otherwise continue working.")
NO_ACTION_PROMPT = ("Your reply had no valid JSON with commands. Reply with the JSON object "
                    "described in the instructions.")


class Terminus2Harness:
    """Terminus-2 port. Class constants are the knobs the proposer is expected to tune."""

    SYSTEM = TERMINUS2_SYSTEM
    MAX_STEPS = 300  # paper default; small runs pass max_steps explicitly
    STEP_TIMEOUT = 120.0
    CONTEXT_WINDOW = 128_000
    B1 = B1
    B2 = B2
    RECENT_FRACTION = RECENT_FRACTION
    MIN_RECENT_TURNS = MIN_RECENT_TURNS
    OBSERVATION_CAP = OBSERVATION_CAP_BYTES
    CONFIRM = CONFIRM_PROMPT
    TOOL_NAMES = ("keystrokes", "read", "edit", "write")

    def __init__(self, max_steps: int | None = None, context_window: int | None = None,
                 compaction: bool = True, planning: bool = False, bash_only: bool = False,
                 step_timeout: float | None = None, allow_local_shell: bool = False,
                 bootstrap: str | None = None):
        self.max_steps = int(max_steps or self.MAX_STEPS)
        self.context_window = int(context_window or self.CONTEXT_WINDOW)
        self.compaction = compaction
        self.planning = planning
        self.bash_only = bash_only
        self.step_timeout = float(step_timeout or self.STEP_TIMEOUT)
        self.allow_local_shell = allow_local_shell or os.environ.get("META_HARNESS_ALLOW_LOCAL_SHELL") == "1"
        if bootstrap:
            self.SYSTEM = bootstrap

    # ------------------------------------------------------------------ environment
    def open_sandbox(self, task: Mapping[str, Any]) -> tuple[Any, str | None, bool]:
        """(sandbox, workdir, owned). The domain's container wins; a TempDirSandbox only on opt-in."""
        workdir = str(task.get("workdir") or "/app")
        if task.get("container"):
            return Container.attach(str(task["container"]), workdir), workdir, False
        if task.get("sandbox") is not None:
            return task["sandbox"], None, False
        if not self.allow_local_shell:
            raise PermissionError("no container on the task and allow_local_shell is False; refusing "
                                  "to run model-authored commands outside a sandbox container")
        box = TempDirSandbox()
        box.start()
        return box, None, True

    def initial_message(self, task: Mapping[str, Any], tools: ToolBox) -> str:
        return "Task:\n" + str(task["instruction"])

    # ------------------------------------------------------------------ model I/O
    def render(self, messages: list[dict]) -> str:
        parts = [self.SYSTEM]
        for m in messages:
            parts.append(f"## {m.get('role')}\n{message_text(m)}")
        return "\n\n".join(parts)

    def summarize(self, model: Any, prompt: str) -> str:
        return str(model(prompt))

    @staticmethod
    def _with_injection(messages: list[dict], block: str) -> list[dict]:
        """Ephemeral copy with `block` appended to the last message; history is untouched."""
        if not block or not messages:
            return messages
        last = dict(messages[-1])
        key = "content" if "content" in last else "text"
        last[key] = f"{last.get(key, '')}\n\n{block}"
        return messages[:-1] + [last]

    def next_turn(self, model: Any, messages: list[dict], injection: str) -> tuple[dict, list[tuple[str, dict, str]], bool]:
        """Returns (assistant message, actions [(tool, args, call_id)], task_complete flag)."""
        text = str(model(self.render(self._with_injection(messages, injection))))
        assistant = {"role": "assistant", "text": text, "tool_calls": []}
        data = _parse_json(text)
        if data is None:
            return assistant, [], False
        actions: list[tuple[str, dict, str]] = []
        if self.planning and "update_plan" in data:
            actions.append(("update_plan", {"items": data["update_plan"]}, ""))
        for item in data.get("commands") or []:
            if not isinstance(item, Mapping):
                continue
            if "keystrokes" in item:
                actions.append(("keystrokes", {"keystrokes": str(item["keystrokes"]),
                                               "duration": item.get("duration")}, ""))
            for tool in ("read", "edit", "write"):
                if tool in item and isinstance(item[tool], Mapping):
                    actions.append((tool, dict(item[tool]), ""))
        return assistant, actions, bool(data.get("task_complete"))

    def observation_messages(self, results: list[tuple[str, dict, str, str]]) -> list[dict]:
        body = "\n\n".join(f"$ {self._describe(tool, args)}\n{text}" for tool, args, _, text in results)
        return [{"role": "user", "kind": "observation", "content": cap_output(body, self.OBSERVATION_CAP)}]

    @staticmethod
    def _describe(tool: str, args: Mapping[str, Any]) -> str:
        if tool == "keystrokes":
            return str(args.get("keystrokes", "")).rstrip("\n")
        if tool == "bash":
            return str(args.get("command", ""))
        return f"{tool} {args.get('path', '')}".strip()

    def completion_prompt(self) -> str:
        return self.CONFIRM

    # ------------------------------------------------------------------ the loop
    def run(self, task: Mapping[str, Any], model: Callable[..., Any], trace: Any = None) -> dict[str, Any]:
        if "instruction" not in task:
            _emit(trace, "terminal_skip", {"reason": "no instruction"})
            return {"output": "", "stop_reason": "done"}
        sandbox, workdir, owned = self.open_sandbox(task)
        try:
            return self._loop(task, model, trace, sandbox, workdir)
        finally:
            if owned:
                sandbox.close()

    def _loop(self, task: Mapping[str, Any], model: Any, trace: Any, sandbox: Any,
              workdir: str | None) -> dict[str, Any]:
        policy = ShellPolicy(timeout_seconds=self.step_timeout, max_steps=self.max_steps)
        tools = ToolBox(sandbox, workdir, policy, self.OBSERVATION_CAP)
        deadline = time.monotonic() + float(task.get("timeout") or 1e9)
        compactor = StagedCompactor(self.context_window, self.B1, self.B2, self.RECENT_FRACTION,
                                    self.MIN_RECENT_TURNS) if self.compaction else None
        stuck = StuckDetector()
        plan = Plan()
        labels: list[str] = []
        messages: list[dict] = [{"role": "user", "content": self.initial_message(task, tools)}]
        awaiting_confirm = False
        stop, answer, steps = "step_limit", "step limit reached", 0
        for step in range(self.max_steps):
            steps = step + 1
            if time.monotonic() > deadline:
                stop, answer = "timeout", "task time limit reached"
                break
            if compactor is not None:
                messages, info = compactor.compact(messages, lambda p: self.summarize(model, p), self.SYSTEM)
                if info.elided or info.summary_calls:
                    _emit(trace, "terminal_compaction", info.__dict__)
            if message_tokens(messages) + len(self.SYSTEM) // 4 > self.context_window:
                stop, answer = "overflow", "context window exceeded"
                break
            injection = plan.block() if self.planning else ""
            try:
                assistant, actions, complete = self.next_turn(model, messages, injection)
            except Exception as exc:
                stop, answer = "error", f"model call failed: {type(exc).__name__}: {exc}"
                break
            messages.append(assistant)
            results: list[tuple[str, dict, str, str]] = []
            reminders: list[str] = []
            for tool, args, call_id in actions:
                if self.bash_only and tool not in ("bash", "keystrokes", "task_complete"):
                    obs = Observation(False, f"error: {tool} is disabled (bash_only); use shell commands")
                elif tool == "update_plan":
                    try:
                        obs = Observation(True, plan.update(args.get("items")))
                    except ValueError as exc:
                        obs = Observation(False, f"error: {exc}")
                elif tool == "task_complete":
                    complete = True
                    obs = Observation(True, "noted")
                else:
                    obs = tools.dispatch(tool, args)
                label = label_action(tool, args)
                labels.append(label)
                note = stuck.observe(tool, args, failed=not obs.ok, denied=obs.denied)
                if note:
                    reminders.append(note)
                results.append((tool, args, call_id, obs.text))
                _emit(trace, "terminal_step", {"step": steps, "tool": tool, "args": args, "ok": obs.ok,
                                               "phase": label, "observation": obs.text})
                if stuck.stuck:
                    break
            if stuck.stuck:
                stop, answer = "stuck", "stopped: the same call failed 8 times in a row"
                break
            if complete and awaiting_confirm:
                stop, answer = "done", self._answer(assistant)
                break
            awaiting_confirm = complete
            follow = list(reminders)
            if complete:
                follow.append(self.completion_prompt())
            if not actions and not complete:
                follow.append(NO_ACTION_PROMPT)
            new = self.observation_messages(results) if results else []
            if follow:
                extra = "\n\n".join(follow)
                if new and new[-1].get("role") == "user":
                    new[-1]["content"] = f"{new[-1]['content']}\n\n{extra}"
                else:
                    new.append({"role": "user", "content": extra})
            messages.extend(new)
        stats = phase_summary(labels)
        _emit(trace, "terminal_end", {"stop_reason": stop, "steps": steps, **stats})
        return {"output": answer, "stop_reason": stop, "steps": steps, **stats}

    @staticmethod
    def _answer(assistant: Mapping[str, Any]) -> str:
        data = _parse_json(str(assistant.get("text", ""))) or {}
        return str(data.get("analysis") or data.get("summary") or assistant.get("text", ""))[:2000]


KIRA_SYSTEM = """You are Terminus-KIRA, an autonomous agent solving a command-line task inside a \
Linux container. Work with the tools: bash runs one shell command; read_file shows a file; edit \
replaces old text with new text in a file you have read; write_file writes a whole file; \
task_complete ends the episode after you have verified the result.

Approach: understand the environment (see the snapshot), make the change, then verify it by \
running the tests or the program. Do not stop before writing the solution."""

KIRA_CHECKLIST = """Before this is accepted, go through the completion checklist:
1. Re-read the task instruction; is every requirement met (file names, paths, formats)?
2. Do the expected output files exist with the right content?
3. Did you run the program or tests and see them pass?
4. Did you leave stray processes, temporary files or debugging output behind?
If all four hold, call task_complete again (or reply "task_complete": true). Otherwise keep working."""


class KiraHarness(Terminus2Harness):
    """Terminus-KIRA: native tool calling, completion checklist, in-container environment snapshot."""

    SYSTEM = KIRA_SYSTEM
    CONFIRM = KIRA_CHECKLIST

    def initial_message(self, task: Mapping[str, Any], tools: ToolBox) -> str:
        return f"Task:\n{task['instruction']}\n\n{environment_snapshot(tools.sandbox)}"

    def tool_specs(self) -> list[ToolSpec]:
        def obj(props: dict, required: list[str]) -> dict:
            return {"type": "object", "properties": props, "required": required}
        s = {"type": "string"}
        specs = [ToolSpec("bash", "Run one shell command in the task container; returns exit code and output.",
                          obj({"command": s, "timeout": {"type": "number"}}, ["command"]))]
        if not self.bash_only:
            specs += [
                ToolSpec("read_file", "Read a whole file (required before editing or overwriting it).",
                         obj({"path": s}, ["path"])),
                ToolSpec("edit", "Replace old text with new text in a file you have read. Exact match "
                                 "first, then a whitespace-tolerant / fuzzy unique match.",
                         obj({"path": s, "old": s, "new": s}, ["path", "old", "new"])),
                ToolSpec("write_file", "Write a whole file.", obj({"path": s, "content": s}, ["path", "content"])),
            ]
            if self.planning:
                item = obj({"step": s, "status": {"type": "string", "enum": ["pending", "in_progress",
                                                                              "completed"]}}, ["step", "status"])
                specs.append(ToolSpec("update_plan", "Replace the whole plan; exactly one step in_progress.",
                                      obj({"items": {"type": "array", "items": item}}, ["items"])))
        specs.append(ToolSpec("task_complete", "Declare the task solved (asks for the checklist first).",
                              obj({"summary": s}, ["summary"])))
        return specs

    @staticmethod
    def _uses_chat(model: Any) -> bool:
        return callable(getattr(model, "chat", None))

    def summarize(self, model: Any, prompt: str) -> str:
        if self._uses_chat(model):
            return model.chat("You summarise agent sessions.", [{"role": "user", "content": prompt}], []).text
        return str(model(prompt))

    def next_turn(self, model: Any, messages: list[dict], injection: str):
        if not self._uses_chat(model):
            return super().next_turn(model, messages, injection)
        wire = [{k: v for k, v in m.items() if k != "kind"} for m in self._with_injection(messages, injection)]
        turn = model.chat(self.SYSTEM, wire, self.tool_specs())
        assistant = {"role": "assistant", "text": turn.text, "tool_calls": list(turn.tool_calls)}
        actions, complete = [], False
        for call in turn.tool_calls:
            args = dict(call.arguments)
            if call.name == "task_complete":
                complete = True
                self._summary = str(args.get("summary", ""))
            actions.append((call.name, args, call.id))
        return assistant, actions, complete

    def observation_messages(self, results):
        if not results or not results[0][2]:
            return super().observation_messages(results)
        return [{"role": "tool", "tool_call_id": call_id, "name": tool, "content": text}
                for tool, _, call_id, text in results]

    def _answer(self, assistant: Mapping[str, Any]) -> str:
        summary = getattr(self, "_summary", "")
        return summary or super()._answer(assistant)


class TerminalAgent:
    """The original single-command agent API (`{"command", "done", "answer"}` JSON per turn),
    kept for callers of `TerminalAgent(model, policy).run(task, trace)`. Commands go through a
    sandbox: the task's container when present, otherwise a fresh `TempDirSandbox`."""

    def __init__(self, model: Callable[..., Any], policy: ShellPolicy | None = None, bootstrap: str = "",
                 allow_local_shell: bool = False):
        self.model, self.policy, self.bootstrap = model, policy or ShellPolicy(), bootstrap
        self.allow_local_shell = allow_local_shell or os.environ.get("META_HARNESS_ALLOW_LOCAL_SHELL") == "1"

    @staticmethod
    def parse_action(output: str) -> dict[str, Any]:
        return _parse_json(output) or {"command": "", "done": False, "answer": output}

    def run(self, task: Mapping[str, Any], trace: Any = None) -> TerminalResult:
        instruction = str(task.get("instruction", task.get("input", "")))
        if task.get("container"):
            sandbox, owned = Container.attach(str(task["container"]), str(task.get("workdir") or "/app")), False
        elif not self.allow_local_shell:
            raise PermissionError("no container on the task and allow_local_shell is False; refusing "
                                  "to run model-authored commands outside a sandbox container")
        else:
            sandbox, owned = TempDirSandbox(), True
            sandbox.start()
        tools = ToolBox(sandbox, None, self.policy, self.policy.max_output_chars)
        history: list[dict[str, Any]] = []
        try:
            for step in range(self.policy.max_steps):
                prompt = self.bootstrap + "\n\nTask:\n" + instruction + (
                    "\n\nExecution history:\n" + json.dumps(history, ensure_ascii=False) if history else "")
                action = self.parse_action(str(self.model(prompt)))
                if bool(action.get("done")):
                    result = TerminalResult(True, str(action.get("answer", "")), step + 1, history)
                    _emit(trace, "terminal_complete", result.__dict__)
                    return result
                command = str(action.get("command", ""))
                obs = tools.bash(command) if command else Observation(
                    False, "No command was supplied. Return a command or set done=true.")
                item = {"step": step + 1, "action": action, "ok": obs.ok, "observation": obs.text}
                history.append(item)
                _emit(trace, "terminal_step", item)
        finally:
            if owned:
                sandbox.close()
        _emit(trace, "terminal_limit", {"max_steps": self.policy.max_steps})
        return TerminalResult(False, "step limit reached", self.policy.max_steps, history, "step_limit")


# Backwards-compatible name: the searchable terminal harness is the Terminus-2 port.
TerminalHarness = Terminus2Harness

__all__ = ["DockerPolicy", "KiraHarness", "TerminalAgent", "TerminalBootstrap", "TerminalHarness",
           "TerminalResult", "Terminus2Harness", "ToolBox", "environment_snapshot"]
