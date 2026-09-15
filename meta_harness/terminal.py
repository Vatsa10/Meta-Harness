"""A bounded terminal-agent loop for TerminalBench-style tasks."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass
class ShellPolicy:
    timeout_seconds: float = 30.0
    max_output_chars: int = 30000
    max_steps: int = 80
    cwd: str | None = None
    allowed_commands: tuple[str, ...] = ()

    def check(self, command: str) -> None:
        if len(command) > 20000:
            raise PermissionError("command exceeds maximum length")
        if self.allowed_commands:
            program = command.strip().split(maxsplit=1)[0]
            if program not in self.allowed_commands:
                raise PermissionError(f"command is not allowlisted: {program}")


@dataclass
class TerminalResult:
    completed: bool
    answer: str
    steps: int
    history: list[dict[str, Any]] = field(default_factory=list)


def run_command(command: str, policy: ShellPolicy) -> tuple[int, str]:
    policy.check(command)
    try:
        completed = subprocess.run(command, shell=True, cwd=policy.cwd, capture_output=True, text=True, timeout=policy.timeout_seconds)
        output = (completed.stdout + completed.stderr)[-policy.max_output_chars:]
        return completed.returncode, output
    except subprocess.TimeoutExpired as error:
        output = str(error.stdout or "")[-policy.max_output_chars:]
        return 124, output + "\n[command timed out]"


class TerminalAgent:
    """Model protocol: return JSON with command, done, and answer fields."""

    def __init__(self, model: Callable[..., Any], policy: ShellPolicy | None = None, bootstrap: str = ""):
        self.model, self.policy, self.bootstrap = model, policy or ShellPolicy(), bootstrap

    @staticmethod
    def parse_action(output: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", output, flags=re.DOTALL)
        if not match:
            return {"command": "", "done": False, "answer": output}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {"command": "", "done": False, "answer": output}
        except json.JSONDecodeError:
            return {"command": "", "done": False, "answer": output}

    def run(self, task: Mapping[str, Any], trace: Any) -> TerminalResult:
        instruction = str(task.get("instruction", task.get("input", "")))
        prompt = self.bootstrap + "\n\nTask:\n" + instruction
        history: list[dict[str, Any]] = []
        for step in range(self.policy.max_steps):
            try:
                response = self.model(prompt, response_format={"type": "json_object"})
            except TypeError:
                response = self.model(prompt)
            action = self.parse_action(str(response))
            command = str(action.get("command", ""))
            if bool(action.get("done")):
                result = TerminalResult(True, str(action.get("answer", "")), step + 1, history)
                trace.event("terminal_complete", result.__dict__)
                return result
            if not command:
                observation = "No command was supplied. Return a command or set done=true."
                code = 2
            else:
                code, observation = run_command(command, self.policy)
            item = {"step": step + 1, "action": action, "returncode": code, "observation": observation}
            history.append(item)
            trace.event("terminal_step", item)
            prompt = self.bootstrap + "\n\nTask:\n" + instruction + "\n\nExecution history:\n" + json.dumps(history, ensure_ascii=False)
        trace.event("terminal_limit", {"max_steps": self.policy.max_steps})
        return TerminalResult(False, "step limit reached", self.policy.max_steps, history)
