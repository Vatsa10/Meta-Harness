"""Agentic-coding domain (paper section 4.3): the bounded terminal agent as a searchable harness.

Commands run inside a Docker container by default. Running model-authored commands on the
host is a real risk, so it requires an explicit opt-in.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .terminal import ShellPolicy, TerminalAgent, TerminalResult, run_command


@dataclass
class DockerPolicy(ShellPolicy):
    container: str = ""
    workdir: str = "/app"

    def wrap(self, command: str) -> list[str]:
        return ["docker", "exec", "-w", self.workdir, self.container, "bash", "-lc", command]


def start_container(image: str, workdir: str = "/app") -> str:
    name = f"meta-harness-{uuid.uuid4().hex[:10]}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-w", workdir,
                    image, "sleep", "infinity"], check=True, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    return name


def stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True,
                   encoding="utf-8", errors="replace")


def _run_in_docker(command: str, policy: DockerPolicy) -> tuple[int, str]:
    policy.check(command)
    try:
        completed = subprocess.run(policy.wrap(command), capture_output=True, text=True,
                                   timeout=policy.timeout_seconds,
                                   encoding="utf-8", errors="replace")
        return completed.returncode, (completed.stdout + completed.stderr)[-policy.max_output_chars:]
    except subprocess.TimeoutExpired as error:
        return 124, (str(error.stdout or "")[-policy.max_output_chars:]) + "\n[command timed out]"


def _drive(agent: TerminalAgent, policy: ShellPolicy, runner: Callable[[str], tuple[int, str]],
           task: Mapping[str, Any], trace: Any) -> TerminalResult:
    """TerminalAgent.run with a pluggable command runner (host or container)."""
    instruction = str(task.get("instruction", task.get("input", "")))
    history: list[dict[str, Any]] = []
    for step in range(policy.max_steps):
        prompt = agent.bootstrap + "\n\nTask:\n" + instruction + (
            "\n\nExecution history:\n" + json.dumps(history, ensure_ascii=False) if history else "")
        action = agent.parse_action(str(agent.model(prompt)))
        if bool(action.get("done")):
            result = TerminalResult(True, str(action.get("answer", "")), step + 1, history)
            trace.event("terminal_complete", result.__dict__)
            return result
        command = str(action.get("command", ""))
        if command:
            try:
                code, observation = runner(command)
            except PermissionError as exc:
                code, observation = 126, f"blocked: {exc}"
        else:
            code, observation = 2, "No command was supplied. Return a command or set done=true."
        item = {"step": step + 1, "action": action, "returncode": code, "observation": observation}
        history.append(item)
        trace.event("terminal_step", item)
    trace.event("terminal_limit", {"max_steps": policy.max_steps})
    return TerminalResult(False, "step limit reached", policy.max_steps, history)


class TerminalHarness:
    """A searchable harness around TerminalAgent. The proposer edits bootstrap and step budget."""

    def __init__(self, bootstrap: str = "", max_steps: int = 40, allow_local_shell: bool = False,
                 step_timeout: float = 30.0):
        self.bootstrap = bootstrap or (
            "You are an autonomous terminal agent. Reply ONLY with JSON: "
            '{"command": "<shell command>", "done": false} or {"done": true, "answer": "<summary>"}.')
        self.max_steps = max_steps
        self.allow_local_shell = allow_local_shell or os.environ.get("META_HARNESS_ALLOW_LOCAL_SHELL") == "1"
        self.step_timeout = step_timeout

    def run(self, task: Mapping[str, Any], model: Callable[..., Any], trace: Any) -> str:
        if "instruction" not in task:
            # Interface-validation probe (sandbox.py sends {"input", "label"}). Nothing to run.
            trace.event("terminal_skip", {"reason": "no instruction"})
            return ""
        image = task.get("image")
        workdir = task.get("workdir") or "/app"
        if not image and not self.allow_local_shell:
            raise PermissionError(
                "no image given and allow_local_shell is False; refusing to run model-authored "
                "commands on the host")
        container = start_container(str(image), workdir) if image else None
        try:
            if container:
                policy: ShellPolicy = DockerPolicy(timeout_seconds=self.step_timeout, max_steps=self.max_steps,
                                                   container=container, workdir=workdir)
                runner = lambda command: _run_in_docker(command, policy)  # noqa: E731
            else:
                policy = ShellPolicy(timeout_seconds=self.step_timeout, max_steps=self.max_steps,
                                     cwd=task.get("workdir"))
                runner = lambda command: run_command(command, policy)  # noqa: E731
            agent = TerminalAgent(model, policy, self.bootstrap)
            return _drive(agent, policy, runner, task, trace).answer
        finally:
            if container:
                stop_container(container)


__all__ = ["DockerPolicy", "TerminalHarness", "start_container", "stop_container"]
