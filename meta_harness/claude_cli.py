"""Harness model routed through the local `claude` CLI - no API key, no external gateway.

Claude Code is already authenticated on this machine, so the harness being searched can call a
model through the same CLI that runs the proposer. That makes a whole search run with nothing
but Claude Code installed.

Each call is one `claude -p` process:

    claude -p --model haiku --output-format json --max-turns 1 --tools Read
           --system-prompt "..." --setting-sources "" --strict-mcp-config
           --mcp-config {"mcpServers":{}} --exclude-dynamic-system-prompt-sections

The prompt goes in on stdin, never as an argument: harnesses build prompts far past the
Windows command-line limit.

The flags exist to make the CLI behave like a plain completion endpoint rather than an agent:
no project settings, no MCP servers, no dynamic system-prompt sections, a single turn, and the
smallest tool set the CLI accepts. That last one matters - the default tool schema costs about
27,000 input tokens per call, against roughly 1,900 with `--tools Read`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_MODEL = "haiku"
DEFAULT_SYSTEM_PROMPT = (
    "You are a text completion engine inside an evaluation harness. Answer exactly what the "
    "prompt asks, in the format it asks for. No preamble, no explanation, no follow-up "
    "questions, no tool use."
)
# The CLI requires at least one tool name. Read is the cheapest schema, and the model cannot
# use it anyway: --max-turns 1 ends the turn before any tool result could come back.
MINIMAL_TOOLS = "Read"


class ClaudeCliError(RuntimeError):
    pass


@dataclass
class CliRetryPolicy:
    attempts: int = 3
    initial_delay: float = 2.0
    maximum_delay: float = 60.0


class ClaudeCliModel:
    """Callable adapter over the `claude` CLI. Interface matches the HTTP providers."""

    def __init__(self, model: str = DEFAULT_MODEL, binary: str = "claude", timeout: float = 300.0,
                 retry: CliRetryPolicy | None = None, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 extra_args: Sequence[str] = (), tools: str = MINIMAL_TOOLS):
        self.model = model
        self.binary = binary
        self.timeout = timeout
        self.retry = retry or CliRetryPolicy()
        self.system_prompt = system_prompt
        self.extra_args = list(extra_args)
        self.tools = tools

    @staticmethod
    def available(binary: str = "claude") -> bool:
        return shutil.which(binary) is not None

    def _argv(self) -> list[str]:
        return [
            self.binary, "-p",
            "--model", self.model,
            "--output-format", "json",
            "--max-turns", "1",
            "--tools", self.tools,
            "--system-prompt", self.system_prompt,
            "--exclude-dynamic-system-prompt-sections",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
            *self.extra_args,
        ]

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        # A gateway override meant for the proposer must not silently redirect the executor:
        # the point of this provider is the machine's own Claude Code authentication.
        for name in ("ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY"):
            env.pop(name, None)
        # Stop a nested run from inheriting the outer search's plugin/agent context.
        env.pop("CLAUDE_PLUGIN_ROOT", None)
        return env

    @staticmethod
    def _parse(stdout: str) -> tuple[str, dict[str, Any]]:
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise ClaudeCliError(f"CLI did not return JSON: {stdout[:400]}") from error
        events = payload if isinstance(payload, list) else [payload]
        for event in reversed(events):
            if isinstance(event, dict) and event.get("type") == "result":
                if event.get("is_error"):
                    raise ClaudeCliError(f"CLI reported {event.get('subtype')}: "
                                         f"{str(event.get('result'))[:300]}")
                usage = event.get("usage") or {}
                return str(event.get("result", "")), {
                    # Deliberately not "prompt_tokens": the CLI's own count includes its system
                    # prompt and tool schema, which every candidate pays equally and none
                    # controls. Leaving it out lets MeteredModel price the harness prompt
                    # itself, so context cost stays comparable across candidates.
                    "cli_input_tokens": int(usage.get("input_tokens", 0) or 0),
                    "cli_cache_read_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("output_tokens", 0) or 0),
                    "cost_usd": float(event.get("total_cost_usd", 0.0) or 0.0),
                }
        raise ClaudeCliError(f"CLI returned no result event: {stdout[:400]}")

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        argv = self._argv()
        if kwargs.get("model"):
            argv[argv.index("--model") + 1] = str(kwargs["model"])
        last_error: Exception | None = None
        for attempt in range(self.retry.attempts):
            try:
                completed = subprocess.run(argv, input=str(prompt), capture_output=True, text=True,
                                           timeout=self.timeout, env=self._environment(),
                                           encoding="utf-8", errors="replace")
            except subprocess.TimeoutExpired as error:
                last_error = ClaudeCliError(f"CLI timed out after {self.timeout}s")
            except OSError as error:
                raise ClaudeCliError(f"cannot run {self.binary!r}: {error}") from error
            else:
                if completed.returncode == 0:
                    try:
                        return self._parse(completed.stdout)
                    except ClaudeCliError as error:
                        last_error = error
                else:
                    detail = (completed.stderr or completed.stdout or "").strip()
                    last_error = ClaudeCliError(f"CLI exited {completed.returncode}: {detail[:400]}")
            if attempt + 1 < self.retry.attempts:
                time.sleep(min(self.retry.maximum_delay, self.retry.initial_delay * (2 ** attempt)))
        raise ClaudeCliError(str(last_error or "claude CLI call failed"))

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]


__all__ = ["ClaudeCliError", "ClaudeCliModel", "CliRetryPolicy", "DEFAULT_MODEL"]
