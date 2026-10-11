"""Scripted provider: replays canned responses from a JSONL file, offline."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Sequence

from .chat_types import AssistantTurn, ToolCall, ToolSpec, Usage
from .metrics import estimate_tokens


def _provider_error(message: str, status: int | None = None) -> Exception:
    from .providers import ProviderError  # lazy: providers imports this module lazily too

    error = ProviderError(message)
    if status is not None:
        error.status = status
    return error


class ScriptedModel:
    """Replays canned responses from a JSONL file. For offline tests and demos.

    Each line is one entry:
      {"match": "<regex>", "text": "..."}            reusable; first entry whose regex is found in
                                                    the prompt (call) or last user/tool message (chat)
      {"text": "...", "tool_calls": [{"name": ..., "arguments": {...}}]}
                                                    sequential; consumed once, in file order
      {"default": "..."}                             used by call() when nothing else applies
      {"error": {"status": 429, "message": "..."}}   sequential; raised as ProviderError(.status)
      {"stop_reason": "length", "text": "..."}       sequential; simulates truncation
    usage is estimated as estimate_tokens(prompt) / estimate_tokens(text).
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.model = f"scripted:{self.path.name}"
        self._matches: list[tuple[re.Pattern[str], dict[str, Any]]] = []
        self._sequence: list[dict[str, Any]] = []
        self._default: str | None = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if "match" in entry:
                self._matches.append((re.compile(entry["match"]), entry))
            elif "default" in entry:
                self._default = str(entry["default"])
            else:
                self._sequence.append(entry)
        self._consumed = [False] * len(self._sequence)
        self._lock = threading.Lock()
        self._counter = 0

    # -- helpers -----------------------------------------------------------------
    def _match(self, text: str) -> dict[str, Any] | None:
        for pattern, entry in self._matches:
            if pattern.search(text):
                return entry
        return None

    def _take(self, allow_tool_calls: bool) -> dict[str, Any] | None:
        for index, entry in enumerate(self._sequence):
            if self._consumed[index]:
                continue
            if not allow_tool_calls and entry.get("tool_calls"):
                continue
            self._consumed[index] = True
            return entry
        return None

    def _raise_if_error(self, entry: dict[str, Any]) -> None:
        error = entry.get("error")
        if error is not None:
            raise _provider_error(str(error.get("message", "scripted error")), error.get("status"))

    def _exhausted(self) -> Exception:
        return _provider_error(f"script exhausted: {self.path}")

    # -- completion interface ----------------------------------------------------
    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, int]]:
        with self._lock:
            entry = self._match(prompt)
            if entry is None:
                entry = self._take(allow_tool_calls=False)
            if entry is not None:
                self._raise_if_error(entry)
                text = str(entry.get("text", ""))
            elif self._default is not None:
                text = self._default
            else:
                raise self._exhausted()
        return text, {"input_tokens": estimate_tokens(prompt), "output_tokens": estimate_tokens(text)}

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]

    # -- chat interface ----------------------------------------------------------
    def chat(self, system: str, messages: list[dict], tools: Sequence[ToolSpec],
             max_tokens: int = 4096) -> AssistantTurn:
        with self._lock:
            entry = self._take(allow_tool_calls=True)
            if entry is None:
                entry = self._match(self._last_text(messages))
            if entry is None:
                raise self._exhausted()
            self._raise_if_error(entry)
            calls: list[ToolCall] = []
            for raw in entry.get("tool_calls") or []:
                self._counter += 1
                calls.append(ToolCall(f"scripted-{self._counter}", raw["name"], dict(raw.get("arguments") or {})))
        text = str(entry.get("text", ""))
        prompt = system + "\n" + "\n".join(self._message_text(m) for m in messages)
        stop = entry.get("stop_reason") or ("tool_use" if calls else "end")
        return AssistantTurn(text, calls, Usage(estimate_tokens(prompt), estimate_tokens(text)), stop)

    @staticmethod
    def _message_text(message: dict) -> str:
        value = message.get("content")
        if value is None:
            value = message.get("text", "")
        return str(value)

    def _last_text(self, messages: list[dict]) -> str:
        return self._message_text(messages[-1]) if messages else ""
