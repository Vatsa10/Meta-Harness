"""Provider-neutral chat types, free of any HTTP code.

Neutral messages:
    {"role": "user", "content": str}
    {"role": "assistant", "text": str, "tool_calls": [ToolCall]}
    {"role": "tool", "tool_call_id": str, "name": str, "content": str}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]  # {"_raw": text} when the provider sent invalid JSON


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0
    # OpenAI: True (prompt_tokens already includes cached tokens); Anthropic: False.
    input_includes_cache: bool = False

    def context_tokens(self) -> int:
        """The Pareto axis: everything the model read."""
        if self.input_includes_cache:
            return self.input
        return self.input + self.cache_read + self.cache_write

    def __add__(self, other: "Usage") -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        if self.input_includes_cache == other.input_includes_cache:
            return Usage(self.input + other.input, self.output + other.output,
                         self.cache_read + other.cache_read, self.cache_write + other.cache_write,
                         self.reasoning + other.reasoning, self.input_includes_cache)

        # Mixed conventions: normalise both to cache-exclusive input.
        def base(u: Usage) -> int:
            return max(0, u.input - u.cache_read - u.cache_write) if u.input_includes_cache else u.input

        return Usage(base(self) + base(other), self.output + other.output,
                     self.cache_read + other.cache_read, self.cache_write + other.cache_write,
                     self.reasoning + other.reasoning, False)


STOP_REASONS = ("end", "tool_use", "length", "other")


@dataclass
class AssistantTurn:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str  # normalised: one of STOP_REASONS
    provider_blocks: list[dict] = field(default_factory=list)  # raw content kept for echo


__all__ = ["ToolSpec", "ToolCall", "Usage", "AssistantTurn", "STOP_REASONS"]
