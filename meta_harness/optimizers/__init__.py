"""Harness optimizers (proposers). Each module exposes NAME and build(ctx) -> Proposer."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

from ..registry import discover

if TYPE_CHECKING:
    from ..core import FilesystemExperience

Proposer = Callable[["FilesystemExperience", int, int], list[Path]]


@dataclass
class OptimizerContext:
    proposer_model: Any                 # .chat(...) for tool optimizers, __call__(prompt) for text ones
    domain: ModuleType
    view: str = "full"
    max_tool_calls: int = 40
    seed: int = 0
    forbidden: list[Path] = field(default_factory=list)
    summarizer: Callable[[str], str] | None = None
    test_root: Path | None = None       # SearchConfig.test_root: denied to the proposer as the test split

    def __post_init__(self) -> None:
        if self.test_root is not None:
            self.test_root = Path(self.test_root)
            if self.test_root not in [Path(p) for p in self.forbidden]:
                self.forbidden = [*self.forbidden, self.test_root]

    @classmethod
    def for_config(cls, config: Any, proposer_model: Any, domain: ModuleType, **kwargs: Any) -> "OptimizerContext":
        """The context for a search: carries config.test_root and forbids it."""
        return cls(proposer_model, domain, test_root=Path(config.test_root), **kwargs)


def all_optimizers() -> dict[str, ModuleType]:
    return discover("meta_harness.optimizers")


def build(name: str, ctx: OptimizerContext) -> Proposer:
    optimizers = all_optimizers()
    if name not in optimizers:
        raise KeyError(f"unknown optimizer {name!r}; known: {', '.join(sorted(optimizers))}")
    return optimizers[name].build(ctx)


def extract_python(text: str) -> str:
    """The longest fenced Python block, or the bare text when it is already a harness module."""
    matches = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip() + "\n"
    if "class Harness" in text or "def build_harness" in text:
        return text.strip() + "\n"
    raise ValueError("proposer response did not contain a Python harness")
