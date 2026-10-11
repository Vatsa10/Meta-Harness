"""Proposer integrations and candidate-code extraction."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from .core import CommandProposer, FilesystemExperience
from .optimizers import extract_python

_extract_python = extract_python  # backwards-compatible alias


class PromptedProposer:
    """Uses a model to emit candidate source while exposing full history metadata.

    For maximum filesystem/tool use, prefer CommandProposer with a coding-agent
    CLI. This adapter is useful for hosted model APIs that do not expose tools.
    """

    def __init__(self, model: Callable[..., str], instruction: str, timeout_note: str = ""):
        self.model, self.instruction, self.timeout_note = model, instruction, timeout_note

    def __call__(self, experience: FilesystemExperience, iteration: int, count: int) -> list[Path]:
        manifest = json.dumps(experience.manifest(), indent=2)
        prompt = f"""You are the Meta-Harness proposer at iteration {iteration}.
The experience root is {experience.root.resolve()}.
Inspect prior candidates, scores, and traces from that directory before proposing code.
Write a complete Python module exposing class Harness with run(task, model, trace).
Produce {count} independent candidates. Optimize search-set reward and context cost.
{self.instruction}
Current manifest:
{manifest}
{self.timeout_note}
Return each candidate as a separate fenced Python code block."""
        output = self.model(prompt)
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", output, flags=re.DOTALL | re.IGNORECASE)
        if not blocks:
            blocks = [_extract_python(output)]
        out = experience.root / "proposals" / f"iteration-{iteration:04d}"
        out.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, block in enumerate(blocks[:count]):
            path = out / f"candidate-{index:02d}.py"
            path.write_text(block.strip() + "\n", encoding="utf-8")
            paths.append(path)
        return paths


__all__ = ["CommandProposer", "PromptedProposer", "extract_python"]
