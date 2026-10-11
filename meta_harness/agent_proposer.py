"""Proposer view modes: the paper's Table 3 ablation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable, Sequence

from .core import FilesystemExperience

VIEW_MODES = ("scores", "summary", "full")

_SUMMARY_PROMPT = (
    "Summarize this harness evaluation for an engineer who will try to improve it. "
    "Cover: what the harness does, its score and context cost, and the failure pattern in the "
    "trace. Maximum 150 words.\n\nSource:\n{source}\n\nScores:\n{scores}\n\nTrace excerpt:\n{trace}"
)


def _copy_view(experience: FilesystemExperience, destination: Path, keep: Sequence[str]) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidates").mkdir(exist_ok=True)
    for name in ("run.json", "frontier.json"):
        source = experience.root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for directory in sorted(experience.candidates.iterdir()):
        if not directory.is_dir():
            continue
        target = destination / "candidates" / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in keep:
            source = directory / name
            if source.is_file():
                shutil.copy2(source, target / name)
    return destination


def _trace_excerpt(path: Path, lines: int = 40) -> str:
    if not path.is_file():
        return "(no trace)"
    with path.open(encoding="utf-8", errors="replace") as handle:
        return "".join(line for _, line in zip(range(lines), handle))


def build_view(experience: FilesystemExperience, mode: str, destination: Path,
               summarizer: Callable[..., str] | None = None) -> Path:
    """Materialize what the proposer is allowed to read. Paper Table 3 ablation."""
    if mode not in VIEW_MODES:
        raise ValueError(f"view must be one of {VIEW_MODES}")
    if mode == "full":
        return experience.root
    view = _copy_view(experience, Path(destination), ("harness.py", "scores.json", "proposal.json"))
    if mode == "summary":
        if summarizer is None:
            raise ValueError("summary view requires a summarizer model")
        for directory in sorted(experience.candidates.iterdir()):
            if not directory.is_dir():
                continue
            scores = directory / "scores.json"
            prompt = _SUMMARY_PROMPT.format(
                source=(directory / "harness.py").read_text(encoding="utf-8", errors="replace")[:6000]
                if (directory / "harness.py").is_file() else "(missing)",
                scores=scores.read_text(encoding="utf-8", errors="replace") if scores.is_file() else "{}",
                trace=_trace_excerpt(directory / "traces.jsonl"))
            (view / "candidates" / directory.name / "summary.md").write_text(
                str(summarizer(prompt)), encoding="utf-8")
    return view


__all__ = ["VIEW_MODES", "build_view"]
