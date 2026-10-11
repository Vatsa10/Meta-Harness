"""Experience views the proposer may read: the paper's Table 3 ablation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from .core import FilesystemExperience

VIEW_MODES = ("scores", "summary", "full")  # "seed" is internal (best-of-n)
VIEW_FILES = {"scores": ("harness.py", "scores.json"),
              "summary": ("harness.py", "scores.json", "summary.md"),
              "seed": ("harness.py", "scores.json")}

_SOURCE_CHARS = 6000
_TRACE_CHARS = 12_000

_SUMMARY_PROMPT = (
    "Summarize this harness evaluation for an engineer who will try to improve it. "
    "Cover: what the harness does, its score and context cost, and the failure pattern in the "
    "trace. Maximum 150 words.\n\nSource:\n{source}\n\nScores:\n{scores}\n\nTrace excerpt:\n{trace}"
)


def _read(path: Path, limit: int | None = None, missing: str = "") -> str:
    if not path.is_file():
        return missing
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read(limit) if limit else handle.read()


def _ensure_summary(directory: Path, summarizer: Callable[[str], str]) -> None:
    target = directory / "summary.md"
    if target.is_file():
        return
    prompt = _SUMMARY_PROMPT.format(
        source=_read(directory / "harness.py", _SOURCE_CHARS, "(missing)"),
        scores=_read(directory / "scores.json", None, "{}"),
        trace=_read(directory / "traces.jsonl", _TRACE_CHARS, "(no trace)"))
    target.write_text(str(summarizer(prompt)), encoding="utf-8")


def build_view(experience: FilesystemExperience, mode: str, destination: Path,
               summarizer: Callable[[str], str] | None = None) -> Path:
    """Materialize what the proposer is allowed to read."""
    if mode not in VIEW_MODES and mode != "seed":
        raise ValueError(f"view must be one of {VIEW_MODES}")
    if mode == "full":
        return experience.root
    if mode == "summary" and summarizer is None:
        raise ValueError("summary view requires a summarizer model")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "candidates").mkdir(exist_ok=True)
    for name in ("run.json", "frontier.json"):
        source = experience.root / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for directory in sorted(experience.candidates.iterdir()):
        if not directory.is_dir():
            continue
        if mode == "seed" and not directory.name.startswith("initial-"):
            continue
        if mode == "summary":
            _ensure_summary(directory, summarizer)
        target = destination / "candidates" / directory.name
        target.mkdir(parents=True, exist_ok=True)
        for name in VIEW_FILES[mode]:
            source = directory / name
            if source.is_file():
                shutil.copy2(source, target / name)
    return destination


__all__ = ["VIEW_FILES", "VIEW_MODES", "build_view"]
