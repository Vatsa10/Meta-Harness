"""Coding-agent proposer: the paper's section 3 design, plus the Table 3 view ablation."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

from .core import FilesystemExperience

SKILL_PATH = Path(__file__).resolve().parent / "skill" / "SKILL.md"

VIEW_MODES = ("scores", "summary", "full")

PROPOSER_PROMPT = (
    "Read SKILL.md in this directory and follow it exactly. You are the Meta-Harness proposer "
    "at iteration $META_HARNESS_ITERATION. Inspect the experience filesystem in this directory "
    "(candidates/, run.json, frontier.json), diagnose why the leading harnesses fail, then write "
    "$META_HARNESS_COUNT candidate harness file(s) plus their .reasoning.md sidecars into "
    "$META_HARNESS_OUTPUT. Do not modify anything else."
)

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


class ClaudeCodeProposer:
    """Runs a coding-agent CLI over the experience filesystem.

    Defaults target Claude Code:
        claude -p "<prompt>" --permission-mode acceptEdits [--model M]
    Point it at Unikey by exporting, before the run:
        ANTHROPIC_BASE_URL=https://www.getunikey.ai
        ANTHROPIC_AUTH_TOKEN=$UNIKEY_API_KEY
    """

    def __init__(self, binary: str = "claude", model: str | None = None, view: str = "full",
                 summarizer: Callable[..., str] | None = None, timeout: float = 1800.0,
                 extra_args: Sequence[str] = (), skill_path: Path | str | None = None,
                 permission_mode: str = "acceptEdits", prompt_as_argument: bool = True):
        if view not in VIEW_MODES:
            raise ValueError(f"view must be one of {VIEW_MODES}")
        self.binary = binary
        self.model = model
        self.view = view
        self.summarizer = summarizer
        self.timeout = timeout
        self.extra_args = list(extra_args)
        self.skill_path = Path(skill_path) if skill_path else SKILL_PATH
        self.permission_mode = permission_mode
        self.prompt_as_argument = prompt_as_argument

    def _command(self) -> list[str]:
        if not self.prompt_as_argument:
            return [self.binary, *self.extra_args]
        command = [self.binary, "-p", PROPOSER_PROMPT, "--permission-mode", self.permission_mode]
        if self.model:
            command += ["--model", self.model]
        return command + self.extra_args

    def __call__(self, experience: FilesystemExperience, iteration: int, count: int) -> list[Path]:
        output = experience.root / "proposals" / f"iteration-{iteration:04d}"
        output.mkdir(parents=True, exist_ok=True)
        view = build_view(experience, self.view,
                          experience.root / "views" / f"iteration-{iteration:04d}", self.summarizer)
        shutil.copy2(self.skill_path, Path(view) / "SKILL.md")
        env = os.environ.copy()
        env.update({
            "META_HARNESS_ROOT": str(Path(view).resolve()),
            "META_HARNESS_OUTPUT": str(output.resolve()),
            "META_HARNESS_ITERATION": str(iteration),
            "META_HARNESS_COUNT": str(count),
            "META_HARNESS_VIEW": self.view,
        })
        completed = subprocess.run(self._command(), cwd=view, env=env, text=True,
                                   capture_output=True, timeout=self.timeout,
                                   encoding="utf-8", errors="replace")
        (output / "proposer.stdout").write_text(completed.stdout or "", encoding="utf-8")
        (output / "proposer.stderr").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"proposer exited with {completed.returncode}")
        candidates = sorted(output.glob("*.py"))[:count]
        for path in candidates:
            sidecar = path.with_name(path.stem + ".reasoning.md")
            if not sidecar.is_file():
                sidecar.write_text(completed.stdout or "(no reasoning recorded)", encoding="utf-8")
        return candidates


__all__ = ["PROPOSER_PROMPT", "SKILL_PATH", "VIEW_MODES", "ClaudeCodeProposer", "build_view"]
