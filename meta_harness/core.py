"""The Meta-Harness outer loop.

The implementation keeps the proposer deliberately unopinionated: it receives
an experience directory and may inspect any prior candidate, score, or trace.
That is the important design choice in the paper; parent selection and edit
operators are not hard-coded here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


class HarnessValidationError(Exception):
    """Raised when a proposed harness cannot be safely loaded or called."""


@dataclass
class EvaluationResult:
    candidate_id: str
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    context_cost: float = 0.0
    examples: int = 0
    elapsed_seconds: float = 0.0
    valid: bool = True
    error: str | None = None

    def objective_vector(self) -> tuple[float, float]:
        # Higher score and lower context cost are both preferred.
        return (self.score, -self.context_cost)


@dataclass
class SearchConfig:
    root: Path | str = Path(".meta-harness")
    iterations: int = 10
    candidates_per_iteration: int = 1
    validation_timeout: float = 20.0
    keep_invalid: bool = True
    maximize: str = "score"

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.iterations < 0 or self.candidates_per_iteration < 1:
            raise ValueError("iterations must be non-negative and candidates_per_iteration positive")


class TraceRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self.count = 0
        self.context_cost = 0.0

    def add_context_cost(self, amount: float) -> None:
        """Record model-input cost in arbitrary, caller-defined units."""
        self.context_cost += max(0.0, float(amount))

    def event(self, name: str, payload: Any = None, **fields: Any) -> None:
        record = {"time": time.time(), "event": name, "payload": payload, **fields}
        self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._file.flush()
        self.count += 1

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class FilesystemExperience:
    """The queryable filesystem D from the paper."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidates.mkdir(parents=True, exist_ok=True)

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates / candidate_id

    def add_source(self, source: Path | str, candidate_id: str | None = None) -> str:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        candidate_id = candidate_id or f"candidate-{int(time.time() * 1000)}-{digest}"
        directory = self.candidate_dir(candidate_id)
        suffix = 1
        while directory.exists():
            directory = self.candidate_dir(f"{candidate_id}-{suffix}")
            suffix += 1
        candidate_id = directory.name
        directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, directory / "harness.py")
        (directory / "proposal.json").write_text(json.dumps({"candidate_id": candidate_id}, indent=2), encoding="utf-8")
        return candidate_id

    def write_result(self, result: EvaluationResult) -> None:
        directory = self.candidate_dir(result.candidate_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "scores.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    def results(self) -> list[EvaluationResult]:
        out: list[EvaluationResult] = []
        for path in sorted(self.candidates.glob("*/scores.json")):
            try:
                out.append(EvaluationResult(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, TypeError, ValueError):
                continue
        return out

    def manifest(self) -> list[dict[str, Any]]:
        return [{"candidate_id": r.candidate_id, "score": r.score,
                 "context_cost": r.context_cost, "valid": r.valid}
                for r in self.results()]


def _load_harness(path: Path) -> Any:
    name = f"meta_harness_candidate_{hashlib.sha256(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise HarnessValidationError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise HarnessValidationError(f"import failed: {exc}") from exc
    if hasattr(module, "build_harness"):
        harness = module.build_harness()
    elif hasattr(module, "Harness"):
        harness = module.Harness()
    else:
        raise HarnessValidationError("candidate must expose build_harness() or Harness")
    if not callable(getattr(harness, "run", None)):
        raise HarnessValidationError("harness must implement run(task, model, trace)")
    return harness


class CandidateEvaluator:
    def __init__(self, model: Callable[..., Any], metric: Callable[[Any, Any], float] | None = None):
        self.model = model
        self.metric = metric or (lambda prediction, task: float(prediction == task.get("label")))

    def validate(self, source: Path | str) -> None:
        harness = _load_harness(Path(source))
        with tempfile.TemporaryDirectory() as td:
            with TraceRecorder(Path(td) / "validation.jsonl") as trace:
                try:
                    harness.run({"input": "validation", "label": "ok"}, self.model, trace)
                except Exception as exc:
                    raise HarnessValidationError(f"validation run failed: {exc}") from exc

    def evaluate(self, source: Path | str, tasks: Sequence[Mapping[str, Any]], candidate_id: str,
                 trace_path: Path) -> EvaluationResult:
        started = time.perf_counter()
        try:
            harness = _load_harness(Path(source))
            score_sum = 0.0
            costs = 0.0
            metrics: dict[str, float] = {}
            with TraceRecorder(trace_path) as trace:
                for index, task in enumerate(tasks):
                    trace.event("task_start", {"index": index, "task": dict(task)})
                    t0 = trace.count
                    c0 = trace.context_cost
                    prediction = harness.run(task, self.model, trace)
                    value = float(self.metric(prediction, task))
                    score_sum += value
                    # Harnesses may report tokens/chars explicitly. The event
                    # count remains a useful deterministic fallback for demos.
                    costs += (trace.context_cost - c0) if trace.context_cost > c0 else max(0, trace.count - t0)
                    trace.event("task_end", {"index": index, "prediction": prediction, "reward": value})
            n = len(tasks) or 1
            return EvaluationResult(candidate_id, score_sum / n, metrics, costs / n, len(tasks), time.perf_counter() - started)
        except Exception as exc:
            return EvaluationResult(candidate_id, float("-inf"), {}, float("inf"), len(tasks), time.perf_counter() - started, False, repr(exc))


class ParetoFrontier:
    @staticmethod
    def select(results: Iterable[EvaluationResult]) -> list[EvaluationResult]:
        valid = [r for r in results if r.valid]
        frontier = []
        for candidate in valid:
            dominated = any(
                other is not candidate and
                other.score >= candidate.score and other.context_cost <= candidate.context_cost and
                (other.score > candidate.score or other.context_cost < candidate.context_cost)
                for other in valid
            )
            if not dominated:
                frontier.append(candidate)
        return sorted(frontier, key=lambda r: (r.score, -r.context_cost), reverse=True)


class CommandProposer:
    """Adapter for a coding agent that writes candidate Python files.

    The command receives the experience root as its first argument and must
    write one or more `.py` files into the supplied output directory. The
    command is run once per iteration, with no hidden prompt or parent rule.
    """

    def __init__(self, command: Sequence[str], timeout: float = 1800):
        self.command = list(command)
        self.timeout = timeout

    def __call__(self, experience: FilesystemExperience, iteration: int, count: int) -> list[Path]:
        out = experience.root / "proposals" / f"iteration-{iteration:04d}"
        out.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({"META_HARNESS_ROOT": str(experience.root.resolve()), "META_HARNESS_OUTPUT": str(out.resolve()),
                    "META_HARNESS_ITERATION": str(iteration), "META_HARNESS_COUNT": str(count)})
        completed = subprocess.run(self.command, cwd=experience.root, env=env, text=True,
                                   capture_output=True, timeout=self.timeout)
        (out / "proposer.stdout").write_text(completed.stdout, encoding="utf-8")
        (out / "proposer.stderr").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"proposer exited with {completed.returncode}")
        return sorted(out.glob("*.py"))[:count]


class SearchRunner:
    def __init__(self, config: SearchConfig, evaluator: CandidateEvaluator,
                 proposer: Callable[[FilesystemExperience, int, int], Sequence[Path]]):
        self.config, self.evaluator, self.proposer = config, evaluator, proposer
        self.experience = FilesystemExperience(config.root)

    def _evaluate_source(self, source: Path, tasks: Sequence[Mapping[str, Any]], label: str) -> EvaluationResult | None:
        try:
            self.evaluator.validate(source)
        except Exception as exc:
            candidate_id = self.experience.add_source(source, label)
            result = EvaluationResult(candidate_id, float("-inf"), valid=False, error=str(exc))
            self.experience.write_result(result)
            return result if self.config.keep_invalid else None
        candidate_id = self.experience.add_source(source, label)
        result = self.evaluator.evaluate(self.experience.candidate_dir(candidate_id) / "harness.py", tasks, candidate_id,
                                         self.experience.candidate_dir(candidate_id) / "traces.jsonl")
        self.experience.write_result(result)
        return result

    def run(self, initial: Sequence[Path | str], tasks: Sequence[Mapping[str, Any]]) -> list[EvaluationResult]:
        self._write_run_manifest(tasks)
        results: list[EvaluationResult] = []
        for index, source in enumerate(initial):
            result = self._evaluate_source(Path(source), tasks, f"initial-{index:04d}")
            if result:
                results.append(result)
        for iteration in range(1, self.config.iterations + 1):
            try:
                proposals = self.proposer(self.experience, iteration, self.config.candidates_per_iteration)
            except Exception as exc:
                proposal_dir = self.experience.root / "proposals" / f"iteration-{iteration:04d}"
                proposal_dir.mkdir(parents=True, exist_ok=True)
                (proposal_dir / "proposer.error").write_text(repr(exc), encoding="utf-8")
                proposals = []
            for index, source in enumerate(proposals):
                result = self._evaluate_source(Path(source), tasks, f"iteration-{iteration:04d}-{index:02d}")
                if result:
                    results.append(result)
        frontier = ParetoFrontier.select(results)
        (self.experience.root / "frontier.json").write_text(json.dumps([asdict(x) for x in frontier], indent=2), encoding="utf-8")
        return frontier

    def _write_run_manifest(self, tasks: Sequence[Mapping[str, Any]]) -> None:
        manifest = {
            "created_at": time.time(),
            "python": sys.version,
            "config": {"root": str(self.config.root), "iterations": self.config.iterations,
                       "candidates_per_iteration": self.config.candidates_per_iteration,
                       "keep_invalid": self.config.keep_invalid},
            "task_count": len(tasks),
            "tasks": [dict(task) for task in tasks],
        }
        (self.experience.root / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
