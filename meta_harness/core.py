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
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .metrics import estimate_tokens


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
    repeats: int = 1
    score_std: float = 0.0
    split: str = "search"

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
    repeats: int = 1
    max_workers: int = 1
    task_timeout: float = 600.0
    test_root: Path | str | None = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.iterations < 0 or self.candidates_per_iteration < 1:
            raise ValueError("iterations must be non-negative and candidates_per_iteration positive")
        if self.repeats < 1 or self.max_workers < 1:
            raise ValueError("repeats and max_workers must be positive")
        self.test_root = Path(self.test_root) if self.test_root else self.root.with_name(self.root.name + "-test")


class TraceRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self.count = 0
        self.context_cost = 0.0

    def add_context_cost(self, amount: float) -> None:
        """Record model-input cost in tokens."""
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


class MeteredModel:
    """Wraps the base model so every call is priced in tokens and written into the trace."""

    def __init__(self, model: Callable[..., Any], trace: TraceRecorder):
        self.model = model
        self.trace = trace
        self.calls = 0

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        upstream = getattr(self.model, "call", None)
        if callable(upstream):
            text, usage = upstream(prompt, **kwargs)
            text, usage = str(text), dict(usage or {})
        else:
            text, usage = str(self.model(prompt, **kwargs)), {}
        tokens = int(usage.get("prompt_tokens") or 0) or estimate_tokens(prompt)
        self.calls += 1
        self.trace.add_context_cost(tokens)
        self.trace.event("model_call", {"prompt": prompt, "response": text, "tokens": tokens,
                                        "usage": usage, "kwargs": dict(kwargs)})
        return text, usage

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]


class FilesystemExperience:
    """The queryable filesystem D from the paper."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.candidates = self.root / "candidates"
        self.root.mkdir(parents=True, exist_ok=True)
        self.candidates.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def candidate_dir(self, candidate_id: str) -> Path:
        return self.candidates / candidate_id

    def add_source(self, source: Path | str, candidate_id: str | None = None,
                   notes: Mapping[str, str] | None = None) -> str:
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        candidate_id = candidate_id or f"candidate-{int(time.time() * 1000)}-{digest}"
        with self._lock:
            directory = self.candidate_dir(candidate_id)
            suffix = 1
            while directory.exists():
                directory = self.candidate_dir(f"{candidate_id}-{suffix}")
                suffix += 1
            candidate_id = directory.name
            directory.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source, directory / "harness.py")
        for name, body in (notes or {}).items():
            # never let a proposal write outside its own directory
            (directory / Path(name).name).write_text(str(body), encoding="utf-8")
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
    def __init__(self, model: Callable[..., Any], metric: Callable[[Any, Any], float] | None = None,
                 task_timeout: float = 600.0):
        self.model = model
        self.metric = metric or (lambda prediction, task: float(prediction == task.get("label")))
        self.task_timeout = task_timeout

    def validate(self, source: Path | str) -> None:
        harness = _load_harness(Path(source))
        with tempfile.TemporaryDirectory() as td:
            with TraceRecorder(Path(td) / "validation.jsonl") as trace:
                try:
                    harness.run({"input": "validation", "label": "ok"}, MeteredModel(self.model, trace), trace)
                except Exception as exc:
                    raise HarnessValidationError(f"validation run failed: {exc}") from exc

    def evaluate(self, source: Path | str, tasks: Sequence[Mapping[str, Any]], candidate_id: str,
                 trace_path: Path) -> EvaluationResult:
        started = time.perf_counter()
        try:
            harness = _load_harness(Path(source))
            score_sum = 0.0
            tokens = 0.0
            calls = 0
            with TraceRecorder(trace_path) as trace:
                metered = MeteredModel(self.model, trace)
                for index, task in enumerate(tasks):
                    trace.event("task_start", {"index": index, "task": dict(task)})
                    task_started = time.perf_counter()
                    before = trace.context_cost
                    prediction = harness.run(task, metered, trace)
                    elapsed = time.perf_counter() - task_started
                    # ponytail: post-hoc deadline check, not preemption; subprocess validation
                    # (sandbox.validate_in_subprocess) catches hard hangs before we get here.
                    if elapsed > self.task_timeout:
                        raise TimeoutError(f"task {index} exceeded task_timeout ({elapsed:.1f}s)")
                    value = float(self.metric(prediction, task))
                    score_sum += value
                    tokens += trace.context_cost - before
                    trace.event("task_end", {"index": index, "prediction": prediction, "reward": value})
                calls = metered.calls
            n = len(tasks) or 1
            return EvaluationResult(candidate_id, score_sum / n,
                                    {"model_calls": float(calls), "prompt_tokens": tokens},
                                    tokens / n, len(tasks), time.perf_counter() - started)
        except Exception as exc:
            return EvaluationResult(candidate_id, float("-inf"), {}, float("inf"), len(tasks),
                                    time.perf_counter() - started, False, repr(exc))


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

    The command receives the experience root as its working directory and must write one or
    more `.py` files into the supplied output directory. The command is run once per
    iteration, with no hidden prompt or parent rule.
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
        (out / "proposer.stdout").write_text(completed.stdout or "", encoding="utf-8")
        (out / "proposer.stderr").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"proposer exited with {completed.returncode}")
        return sorted(out.glob("*.py"))[:count]


class SearchRunner:
    def __init__(self, config: SearchConfig, evaluator: CandidateEvaluator,
                 proposer: Callable[[FilesystemExperience, int, int], Sequence[Path]]):
        self.config, self.evaluator, self.proposer = config, evaluator, proposer
        self.experience = FilesystemExperience(config.root)

    @staticmethod
    def _sidecar_notes(source: Path) -> dict[str, str]:
        sidecar = source.with_name(source.stem + ".reasoning.md")
        if sidecar.is_file():
            return {"proposer_reasoning.md": sidecar.read_text(encoding="utf-8", errors="replace")}
        return {}

    @staticmethod
    def _merge_repeats(candidate_id: str, runs: Sequence[EvaluationResult], split: str) -> EvaluationResult:
        failed = next((run for run in runs if not run.valid), None)
        if failed is not None:
            return EvaluationResult(candidate_id, float("-inf"), {}, float("inf"), failed.examples,
                                    sum(run.elapsed_seconds for run in runs), False, failed.error,
                                    len(runs), 0.0, split)
        scores = [run.score for run in runs]
        merged_metrics: dict[str, float] = {}
        for key in {key for run in runs for key in run.metrics}:
            values = [run.metrics.get(key, 0.0) for run in runs]
            merged_metrics[key] = sum(values) / len(values)
        return EvaluationResult(
            candidate_id,
            sum(scores) / len(scores),
            merged_metrics,
            sum(run.context_cost for run in runs) / len(runs),
            runs[0].examples,
            sum(run.elapsed_seconds for run in runs),
            True,
            None,
            len(runs),
            statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            split,
        )

    def _evaluate_source(self, source: Path, tasks: Sequence[Mapping[str, Any]], label: str,
                         split: str = "search") -> EvaluationResult | None:
        from .sandbox import validate_in_subprocess

        notes = self._sidecar_notes(source)
        try:
            validate_in_subprocess(source, self.config.validation_timeout)
        except Exception as exc:
            candidate_id = self.experience.add_source(source, label, notes)
            result = EvaluationResult(candidate_id, float("-inf"), valid=False, error=str(exc), split=split)
            self.experience.write_result(result)
            return result if self.config.keep_invalid else None
        candidate_id = self.experience.add_source(source, label, notes)
        directory = self.experience.candidate_dir(candidate_id)
        harness_path = directory / "harness.py"
        runs = [self.evaluator.evaluate(harness_path, tasks, candidate_id,
                                        directory / ("traces.jsonl" if repeat == 0 else f"traces-{repeat}.jsonl"))
                for repeat in range(self.config.repeats)]
        result = self._merge_repeats(candidate_id, runs, split)
        self.experience.write_result(result)
        return result

    def _evaluate_many(self, work: Sequence[tuple[Path, str]], tasks: Sequence[Mapping[str, Any]],
                       split: str = "search") -> list[EvaluationResult]:
        if self.config.max_workers == 1 or len(work) <= 1:
            return [r for r in (self._evaluate_source(s, tasks, label, split) for s, label in work) if r]
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(work))) as pool:
            futures = [pool.submit(self._evaluate_source, source, tasks, label, split) for source, label in work]
            return [r for r in (future.result() for future in futures) if r]

    def evaluate_on_test(self, frontier: Sequence[EvaluationResult],
                         test_tasks: Sequence[Mapping[str, Any]]) -> list[EvaluationResult]:
        """Score the frontier once on held-out tasks. Paper section 3: the proposer never sees this."""
        test_root = Path(self.config.test_root)
        (test_root / "traces").mkdir(parents=True, exist_ok=True)
        results: list[EvaluationResult] = []
        for item in frontier:
            harness_path = self.experience.candidate_dir(item.candidate_id) / "harness.py"
            runs = []
            for repeat in range(self.config.repeats):
                name = f"{item.candidate_id}.jsonl" if repeat == 0 else f"{item.candidate_id}-{repeat}.jsonl"
                runs.append(self.evaluator.evaluate(harness_path, test_tasks, item.candidate_id,
                                                    test_root / "traces" / name))
            results.append(self._merge_repeats(item.candidate_id, runs, "test"))
        payload = {"search_root": str(self.experience.root), "results": [asdict(r) for r in results]}
        (test_root / "test_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return results

    def run(self, initial: Sequence[Path | str], search_tasks: Sequence[Mapping[str, Any]],
            test_tasks: Sequence[Mapping[str, Any]] | None = None) -> list[EvaluationResult]:
        self._write_run_manifest(search_tasks, test_tasks)
        results = self._evaluate_many([(Path(s), f"initial-{i:04d}") for i, s in enumerate(initial)], search_tasks)
        for iteration in range(1, self.config.iterations + 1):
            try:
                proposals = self.proposer(self.experience, iteration, self.config.candidates_per_iteration)
            except Exception as exc:
                proposal_dir = self.experience.root / "proposals" / f"iteration-{iteration:04d}"
                proposal_dir.mkdir(parents=True, exist_ok=True)
                (proposal_dir / "proposer.error").write_text(repr(exc), encoding="utf-8")
                proposals = []
            work = [(Path(s), f"iteration-{iteration:04d}-{i:02d}") for i, s in enumerate(proposals)]
            results.extend(self._evaluate_many(work, search_tasks))
        frontier = ParetoFrontier.select(results)
        (self.experience.root / "frontier.json").write_text(
            json.dumps([asdict(x) for x in frontier], indent=2), encoding="utf-8")
        if test_tasks:
            self.evaluate_on_test(frontier, test_tasks)
        return frontier

    def _write_run_manifest(self, tasks: Sequence[Mapping[str, Any]],
                            test_tasks: Sequence[Mapping[str, Any]] | None = None) -> None:
        manifest = {
            "created_at": time.time(),
            "python": sys.version,
            "config": {"root": str(self.config.root), "iterations": self.config.iterations,
                       "candidates_per_iteration": self.config.candidates_per_iteration,
                       "keep_invalid": self.config.keep_invalid, "repeats": self.config.repeats,
                       "max_workers": self.config.max_workers},
            "task_count": len(tasks),
            # Only the search split is written here. Test tasks stay out of the proposer's reach.
            "tasks": [dict(task) for task in tasks],
            "test_task_count": len(test_tasks or []),
        }
        (self.experience.root / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
