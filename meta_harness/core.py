"""The Meta-Harness outer loop.

The implementation keeps the proposer deliberately unopinionated: it receives
an experience directory and may inspect any prior candidate, score, or trace.
That is the important design choice in the paper; parent selection and edit
operators are not hard-coded here.

Every step's result is a file (proposals, `candidates/<id>/scores.json`, `receipt.json`, test
results) and `run.json` is checkpointed atomically, so a killed search resumes where it stopped.
"""

from __future__ import annotations

import copy
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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import protocol as _protocol
from .metrics import estimate_tokens
from .redact import bounded, redact_obj
from .timeouts import Aborted, Watchdog

# Addendum A idea 4: every task ends with one of these; scores.json carries the histogram.
STOP_REASONS = ("done", "step_limit", "overflow", "stuck", "timeout", "error")
_STOP_ALIASES = {"end": "done", "stop": "done", "finished": "done", "length": "overflow",
                 "max_tokens": "overflow", "context_overflow": "overflow", "max_steps": "step_limit",
                 "budget": "step_limit", "aborted": "timeout", "idle": "timeout", "deadline": "timeout"}
VERDICTS = ("accepted", "rejected_invalid_source", "rejected_split_violation", "stale_parent",
            "budget_expired", "failed_eval", "incomplete")
USAGE_KEYS = ("input", "output", "cache_read", "cache_write", "reasoning")
MAX_AUTOMATIC_RESUMES = 3


def normalise_stop_reason(value: Any) -> str:
    """Map a harness-reported stop reason onto STOP_REASONS; no report means `done`."""
    if value is None:
        return "done"
    text = str(value).strip().lower()
    if text in STOP_REASONS:
        return text
    return _STOP_ALIASES.get(text, "error")


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory and `replace`, so readers never see a torn file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(text, encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temp, path)
            return
        except PermissionError:  # Windows: a reader holds the target open
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


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
    stop_reasons: dict[str, int] = field(default_factory=dict)
    abort_reason: str | None = None

    def objective_vector(self) -> tuple[float, float]:
        # Higher score and lower context cost are both preferred.
        return (self.score, -self.context_cost)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvaluationResult":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def _usage_record() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


@dataclass
class Budget:
    harness_evaluations: int = 0      # one per (candidate, repeat) on the search split
    test_evaluations: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    train_prompt_tokens: int = 0      # online protocol: tokens spent streaming the train split
    proposer_calls: int = 0
    proposer_input_tokens: int = 0
    proposer_output_tokens: int = 0
    proposals: int = 0
    invalid_proposals: int = 0
    # Usage totals {input, output, cache_read, cache_write, reasoning}, base model vs proposer.
    model_usage: dict[str, int] = field(default_factory=_usage_record)
    proposer_usage: dict[str, int] = field(default_factory=_usage_record)
    cost_usd: float | None = None     # billed cost; None when a price is unknown

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "Budget":
        known = {f.name for f in fields(cls)}
        budget = cls(**{k: v for k, v in dict(data or {}).items() if k in known})
        budget.model_usage = {**_usage_record(), **dict(budget.model_usage or {})}
        budget.proposer_usage = {**_usage_record(), **dict(budget.proposer_usage or {})}
        return budget

    def add_run(self, metrics: Mapping[str, float], split: str) -> None:
        if split == "search":
            self.harness_evaluations += 1
        else:
            self.test_evaluations += 1
        prompt = int(metrics.get("prompt_tokens", 0) or 0)
        completion = int(metrics.get("completion_tokens", 0) or 0)
        self.model_calls += int(metrics.get("model_calls", 0) or 0)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.train_prompt_tokens += int(metrics.get("train_prompt_tokens", 0) or 0)
        self.model_usage["input"] += prompt
        self.model_usage["output"] += completion
        for key in ("cache_read", "cache_write", "reasoning"):
            self.model_usage[key] += int(metrics.get(f"{key}_tokens", 0) or 0)

    def add_proposer(self, delta: Mapping[str, int]) -> None:
        self.proposer_calls += int(delta.get("calls", 0) or 0)
        self.proposer_input_tokens += int(delta.get("input_tokens", 0) or 0)
        self.proposer_output_tokens += int(delta.get("output_tokens", 0) or 0)
        self.proposer_usage["input"] += int(delta.get("input_tokens", 0) or 0)
        self.proposer_usage["output"] += int(delta.get("output_tokens", 0) or 0)
        for key in ("cache_read", "cache_write", "reasoning"):
            self.proposer_usage[key] += int(delta.get(key, 0) or 0)

    def price(self, model: str | None, proposer_model: str | None) -> float | None:
        """Billed USD for both models, or None when any model that was used has no known price."""
        from .chat_types import Usage
        from .providers import cost_usd, price_for

        total = 0.0
        for name, usage, used in ((model, self.model_usage, self.model_calls),
                                  (proposer_model, self.proposer_usage, self.proposer_calls)):
            if not used:
                continue
            cost = cost_usd(Usage(**usage), price_for(name)) if name else None
            if cost is None:
                return None
            total += cost
        return total


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
    max_evaluations: int | None = None   # equal-budget cap over harness_evaluations (search split)
    run_meta: dict = field(default_factory=dict)   # provider/model names, optimizer, view (never keys)
    evaluation_timeout: float | None = None       # per evaluation; default task_timeout * len(tasks)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        if self.iterations < 0 or self.candidates_per_iteration < 1:
            raise ValueError("iterations must be non-negative and candidates_per_iteration positive")
        if self.repeats < 1 or self.max_workers < 1:
            raise ValueError("repeats and max_workers must be positive")
        if self.max_evaluations is not None and self.max_evaluations < 0:
            raise ValueError("max_evaluations must be non-negative")
        self.test_root = Path(self.test_root) if self.test_root else self.root.with_name(self.root.name + "-test")
        root, test = Path(os.path.abspath(self.root)), Path(os.path.abspath(self.test_root))
        if test == root or root in test.parents:
            raise ValueError(f"test_root must be outside root: {self.test_root} is inside {self.root}")


class TraceRecorder:
    """JSONL trace. Payloads are redacted and string fields bounded: no key reaches a trace."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self.closed = False
        self.count = 0
        # Metered prompt tokens so far (kept up to date by MeteredModel), for cost-aware harnesses.
        self.context_cost = 0.0

    def add_context_cost(self, amount: float) -> None:
        """Legacy hook: cost is metered once, in provider tokens, by MeteredModel."""
        self.event("harness_reported_cost", {"amount": amount})

    def event(self, name: str, payload: Any = None, **fields: Any) -> None:
        clean = redact_obj(json.loads(json.dumps(payload, ensure_ascii=False, default=str)))
        extra = {key: bounded(value) if isinstance(value, str)
                 else redact_obj(json.loads(json.dumps(value, ensure_ascii=False, default=str)))
                 for key, value in fields.items()}
        record = {"time": time.time(), "event": name, "payload": clean, **extra}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            if self.closed:  # a timed-out harness thread may still be running
                return
            self._file.write(line)
            self._file.flush()
            self.count += 1

    def close(self) -> None:
        with self._lock:
            if not self.closed:
                self.closed = True
                self._file.close()

    def __enter__(self) -> "TraceRecorder":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class EvaluationClosed(RuntimeError):
    """A model call arrived after its evaluation ended (a timed-out harness thread)."""


class MeteredModel:
    """Wraps the base model so every call is priced in tokens and written into the trace."""

    def __init__(self, model: Callable[..., Any], trace: TraceRecorder, sample_index: int = 0):
        self.model = model
        self.trace = trace
        self.sample_index = sample_index
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.usage = _usage_record()
        self.closed = False
        self._lock = threading.Lock()

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
        if self.closed:
            raise EvaluationClosed("model call after the evaluation ended")
        upstream = getattr(self.model, "call", None)
        if callable(upstream):
            if getattr(self.model, "supports_sample_index", False):
                kwargs = {**kwargs, "sample_index": self.sample_index}
            text, usage = upstream(prompt, **kwargs)
            text, usage = str(text), dict(usage or {})
        else:
            text, usage = str(self.model(prompt, **kwargs)), {}
        tokens = int(usage.get("prompt_tokens") or usage.get("input") or 0) or estimate_tokens(prompt)
        completion = int(usage.get("completion_tokens") or usage.get("output") or 0)
        with self._lock:
            if self.closed:
                raise EvaluationClosed("model call after the evaluation ended")
            self.calls += 1
            self.prompt_tokens += tokens
            self.completion_tokens += completion
            self.usage["input"] += tokens
            self.usage["output"] += completion
            for key in ("cache_read", "cache_write", "reasoning"):
                self.usage[key] += int(usage.get(key) or 0)
            self.trace.context_cost = float(self.prompt_tokens)
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
                out.append(EvaluationResult.from_dict(json.loads(path.read_text(encoding="utf-8"))))
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


def _bounded_call(watchdog: Watchdog, fn: Callable[[], Any]) -> tuple[Any, str | None, str | None]:
    """Run fn in a daemon thread under the watchdog: (value, error, abort reason).

    A harness cannot be killed in-thread; a timed-out thread is abandoned and its model is
    closed when the evaluation ends, so it can neither spend budget nor write the trace.
    """
    watchdog.kick()
    try:
        watchdog.check()
    except Aborted as exc:
        return None, f"timeout: {exc.reason}", exc.reason
    box: dict[str, Any] = {}

    def target() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - a harness failure is a task score
            box["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(watchdog.remaining())
    if thread.is_alive():
        reason = "idle"
        try:
            watchdog.check()
        except Aborted as exc:
            reason = exc.reason
        return None, f"timeout: no result before the {reason} limit", reason
    return box.get("value"), box.get("error"), None


class CandidateEvaluator:
    """Scores one harness on a task list.

    plain : per task  result = call_run(...); score = metric(task, result); call_learn(...)
    online: fresh harness -> stream the train tasks (run + learn) -> each task scored on a frozen
            copy with its target hidden.
    A harness exception or timeout on one task scores that task 0 and is never retried; only an
    import failure makes the evaluation invalid.
    """

    def __init__(self, model: Callable[..., Any], metric: _protocol.Metric | None = None,
                 task_timeout: float = 600.0, protocol: str = "plain",
                 train_tasks: Sequence[Mapping[str, Any]] = (), evaluation_timeout: float | None = None):
        if protocol not in ("plain", "online"):
            raise ValueError("protocol must be 'plain' or 'online'")
        self.model = model
        self.metric = metric or _protocol.exact
        self.task_timeout = task_timeout
        self.protocol = protocol
        self.train_tasks = [dict(t) for t in train_tasks]
        self.evaluation_timeout = evaluation_timeout

    def validate(self, source: Path | str, probe: Mapping[str, Any] | None = None) -> None:
        harness = _load_harness(Path(source))
        row = dict(probe) if probe is not None else {"input": "validation"}
        with tempfile.TemporaryDirectory() as td:
            with TraceRecorder(Path(td) / "validation.jsonl") as trace:
                try:
                    _protocol.call_run(harness, row, MeteredModel(self.model, trace), trace)
                except Exception as exc:
                    raise HarnessValidationError(f"validation run failed: {exc}") from exc

    def _plain(self, harness: Any, task: Mapping[str, Any], model: MeteredModel,
               trace: TraceRecorder) -> tuple[float, dict]:
        result = _protocol.call_run(harness, task, model, trace)
        reward = float(self.metric(task, result))
        try:
            _protocol.call_learn(harness, task, result, model)
        except Exception as exc:  # the score stands; the learning failure is visible in the trace
            trace.event("learn_error", None, error=f"{type(exc).__name__}: {exc}")
        return reward, result

    def _frozen(self, harness: Any, task: Mapping[str, Any], model: MeteredModel,
                trace: TraceRecorder) -> tuple[float, dict]:
        frozen = copy.deepcopy(harness)
        result = _protocol.call_run(frozen, _protocol.hide_target(task), model, trace)
        return float(self.metric(task, result)), result

    def _train(self, harness: Any, task: Mapping[str, Any], model: MeteredModel, trace: TraceRecorder) -> None:
        result = _protocol.call_run(harness, task, model, trace)
        _protocol.call_learn(harness, task, result, model)

    def evaluate(self, source: Path | str, tasks: Sequence[Mapping[str, Any]], candidate_id: str,
                 trace_path: Path, sample_index: int = 0) -> EvaluationResult:
        started = time.perf_counter()
        try:
            harness = _load_harness(Path(source))
        except Exception as exc:
            return EvaluationResult(candidate_id, float("-inf"), {}, float("inf"), len(tasks),
                                    time.perf_counter() - started, False, repr(exc))
        online = self.protocol == "online"
        steps = len(tasks) + (len(self.train_tasks) if online else 0)
        deadline = self.evaluation_timeout or self.task_timeout * max(1, steps)
        watchdog = Watchdog(idle=self.task_timeout, deadline=deadline)
        rewards: list[float] = []
        stops: Counter[str] = Counter()
        failures = 0
        abort: str | None = None
        train_tokens = 0
        with TraceRecorder(trace_path) as trace:
            metered = MeteredModel(self.model, trace, sample_index)
            try:
                if online:
                    for index, task in enumerate(self.train_tasks):
                        trace.event("train_task", {"index": index, "input": task.get("input")})
                        _, error, reason = _bounded_call(
                            watchdog, lambda task=task: self._train(harness, task, metered, trace))
                        if error:
                            trace.event("train_error", {"index": index}, error=error)
                        if reason == "deadline":
                            abort = abort or reason
                            break
                    train_tokens = metered.prompt_tokens
                run_one = self._frozen if online else self._plain
                for index, task in enumerate(tasks):
                    trace.event("task_start", {"index": index, "task": dict(task)})
                    if abort == "deadline":
                        value, error, reason = None, "timeout: evaluation deadline already passed", "deadline"
                    else:
                        value, error, reason = _bounded_call(
                            watchdog, lambda task=task: run_one(harness, task, metered, trace))
                    if reason:
                        abort = abort if abort == "deadline" else reason
                    if error is None and value is not None:
                        reward, result = value
                        stop = normalise_stop_reason(result.get("stop_reason") if isinstance(result, Mapping) else None)
                        trace.event("task_end", {"index": index, "output": result.get("output"),
                                                 "reward": reward, "stop_reason": stop})
                    else:
                        reward, stop = 0.0, "timeout" if reason else "error"
                        failures += 1
                        trace.event("task_end", {"index": index, "output": None, "reward": 0.0,
                                                 "stop_reason": stop}, error=error or "no result")
                    rewards.append(reward)
                    stops[stop] += 1
            finally:
                metered.closed = True
        n = len(tasks) or 1
        eval_tokens = metered.prompt_tokens - train_tokens
        return EvaluationResult(
            candidate_id, sum(rewards) / n,
            {"model_calls": float(metered.calls), "prompt_tokens": metered.prompt_tokens,
             "completion_tokens": metered.completion_tokens, "train_prompt_tokens": train_tokens,
             "cache_read_tokens": metered.usage["cache_read"],
             "cache_write_tokens": metered.usage["cache_write"],
             "reasoning_tokens": metered.usage["reasoning"], "task_failures": failures},
            eval_tokens / n, len(tasks), time.perf_counter() - started,
            stop_reasons=dict(stops), abort_reason=abort)


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
                                   capture_output=True, timeout=self.timeout,
                                   encoding="utf-8", errors="replace")
        (out / "proposer.stdout").write_text(completed.stdout or "", encoding="utf-8")
        (out / "proposer.stderr").write_text(completed.stderr or "", encoding="utf-8")
        if completed.returncode:
            raise RuntimeError(f"proposer exited with {completed.returncode}")
        return sorted(out.glob("*.py"))[:count]


class SearchRunner:
    # proposal sidecars written next to <stem>.py, copied into the candidate directory
    SIDECARS = {".reasoning.md": "proposer_reasoning.md", ".proposer_log.jsonl": "proposer_log.jsonl",
                ".outcomes.json": "outcomes.json", ".parent.json": "parent.json"}

    def __init__(self, config: SearchConfig, evaluator: CandidateEvaluator,
                 proposer: Callable[[FilesystemExperience, int, int], Sequence[Path]]):
        self.config, self.evaluator, self.proposer = config, evaluator, proposer
        self.experience = FilesystemExperience(config.root)
        self.probe: dict[str, Any] | None = None
        self.budget = Budget()
        self.cursor = 0
        self._lock = threading.RLock()
        self._results: dict[str, EvaluationResult] = {}   # candidate id -> merged search result
        self._verdicts: dict[str, str] = {}
        self._resuming = False
        self._proposer_seen: dict[str, int] = {}
        self._manifest: dict[str, Any] = {}

    # -- proposal files ------------------------------------------------------------------------

    @classmethod
    def _sidecar_notes(cls, source: Path) -> dict[str, str]:
        notes = {}
        for suffix, name in cls.SIDECARS.items():
            sidecar = source.with_name(source.stem + suffix)
            if sidecar.is_file():
                notes[name] = sidecar.read_text(encoding="utf-8", errors="replace")
        return notes

    @staticmethod
    def _json_sidecar(source: Path, suffix: str) -> dict[str, Any]:
        path = source.with_name(source.stem + suffix)
        try:
            data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    # -- receipts and the acceptance gate ------------------------------------------------------

    @staticmethod
    def _write_receipt(directory: Path, receipt: Mapping[str, Any]) -> None:
        atomic_write(directory / "receipt.json", json.dumps(redact_obj(dict(receipt)), indent=2, default=str))

    @staticmethod
    def read_receipt(directory: Path) -> dict[str, Any] | None:
        try:
            return json.loads((Path(directory) / "receipt.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _test_root_spellings(self) -> list[str]:
        test_root = Path(self.config.test_root)
        spellings = set()
        for form in (str(test_root), os.path.abspath(test_root), str(test_root.resolve())):
            spellings.update({form, form.replace("\\", "/"), form.replace("\\", "\\\\")})
        return [s for s in spellings if s]

    def _references_test_root(self, text: str) -> bool:
        haystack = text.lower() if os.name == "nt" else text
        return any((s.lower() if os.name == "nt" else s) in haystack for s in self._test_root_spellings())

    def _references_test_data(self, text: str) -> bool:
        return Path(self.config.test_root).name in text or "test_results" in text

    def _gate(self, text: str, outcomes: Mapping[str, Any], proposal: Mapping[str, Any],
              fingerprint_now: str | None, over_budget: bool) -> tuple[str | None, str | None]:
        """Host re-validation before a proposal becomes a candidate: (verdict, detail) or (None, None)."""
        try:
            compile(text, "harness.py", "exec")
        except (SyntaxError, ValueError) as exc:
            return "rejected_invalid_source", f"source does not compile: {exc}"
        if self._references_test_root(text):
            return "rejected_split_violation", "source contains the test-split path"
        if int(outcomes.get("denied_split", 0) or 0) > 0 and self._references_test_data(text):
            return "rejected_split_violation", "proposer was denied the test split and the source references test data"
        claimed = proposal.get("fingerprint")
        if claimed and fingerprint_now and claimed != fingerprint_now:
            return "stale_parent", f"experience changed since the proposal read it ({claimed} != {fingerprint_now})"
        if over_budget:
            return "budget_expired", "equal-budget cap reached before this proposal was evaluated"
        return None, None

    # -- evaluation ----------------------------------------------------------------------------

    @staticmethod
    def _merge_repeats(candidate_id: str, runs: Sequence[EvaluationResult], split: str) -> EvaluationResult:
        stops: Counter[str] = Counter()
        for run in runs:
            stops.update(run.stop_reasons)
        abort = next((run.abort_reason for run in runs if run.abort_reason), None)
        failed = next((run for run in runs if not run.valid), None)
        if failed is not None:
            return EvaluationResult(candidate_id, float("-inf"), {}, float("inf"), failed.examples,
                                    sum(run.elapsed_seconds for run in runs), False, failed.error,
                                    len(runs), 0.0, split, dict(stops), abort)
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
            dict(stops),
            abort,
        )

    def _finished(self, label: str) -> EvaluationResult | None:
        """On resume: the stored result of a finished candidate, or None (an incomplete one is wiped)."""
        if not self._resuming:
            return None
        directory = self.experience.candidate_dir(label)
        if not directory.exists():
            return None
        receipt = self.read_receipt(directory) or {}
        scores = directory / "scores.json"
        if receipt.get("verdict") not in (None, "incomplete") and scores.is_file():
            try:
                result = EvaluationResult.from_dict(json.loads(scores.read_text(encoding="utf-8")))
            except (OSError, TypeError, ValueError):
                result = None
            if result is not None:
                with self._lock:
                    self._results[label] = result
                    self._verdicts[label] = receipt["verdict"]
                return result
        shutil.rmtree(directory, ignore_errors=True)  # torn step: evaluate again
        return None

    def _reject(self, candidate_id: str, directory: Path, receipt: dict[str, Any], verdict: str,
                detail: str, split: str) -> EvaluationResult | None:
        result = EvaluationResult(candidate_id, float("-inf"), valid=False, error=f"{verdict}: {detail}",
                                  split=split)
        self.experience.write_result(result)
        receipt["verdict"] = verdict
        receipt["fallback"] = detail
        self._write_receipt(directory, receipt)
        with self._lock:
            self._verdicts[candidate_id] = verdict
            if verdict in ("rejected_invalid_source", "rejected_split_violation"):
                self.budget.invalid_proposals += 1
            self._results[candidate_id] = result
            self._checkpoint()
        return result if self.config.keep_invalid else None

    def _evaluate_source(self, source: Path, tasks: Sequence[Mapping[str, Any]], label: str,
                         split: str = "search", fingerprint_now: str | None = None,
                         over_budget: bool = False) -> EvaluationResult | None:
        from .sandbox import validate_in_subprocess

        done = self._finished(label)
        if done is not None:
            return done if done.valid or self.config.keep_invalid else None
        source = Path(source)
        notes = self._sidecar_notes(source)
        proposal = self._json_sidecar(source, ".proposal.json")
        outcomes = self._json_sidecar(source, ".outcomes.json")
        candidate_id = self.experience.add_source(source, label, notes)
        directory = self.experience.candidate_dir(candidate_id)
        if proposal:
            atomic_write(directory / "proposal.json",
                         json.dumps(redact_obj({"candidate_id": candidate_id, **proposal}), indent=2, default=str))
        receipt: dict[str, Any] = {
            "candidate_id": candidate_id, "proposal_id": proposal.get("candidate") or source.stem,
            "split": split, "parents_read": list(proposal.get("parents_read") or []),
            "tool_outcomes": outcomes, "fingerprint": proposal.get("fingerprint"),
            "validation": None, "evaluation": None, "fallback": None, "verdict": "incomplete"}
        self._write_receipt(directory, receipt)
        text = (directory / "harness.py").read_text(encoding="utf-8", errors="replace")
        verdict, detail = self._gate(text, outcomes, proposal, fingerprint_now, over_budget)
        if verdict:
            return self._reject(candidate_id, directory, receipt, verdict, detail or "", split)
        try:
            validate_in_subprocess(directory / "harness.py", self.config.validation_timeout, self.probe,
                                   self.experience.root)
        except Exception as exc:
            receipt["validation"] = {"ok": False, "error": bounded(str(exc))}
            return self._reject(candidate_id, directory, receipt, "rejected_invalid_source",
                                f"validation failed: {exc}", split)
        receipt["validation"] = {"ok": True}
        self._write_receipt(directory, receipt)
        harness_path = directory / "harness.py"
        runs = []
        for repeat in range(self.config.repeats):
            run = self.evaluator.evaluate(harness_path, tasks, candidate_id,
                                          directory / ("traces.jsonl" if repeat == 0 else f"traces-{repeat}.jsonl"),
                                          sample_index=repeat)
            runs.append(run)
            with self._lock:
                self.budget.add_run(run.metrics, split)
        result = self._merge_repeats(candidate_id, runs, split)
        self.experience.write_result(result)
        receipt["evaluation"] = {"repeats": len(runs), "score": result.score, "context_cost": result.context_cost,
                                 "task_failures": result.metrics.get("task_failures", 0.0),
                                 "stop_reasons": result.stop_reasons, "abort_reason": result.abort_reason,
                                 "error": result.error}
        receipt["verdict"] = "accepted" if result.valid else "failed_eval"
        self._write_receipt(directory, receipt)
        with self._lock:
            self._verdicts[candidate_id] = receipt["verdict"]
            self._results[candidate_id] = result
            self._checkpoint()
        return result if result.valid or self.config.keep_invalid else None

    def _evaluate_many(self, work: Sequence[tuple[Path, str]], tasks: Sequence[Mapping[str, Any]],
                       split: str = "search", fingerprint_now: str | None = None,
                       over_budget: Sequence[bool] = ()) -> list[EvaluationResult]:
        flags = list(over_budget) + [False] * (len(work) - len(over_budget))
        jobs = [(s, label, flag) for (s, label), flag in zip(work, flags)]
        if self.config.max_workers == 1 or len(jobs) <= 1:
            return [r for r in (self._evaluate_source(s, tasks, label, split, fingerprint_now, flag)
                                for s, label, flag in jobs) if r]
        with ThreadPoolExecutor(max_workers=min(self.config.max_workers, len(jobs))) as pool:
            futures = [pool.submit(self._evaluate_source, s, tasks, label, split, fingerprint_now, flag)
                       for s, label, flag in jobs]
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
                run = self.evaluator.evaluate(harness_path, test_tasks, item.candidate_id,
                                              test_root / "traces" / name, sample_index=repeat)
                runs.append(run)
                with self._lock:
                    self.budget.add_run(run.metrics, "test")
            results.append(self._merge_repeats(item.candidate_id, runs, "test"))
        payload = {"search_root": str(self.experience.root), "results": [asdict(r) for r in results]}
        atomic_write(test_root / "test_results.json", json.dumps(payload, indent=2))
        return results

    # -- budget, checkpoint, resume ------------------------------------------------------------

    def _remaining(self) -> int | None:
        cap = self.config.max_evaluations
        return None if cap is None else max(0, cap - self.budget.harness_evaluations)

    def _absorb_proposer_usage(self) -> None:
        usage = dict(getattr(self.proposer, "usage", None) or {})
        delta = {key: int(value or 0) - self._proposer_seen.get(key, 0)
                 for key, value in usage.items() if isinstance(value, (int, float))}
        self._proposer_seen = {key: int(value or 0) for key, value in usage.items()
                               if isinstance(value, (int, float))}
        with self._lock:
            self.budget.add_proposer(delta)

    def frontier(self) -> list[EvaluationResult]:
        with self._lock:
            accepted = [r for cid, r in self._results.items() if self._verdicts.get(cid) == "accepted"]
        return ParetoFrontier.select(accepted)

    def _checkpoint(self) -> None:
        """Atomically rewrite run.json with the cursor, frontier ids and budget."""
        with self._lock:
            meta = redact_obj(dict(self.config.run_meta))
            self.budget.cost_usd = self.budget.price(meta.get("model"), meta.get("proposer_model"))
            payload = {**self._manifest, "run_meta": meta,
                       "cursor": {"iteration": self.cursor},
                       "frontier": [r.candidate_id for r in self.frontier()],
                       "budget": asdict(self.budget)}
            atomic_write(self.experience.root / "run.json", json.dumps(payload, indent=2, default=str))

    def _begin_resume(self, force: bool) -> None:
        root = self.experience.root
        counter = root / ".resume_attempts"
        try:
            attempts = int(counter.read_text(encoding="utf-8").strip() or 0)
        except (OSError, ValueError):
            attempts = 0
        attempts += 1
        if attempts > MAX_AUTOMATIC_RESUMES and not force:
            raise RuntimeError(f"{root} was already resumed {attempts - 1} times automatically "
                               f"(limit {MAX_AUTOMATIC_RESUMES}); inspect it and use --force-resume")
        atomic_write(counter, str(attempts))
        try:
            previous = json.loads((root / "run.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        self.budget = Budget.from_dict(previous.get("budget"))
        self.cursor = int((previous.get("cursor") or {}).get("iteration", 0) or 0)
        shutil.rmtree(root / "views", ignore_errors=True)   # derived; rebuilt on demand
        self._repair_logs()
        self._resuming = True
        for directory in sorted(self.experience.candidates.iterdir()):
            if directory.is_dir():
                self._finished(directory.name)

    def _repair_logs(self) -> None:
        """Drop a torn trailing line from traces and proposer logs left by a killed process."""
        from .httpdata import read_jsonl, write_jsonl

        root = self.experience.root
        paths = [*root.glob("candidates/*/traces*.jsonl"), *root.glob("candidates/*/proposer_log.jsonl"),
                 *root.glob("proposals/*/*.proposer_log.jsonl")]
        for path in paths:
            try:
                lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                         if line.strip()]
                rows = read_jsonl(path, tolerate_torn=True)
            except (OSError, ValueError):
                continue
            if len(rows) != len(lines):
                write_jsonl(path, rows)

    @classmethod
    def _persist_proposals(cls, proposals: Sequence[Path | str], proposal_dir: Path) -> list[Path]:
        """Every proposal is a file under proposals/iteration-NNNN/ (with its sidecars), so a resume
        reuses it instead of proposing again."""
        proposal_dir.mkdir(parents=True, exist_ok=True)
        kept: list[Path] = []
        for item in proposals:
            source = Path(item)
            if source.parent.resolve() == proposal_dir.resolve():
                kept.append(source)
                continue
            target = proposal_dir / source.name
            shutil.copy2(source, target)
            for suffix in (*cls.SIDECARS, ".proposal.json"):
                sidecar = source.with_name(source.stem + suffix)
                if sidecar.is_file():
                    shutil.copy2(sidecar, proposal_dir / sidecar.name)
            kept.append(target)
        return kept

    # -- the loop ------------------------------------------------------------------------------

    def run(self, initial: Sequence[Path | str], search_tasks: Sequence[Mapping[str, Any]],
            test_tasks: Sequence[Mapping[str, Any]] | None = None, resume: bool = False,
            force_resume: bool = False) -> list[EvaluationResult]:
        from .proposer_agent import fingerprint

        if self.config.evaluation_timeout is not None:
            self.evaluator.evaluation_timeout = self.config.evaluation_timeout
        if resume:
            self._begin_resume(force_resume)
        self.probe = _protocol.hide_target(search_tasks[0]) if search_tasks else None
        self._manifest = self._run_manifest(search_tasks, test_tasks)
        self._checkpoint()
        self._evaluate_many([(Path(s), f"initial-{i:04d}") for i, s in enumerate(initial)], search_tasks)
        count = self.config.candidates_per_iteration
        for iteration in range(1, self.config.iterations + 1):
            remaining = self._remaining()
            if remaining is not None and remaining < self.config.repeats:
                break
            proposal_dir = self.experience.root / "proposals" / f"iteration-{iteration:04d}"
            existing = sorted(proposal_dir.glob("*.py"))[:count] if resume and proposal_dir.is_dir() else []
            if existing:
                proposals: Sequence[Path] = existing
            else:
                try:
                    proposals = self.proposer(self.experience, iteration, count)
                except Exception as exc:
                    proposal_dir.mkdir(parents=True, exist_ok=True)
                    (proposal_dir / "proposer.error").write_text(repr(exc), encoding="utf-8")
                    proposals = []
                finally:
                    self._absorb_proposer_usage()
                proposals = self._persist_proposals(proposals, proposal_dir)
                with self._lock:
                    self.budget.proposals += len(proposals)
            work = [(Path(s), f"iteration-{iteration:04d}-{i:02d}") for i, s in enumerate(proposals)]
            fits = len(work) if remaining is None else remaining // self.config.repeats
            over = [index >= fits for index in range(len(work))]
            self._evaluate_many(work, search_tasks, fingerprint_now=fingerprint(self.experience.root),
                                over_budget=over)
            with self._lock:
                self.cursor = max(self.cursor, iteration)
                self._checkpoint()
        frontier = self.frontier()
        atomic_write(self.experience.root / "frontier.json",
                     json.dumps([asdict(x) for x in frontier], indent=2))
        if test_tasks:
            self.evaluate_on_test(frontier, test_tasks)
        self._checkpoint()
        return frontier

    def _run_manifest(self, tasks: Sequence[Mapping[str, Any]],
                      test_tasks: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        return {
            "created_at": time.time(),
            "python": sys.version,
            "config": {"root": str(self.config.root), "iterations": self.config.iterations,
                       "candidates_per_iteration": self.config.candidates_per_iteration,
                       "keep_invalid": self.config.keep_invalid, "repeats": self.config.repeats,
                       "max_workers": self.config.max_workers,
                       "max_evaluations": self.config.max_evaluations,
                       "protocol": getattr(self.evaluator, "protocol", "plain")},
            "task_count": len(tasks),
            # Only the search split is written here. Test tasks stay out of the proposer's reach.
            "tasks": [dict(task) for task in tasks],
            "test_task_count": len(test_tasks or []),
        }
