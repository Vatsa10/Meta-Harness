"""Harness and metric contracts plus the online train-then-frozen-test protocol."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from .metrics import math_equivalence

TARGET_FIELDS = ("expected", "label", "answer", "target")
Metric = Callable[[Mapping, Mapping], float]


def expected_value(task: Mapping) -> Any:
    for key in TARGET_FIELDS:
        if key in task:
            return task[key]
    return None


def hide_target(task: Mapping) -> dict:
    return {k: v for k, v in task.items() if k not in TARGET_FIELDS}


def _positional(fn) -> list:
    return [p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]


def call_run(harness, task, model, trace) -> dict:
    value = harness.run(task, model, trace) if len(_positional(harness.run)) >= 3 else harness.run(task, model)
    return dict(value) if isinstance(value, Mapping) and "output" in value else {"output": value}


def call_learn(harness, task, result, model) -> None:
    learn = getattr(harness, "learn", None)
    if not callable(learn):
        return
    if len(_positional(learn)) >= 3:
        learn(task, result, model)
    else:
        learn(task, result)


def stream_train(harness, train_tasks, model, trace) -> None:
    for task in train_tasks:
        trace.event("train_task", {"input": task.get("input")})
        result = call_run(harness, task, model, trace)
        call_learn(harness, task, result, model)


def evaluate_frozen(harness, tasks, model, trace, metric) -> list[tuple[float, dict]]:
    out = []
    for task in tasks:
        frozen = copy.deepcopy(harness)
        result = call_run(frozen, hide_target(task), model, trace)
        out.append((float(metric(task, result)), result))
    return out


def _text(value: Any) -> str:
    return str(value).strip().lower()


def _target(task: Mapping) -> str | None:
    want = expected_value(task)
    return None if want is None or not str(want).strip() else str(want)


def exact(task: Mapping, result: Mapping) -> float:
    want = _target(task)
    return float(want is not None and _text(result.get("output", "")) == _text(want))


def contains(task: Mapping, result: Mapping) -> float:
    want = _target(task)
    return float(want is not None and _text(want) in str(result.get("output", "")).lower())


def numeric(task: Mapping, result: Mapping) -> float:
    want = _target(task)
    return 0.0 if want is None else math_equivalence(result.get("output", ""), {"answer": want})


BUILTIN_METRICS = {"exact": exact, "contains": contains, "numeric": numeric}


def _load_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"_metric_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load metric file {str(path)!r}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError:
        raise ValueError(f"metric file {str(path)!r} not found") from None
    except SyntaxError as exc:
        raise ValueError(f"cannot load metric file {str(path)!r}: SyntaxError: {exc.msg} (line {exc.lineno})") from exc
    return module


def resolve_metric(spec: str, base: Path | None = None, extra: Mapping[str, Metric] | None = None) -> Metric:
    if spec in BUILTIN_METRICS:
        return BUILTIN_METRICS[spec]
    if extra and spec in extra:
        return extra[spec]
    base = Path.cwd() if base is None else Path(base)
    target, sep, name = spec.rpartition(":")
    if not sep or not target or not name or (len(target) == 1 and target.isalpha()):
        raise ValueError(f"metric must be one of {sorted(BUILTIN_METRICS)} or module:function, got {spec!r}")
    if target.endswith(".py"):
        module = _load_file(base / target if not Path(target).is_absolute() else Path(target))
    else:
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        try:
            module = importlib.import_module(target)
        except ImportError as exc:
            raise ValueError(f"cannot import metric module for {spec!r}: {exc}") from exc
    fn = getattr(module, name, None)
    if not callable(fn):
        raise ValueError(f"{spec!r} is not a callable")
    return fn
