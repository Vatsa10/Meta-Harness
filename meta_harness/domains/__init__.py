"""Task domains.

Domain module contract:
    NAME: str
    PROMPT: str                      domain notes appended to the proposer prompt
    METRICS: dict[str, Callable[[Mapping, Mapping], float]]
    SEED_DIR: Path | None
    PROTOCOL: "plain" | "online"
    def adapt(records: list[dict]) -> list[dict]
"""

from __future__ import annotations

from types import ModuleType

from ..registry import discover


def all_domains() -> dict[str, ModuleType]:
    return discover("meta_harness.domains")


def get_domain(name: str) -> ModuleType:
    domains = all_domains()
    if name not in domains:
        raise KeyError(f"unknown domain {name!r}; known: {', '.join(sorted(domains))}")
    return domains[name]


def _terminal_adapt(records):
    from . import terminal
    return terminal.adapt(records)


def _terminal_metric(task, result):
    from . import terminal
    return terminal.metric(task, result)


def _maps():
    from ..datasets import classification_tasks, math_tasks
    from ..protocol import exact, numeric
    adapters = {"classification": classification_tasks, "math": math_tasks, "terminal": _terminal_adapt}
    metrics = {"classification": exact, "math": numeric, "terminal": _terminal_metric}
    return adapters, metrics


def adapter_for(task_type: str):
    """The record adapter shared by `search` and `sweep` for a task type."""
    adapters, _ = _maps()
    if task_type not in adapters:
        raise KeyError(f"unknown task type {task_type!r}; known: {', '.join(sorted(adapters))}")
    return adapters[task_type]


def metric_for(task_type: str):
    """The default (task, result) metric shared by `search` and `sweep` for a task type."""
    _, metrics = _maps()
    if task_type not in metrics:
        raise KeyError(f"unknown task type {task_type!r}; known: {', '.join(sorted(metrics))}")
    return metrics[task_type]
