"""Scoring functions and token accounting shared by evaluators and harnesses."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping

_BOXED = re.compile(r"\\boxed\s*\{")
_FRAC = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_TEXTLIKE = re.compile(r"\\(?:text|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}")
_WRAPPERS = ("\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", "\\(", "\\)", "\\[", "\\]")
_ANSWER_PREFIX = re.compile(r"^(?:the\s+)?(?:final\s+)?answer(?:\s+is)?\s*[:=]?\s*", re.IGNORECASE)


def estimate_tokens(text: Any) -> int:
    """Cheap chars/4 proxy used when a provider reports no usage."""
    return math.ceil(len(str(text)) / 4)


def _extract_boxed(text: str) -> str:
    match = _BOXED.search(text)
    if not match:
        return text
    depth, start = 1, match.end()
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return text[start:]


def normalize_answer(text: Any) -> str:
    value = _extract_boxed(str(text)).strip()
    value = _ANSWER_PREFIX.sub("", value).strip()
    value = _FRAC.sub(r"\1/\2", value)
    value = _TEXTLIKE.sub(r"\1", value)
    for wrapper in _WRAPPERS:
        value = value.replace(wrapper, "")
    value = value.replace("\\%", "").replace("%", "")
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.rstrip(".")
    return value.lower()


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        pass
    if "/" in value:
        numerator, _, denominator = value.partition("/")
        try:
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def exact_match(prediction: Any, task: Mapping[str, Any]) -> float:
    return float(str(prediction).strip().lower() == str(task.get("label", "")).strip().lower())


def math_equivalence(prediction: Any, task: Mapping[str, Any], tolerance: float = 1e-6) -> float:
    if "answer" not in task:
        return 0.0
    left, right = normalize_answer(prediction), normalize_answer(task["answer"])
    if left and left == right:
        return 1.0
    left_value, right_value = _as_float(left), _as_float(right)
    if left_value is not None and right_value is not None:
        return float(abs(left_value - right_value) <= tolerance * max(1.0, abs(right_value)))
    return 0.0


def terminal_metric(task: Any, result: Any = None, timeout: float | None = None) -> float:
    """Score a terminal task by running its `test_command` INSIDE the task's container.

    Requires `task["container"]` (the id the terminal domain started and the harness worked in):
    `docker exec -w <workdir> <id> bash -lc <test_command>`; exit code 0 scores 1.0. With no
    container, or no test command, the score is 0.0 and nothing runs - never on the host.
    The legacy `(prediction, task)` argument order is still accepted.
    """
    if not isinstance(task, Mapping) and isinstance(result, Mapping):
        task, result = result, task  # legacy (prediction, task)
    if not isinstance(task, Mapping):
        return 0.0
    command = str(task.get("test_command") or "").strip()
    container = task.get("container")
    if not command or not container:
        return 0.0
    from .terminal import Container  # lazy: terminal imports this module

    limit = float(timeout if timeout is not None else task.get("test_timeout") or 600.0)
    try:
        code, _ = Container.attach(str(container), str(task.get("workdir") or "/app")).exec_test(command, limit)
    except (OSError, RuntimeError, ValueError):
        return 0.0
    return float(code == 0)


def __getattr__(name: str) -> Any:
    # METRICS is protocol.BUILTIN_METRICS, (task, result) metrics; resolved lazily because
    # protocol imports this module.
    if name == "METRICS":
        from .protocol import BUILTIN_METRICS
        return BUILTIN_METRICS
    raise AttributeError(name)


__all__ = ["METRICS", "estimate_tokens", "exact_match", "math_equivalence",
           "normalize_answer", "terminal_metric"]
