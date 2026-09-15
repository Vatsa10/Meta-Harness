"""Scoring functions and token accounting shared by evaluators and harnesses."""

from __future__ import annotations

import math
import re
import subprocess
from typing import Any, Callable, Mapping

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


def terminal_metric(prediction: Any, task: Mapping[str, Any], timeout: float = 120.0) -> float:
    """Run the task's verification command. An empty command scores zero, never one."""
    command = str(task.get("test_command", "")).strip()
    if not command:
        return 0.0
    workdir = task.get("workdir")
    container = task.get("container")
    argv: Any = (["docker", "exec", "-w", str(workdir or "/app"), str(container), "bash", "-lc", command]
                 if container else command)
    try:
        completed = subprocess.run(argv, shell=not container, capture_output=True, text=True,
                                   timeout=timeout, cwd=None if container else (workdir or None))
    except (OSError, subprocess.SubprocessError):
        return 0.0
    return float(completed.returncode == 0)


METRICS: dict[str, Callable[[Any, Mapping[str, Any]], float]] = {
    "classification": exact_match,
    "math": math_equivalence,
    "terminal": terminal_metric,
}

__all__ = ["METRICS", "estimate_tokens", "exact_match", "math_equivalence", "normalize_answer",
           "terminal_metric"]
