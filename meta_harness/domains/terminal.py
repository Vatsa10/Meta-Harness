"""TerminalBench-2 style agentic coding (paper section 4.3, spec section 8).

Evaluation of one task is owned by this domain, never by the harness:

    start container (task image, network only if the task allows it)
    -> put its id on the task as task["container"]
    -> harness.run(task, model, trace)          (works inside that container)
    -> copy the task's tests in, then terminal_metric(task, result) runs test_command
       INSIDE the same container
    -> remove the container (also when the harness raises)

The harness never sees `test_command`, the tests or the solution. Search on this domain uses
search = test, as the paper does (`--test` equal to `--search`); report numbers accordingly.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from ..metrics import terminal_metric
from ..protocol import call_run
from ..terminal import Container

NAME = "terminal"
PROTOCOL = "plain"
SEARCH_EQUALS_TEST = True
SEED_DIR = Path(__file__).resolve().parents[2] / "baselines" / "terminal"
HIDDEN_FIELDS = ("test_command", "test_files", "tests_dir", "expected_outputs", "solution",
                 "dockerfile_dir")
PROMPT = """Terminal domain (TerminalBench-2). Seeds: baselines/terminal/terminus2.py (Terminus-2,
JSON keystroke batches) and terminus_kira.py (native tool calling, completion checklist,
environment snapshot). Every task runs in its own Docker container; scoring runs the task's tests
inside it afterwards. Traces record stop_reason (done / step_limit / overflow / stuck / timeout /
error) and phase labels (understand / write / verify / execute); grep them for "overflow",
"stuck" and "ended_without_writing_code".
Model-dependent axes worth testing rather than assuming: PLANNING (update_plan tool; helps weak
models stay on track, saves cost for strong ones) and BASH_ONLY (bash-capable models often do
better with shell alone on shell-centric tasks); the compaction thresholds B1 / B2 and
CONTEXT_WINDOW. Search = test on this domain: never write task ids, test file names or expected
outputs into a harness; the leakage audit rejects such candidates."""

_scores: dict[str, float] = {}


def adapt(records: list[dict]) -> list[dict]:
    tasks = []
    for record in records:
        instruction = record.get("instruction") or record.get("input")
        if not instruction:
            raise ValueError("terminal records need 'instruction'")
        task = dict(record)
        task["instruction"] = str(instruction)
        task["test_command"] = str(record.get("test_command") or "")
        task["workdir"] = str(record.get("workdir") or "/app")
        task["timeout"] = float(record.get("timeout") or 900.0)
        task["image"] = str(record["image"]) if record.get("image") else None
        tasks.append(task)
    return tasks


def harness_view(task: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in task.items() if k not in HIDDEN_FIELDS}


def evaluate_task(harness: Any, task: Mapping[str, Any], model: Any, trace: Any = None,
                  container_factory: Callable[..., Any] = Container) -> tuple[float, dict]:
    """Run one task end to end in a fresh container and score it there."""
    if not task.get("image"):
        raise ValueError(f"terminal task {task.get('id', '?')!r} has no image; refusing to run on the host")
    workdir = str(task.get("workdir") or "/app")
    container = container_factory(str(task["image"]), workdir=workdir, network=bool(task.get("network")))
    try:
        container_id = container.start()
        live = dict(task, container=container_id, workdir=workdir)
        result = call_run(harness, harness_view(live), model, trace)
        for local, remote in task.get("test_files") or []:
            container.put(Path(local), str(remote))
        score = terminal_metric(live, result)
    finally:
        container.remove()
    if trace is not None and hasattr(trace, "event"):
        trace.event("terminal_score", {"id": task.get("id"), "score": score,
                                       "stop_reason": result.get("stop_reason")})
    return score, result


class InContainer:
    """Wraps a candidate harness so the core evaluator's plain protocol gets the domain flow:
    its `run` does the whole `evaluate_task`, and `METRICS["terminal"]` reads the score back by a
    one-time token (a harness cannot hand itself a score)."""

    def __init__(self, harness: Any, container_factory: Callable[..., Any] = Container):
        self.harness = harness
        self.container_factory = container_factory

    def run(self, task: Mapping[str, Any], model: Any, trace: Any = None) -> dict[str, Any]:
        if "instruction" not in task:  # validation probe
            return call_run(self.harness, task, model, trace)
        score, result = evaluate_task(self.harness, task, model, trace, self.container_factory)
        token = uuid.uuid4().hex
        _scores[token] = score
        return {**result, "_terminal_score_token": token}


HARNESS_WRAPPER = InContainer


def metric(task: Mapping[str, Any], result: Mapping[str, Any]) -> float:
    token = result.get("_terminal_score_token") if isinstance(result, Mapping) else None
    if token is not None and token in _scores:
        return _scores.pop(token)
    return terminal_metric(task, result)


METRICS = {"terminal": metric}
