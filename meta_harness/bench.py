"""Resumable {arm x task x trial} benchmark matrix; each cell runs in a sandboxed subprocess."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .protocol import Metric, hide_target

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Arm:
    name: str
    harness: Path
    provider: str
    model: str


def task_id(task: Mapping, index: int) -> str:
    return str(task["id"]) if task.get("id") is not None else f"t{index}"


def _retryable(status: Any) -> bool:
    return isinstance(status, int) and (status == 429 or status >= 500)


# -- worker side (runs inside the sandbox subprocess) ---------------------------------
class _NullTrace:
    def event(self, *args: Any, **kwargs: Any) -> None:
        pass


class _Counting:
    def __init__(self, model: Any):
        self.model = model
        self.usage = {"calls": 0, "input": 0, "output": 0}

    def call(self, prompt: str, **kwargs: Any):
        upstream = getattr(self.model, "call", None)
        if callable(upstream):
            text, usage = upstream(prompt, **kwargs)
        else:
            text, usage = self.model(prompt, **kwargs), {}
        usage = dict(usage or {})
        self.usage["calls"] += 1
        self.usage["input"] += int(usage.get("prompt_tokens") or usage.get("input")
                                   or usage.get("input_tokens") or 0)
        self.usage["output"] += int(usage.get("completion_tokens") or usage.get("output")
                                    or usage.get("output_tokens") or 0)
        return str(text), usage

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def _worker(cell_file: Path, result_file: Path) -> None:
    from .core import _load_harness
    from .protocol import call_run
    from .providers import model_from_environment

    cell = json.loads(cell_file.read_text(encoding="utf-8"))
    out: dict[str, Any] = {"result": None, "error": None, "status": None, "usage": {}}
    model = None
    try:
        model = _Counting(model_from_environment(cell["provider"], cell["model"]))
        harness = _load_harness(Path(cell["harness"]))
        result = call_run(harness, hide_target(cell["task"]), model, _NullTrace())
        json.dumps(result)
        out["result"] = result
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        status = getattr(exc, "status", None)
        out["status"] = status if isinstance(status, int) else None
    if model is not None:
        out["usage"] = model.usage
    result_file.write_text(json.dumps(out), encoding="utf-8")


def _cell_env() -> dict[str, str]:
    from .sandboxes import _minimal_env
    env = _minimal_env()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"
    for key, value in os.environ.items():
        if "API_KEY" in key or "BASE_URL" in key or key.startswith("META_HARNESS_"):
            env[key] = value
    return env


def subprocess_cell(arm: Arm, task: Mapping, trial: int, timeout: float) -> dict:
    """One attempt of one cell in a TempDirSandbox. Returns result/error/status/usage."""
    from .sandboxes import TempDirSandbox

    cell = {"provider": arm.provider, "model": arm.model, "harness": "harness.py",
            "task": dict(task), "trial": trial}
    with TempDirSandbox(env=_cell_env()) as box:
        (box.root / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
        box.put(Path(arm.harness), "harness.py")
        command = subprocess.list2cmdline([sys.executable, "-m", "meta_harness.bench",
                                           "cell.json", "result.json"])
        done = box.exec(command, timeout)
        if done.timed_out:
            return {"result": None, "error": f"timeout after {timeout}s", "status": None, "usage": {}}
        rfile = box.root / "result.json"
        if not rfile.is_file():
            tail = (done.stderr or done.stdout or "").strip()[-500:]
            return {"result": None, "error": f"worker exited {done.code}: {tail}",
                    "status": None, "usage": {}}
        return json.loads(rfile.read_text(encoding="utf-8"))


# -- parent side ----------------------------------------------------------------------
def _read_done(path: Path) -> dict[tuple, dict]:
    done: dict[tuple, dict] = {}
    if not path.is_file():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
            done[(row["arm"], str(row["task_id"]), int(row["trial"]))] = row
        except (ValueError, KeyError, TypeError):
            continue  # torn last line from a killed run: that cell reruns
    return done


def _append(path: Path, row: Mapping) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def summarize(rows: Sequence[Mapping]) -> dict:
    """Per arm: mean, std over trials (of per-trial means), failures, and per-task mean scores
    (`per_task`) plus raw per-trial score lists (`outcomes`) for a paired test to consume."""
    grouped: dict[str, list[Mapping]] = {}
    for row in rows:
        grouped.setdefault(row["arm"], []).append(row)
    out: dict[str, Any] = {}
    for name, items in grouped.items():
        scores = [float(r["score"]) for r in items]
        by_trial: dict[int, list[float]] = {}
        per_task: dict[str, list[float]] = {}
        for r in sorted(items, key=lambda r: int(r["trial"])):
            by_trial.setdefault(int(r["trial"]), []).append(float(r["score"]))
            per_task.setdefault(str(r["task_id"]), []).append(float(r["score"]))
        out[name] = {
            "mean": sum(scores) / len(scores),
            "std": _std([sum(v) / len(v) for _, v in sorted(by_trial.items())]),
            "cells": len(items),
            "failures": sum(1 for r in items if r.get("error")),
            "per_task": {t: sum(v) / len(v) for t, v in per_task.items()},
            "outcomes": per_task,
        }
    return out


def run_matrix(arms: Sequence[Arm], tasks: Sequence[Mapping], trials: int, metric: Metric, out: Path,
               cell_timeout: float = 600.0, retries: int = 2,
               cell_runner: Callable[[Arm, Mapping, int, float], Mapping] | None = None) -> Path:
    runner = cell_runner or subprocess_cell
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    matrix = out / "matrix.jsonl"
    done = _read_done(matrix)
    for arm in arms:
        for index, task in enumerate(tasks):
            tid = task_id(task, index)
            for trial in range(trials):
                if (arm.name, tid, trial) in done:
                    continue
                started = time.monotonic()
                outcome: Mapping = {}
                for attempt in range(retries + 1):
                    try:
                        outcome = runner(arm, task, trial, cell_timeout)
                    except Exception as exc:  # runner failure: kept, scored 0, not retried
                        outcome = {"result": None, "error": f"{type(exc).__name__}: {exc}",
                                   "status": None, "usage": {}}
                    if outcome.get("error") and _retryable(outcome.get("status")) and attempt < retries:
                        continue
                    break
                error, score = outcome.get("error"), 0.0
                if not error:
                    try:
                        score = float(metric(task, outcome.get("result") or {}))
                    except Exception as exc:
                        error, score = f"metric {type(exc).__name__}: {exc}", 0.0
                row = {"arm": arm.name, "task_id": tid, "trial": trial, "score": score,
                       "error": error, "usage": dict(outcome.get("usage") or {}),
                       "seconds": round(time.monotonic() - started, 3)}
                _append(matrix, row)
                done[(arm.name, tid, trial)] = row
    rows = list(_read_done(matrix).values())
    (out / "summary.json").write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    return matrix


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
