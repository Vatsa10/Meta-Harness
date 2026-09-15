"""Out-of-process interface validation for LLM-written candidate harnesses.

Candidate code is untrusted in the "may hang or explode" sense, not the security sense: it
runs with the same privileges as the search. The subprocess exists so an infinite loop or an
interpreter-killing import cannot take the outer loop down with it.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


class SandboxError(RuntimeError):
    pass


class _NullTrace:
    def __init__(self) -> None:
        self.count = 0
        self.context_cost = 0.0

    def add_context_cost(self, amount: float) -> None:
        self.context_cost += max(0.0, float(amount))

    def event(self, name: str, payload: object = None, **fields: object) -> None:
        self.count += 1


def _null_model(prompt: str, **_: object) -> str:
    return "validation-response"


def _check(path: Path) -> None:
    spec = importlib.util.spec_from_file_location("meta_harness_sandbox_candidate", path)
    if spec is None or spec.loader is None:
        raise SandboxError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_harness"):
        harness = module.build_harness()
    elif hasattr(module, "Harness"):
        harness = module.Harness()
    else:
        raise SandboxError("candidate must expose build_harness() or Harness")
    if not callable(getattr(harness, "run", None)):
        raise SandboxError("harness must implement run(task, model, trace)")
    harness.run({"input": "validation", "label": "ok"}, _null_model, _NullTrace())


def validate_in_subprocess(source: Path | str, timeout: float = 20.0) -> None:
    """Import and smoke-run a candidate in a child process. Raises HarnessValidationError."""
    from .core import HarnessValidationError  # local import breaks the import cycle

    source = Path(source)
    try:
        completed = subprocess.run([sys.executable, "-m", "meta_harness.sandbox", str(source)],
                                   capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise HarnessValidationError(f"validation timed out after {timeout}s") from None
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise HarnessValidationError(detail[-1] if detail else f"validation exited {completed.returncode}")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m meta_harness.sandbox <candidate.py>", file=sys.stderr)
        return 2
    try:
        _check(Path(argv[1]))
    except Exception as exc:  # surfaced to the parent through stderr
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
