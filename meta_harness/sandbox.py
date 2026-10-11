"""Out-of-process interface validation for LLM-written candidate harnesses.

Validation reuses the evaluation runner (`meta_harness.isolation`): the child has no API keys
in its environment and runs under memory/CPU limits, so an infinite loop, a memory blow-up or
an interpreter-killing import cannot take the outer loop down. In `process` mode it still runs
as the same OS user (see isolation.py for the remaining gap); `docker` mode closes it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

LEGACY_PROBE = {"input": "validation"}
IMPORT_ONLY = "--import-only"


class SandboxError(RuntimeError):
    pass


class _NullTrace:
    def __init__(self) -> None:
        self.count = 0
        self.context_cost = 0.0

    def add_context_cost(self, amount: float) -> None:
        return None

    def event(self, name: str, payload: object = None, **fields: object) -> None:
        self.count += 1


def _null_model(prompt: str, **_: object) -> str:
    return "validation-response"


def _check(path: Path, probe: Mapping[str, Any] | None = None, run: bool = True) -> None:
    from .protocol import call_run


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
    if not run:
        return
    call_run(harness, dict(probe) if probe is not None else dict(LEGACY_PROBE), _null_model, _NullTrace())


def validate_in_subprocess(source: Path | str, timeout: float = 20.0,
                           probe: Mapping[str, Any] | None = None,
                           probe_dir: Path | str | None = None, import_only: bool = False,
                           isolation: str = "process", memory_mb: int = 2048, cpu_seconds: int = 600,
                           image: str | None = None) -> None:
    """Import and smoke-run a candidate in an isolated child. Raises HarnessValidationError.

    Uses the same runner as evaluation (`isolation.IsolatedHarness`): no API keys in the child's
    environment, memory/CPU limits, model calls answered by this process with a fixed
    "validation-response". `probe` is a real task row (targets removed by the caller); without
    it the legacy {"input": "validation"} row is used. `probe_dir` is accepted for compatibility
    (the probe now travels over the protocol, no temp file). `import_only` checks the import and
    the harness interface without running it (domains whose harnesses need a task container).
    `isolation="none"` still uses a child process (validation is never in-process).
    """
    from .core import HarnessValidationError  # local import breaks the import cycle
    from .isolation import DEFAULT_IMAGE, IsolatedHarness, IsolationError, IsolationTimeout

    mode = "docker" if isolation == "docker" else "process"
    started = time.monotonic()
    child = IsolatedHarness(Path(source), mode, memory_mb, cpu_seconds, image or DEFAULT_IMAGE)
    try:
        child.start(timeout=timeout)
        if import_only:
            return
        remaining = max(0.0, timeout - (time.monotonic() - started))
        child.run(dict(probe) if probe is not None else dict(LEGACY_PROBE), None, None, timeout=remaining)
    except IsolationTimeout:
        raise HarnessValidationError(f"validation timed out after {timeout}s") from None
    except IsolationError as exc:
        raise HarnessValidationError(str(exc)) from None
    finally:
        child.close()


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3):
        print("usage: python -m meta_harness.sandbox <candidate.py> [probe.json]", file=sys.stderr)
        return 2
    try:
        if len(argv) == 3 and argv[2] == IMPORT_ONLY:
            _check(Path(argv[1]), run=False)
        else:
            probe = json.loads(Path(argv[2]).read_text(encoding="utf-8")) if len(argv) == 3 else None
            _check(Path(argv[1]), probe)
    except Exception as exc:  # surfaced to the parent through stderr
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
