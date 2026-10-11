"""Out-of-process interface validation for LLM-written candidate harnesses.

Candidate code is untrusted in the "may hang or explode" sense, not the security sense: it
runs with the same privileges as the search. The subprocess exists so an infinite loop or an
interpreter-killing import cannot take the outer loop down with it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
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
                           probe_dir: Path | str | None = None, import_only: bool = False) -> None:
    """Import and smoke-run a candidate in a child process. Raises HarnessValidationError.

    `probe` is a real task row (targets removed by the caller); it is written to a temp JSON
    file and handed to the child as argv[2]. Without it the legacy {"input": "validation"} row
    is used. `probe_dir` keeps that temp file inside the run directory. `import_only` checks the
    import and the harness interface without running it (domains whose harnesses need a task
    container, like terminal, cannot smoke-run on the host).
    """
    from .core import HarnessValidationError  # local import breaks the import cycle

    source = Path(source)
    argv = [sys.executable, "-m", "meta_harness.sandbox", str(source)]
    probe_file = None
    if import_only:
        argv.append(IMPORT_ONLY)
    elif probe is not None:
        handle, probe_file = tempfile.mkstemp(suffix=".json", prefix="probe-",
                                             dir=str(probe_dir) if probe_dir else None)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dict(probe), stream, ensure_ascii=False, default=str)
        argv.append(probe_file)
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                   encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        raise HarnessValidationError(f"validation timed out after {timeout}s") from None
    finally:
        if probe_file:
            try:
                os.unlink(probe_file)
            except OSError:
                pass
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        raise HarnessValidationError(detail[-1] if detail else f"validation exited {completed.returncode}")


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
