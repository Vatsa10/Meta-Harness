"""Candidate isolation: run an untrusted harness in a limited child process or a container.

Parent <-> child protocol over the child's stdin/stdout, one JSON object per line:

    parent -> child: {"op":"load","path":...} | {"op":"run","task":{...},"learn":bool}
                     | {"op":"snapshot"} | {"op":"restore"}
                     | {"op":"model_result","text":...,"usage":{...}} | {"op":"exit"}
    child  -> parent: {"op":"model_call","prompt":...,"kwargs":{...}} | {"op":"trace","event":...,"payload":...}
                     | {"op":"learn_error","error":...}
                     | {"op":"result","result":{...}} | {"op":"error","error":"..."}

Model calls are served by the PARENT, so the child needs no network and no keys. The parent
bounds every frame (MAX_FRAME_BYTES; overflow kills the child), rejects non-object frames,
records child trace events only as `harness.<name>` with reserved field names dropped, and
forwards only whitelisted model kwargs (`max_tokens` capped).

Modes
-----
process  The child is `python -I -c <bootstrap> <pkgroot> <candidate>` with an ALLOWLISTED
         environment (PATH, SYSTEMROOT, COMSPEC, PATHEXT, SYSTEMDRIVE, WINDIR, TEMP/TMP/TMPDIR,
         HOME/USERPROFILE, LANG/LC_*); everything else, keys included, is dropped. Its cwd is a
         fresh temp directory, removed on close.
         POSIX: `resource.setrlimit` (RLIMIT_AS, RLIMIT_CPU, RLIMIT_NOFILE) in `preexec_fn`, and
         a new session so a kill (`os.killpg`) takes grandchildren with it.
         Windows: a Job Object through `ctypes` (stdlib) with a per-process memory limit, a
         per-process user-time limit and kill-on-close (grandchildren die with the job). If the
         Job Object cannot be set up the child still runs, timeout-only, and an
         `isolation_warning` event goes into the trace.
         Best-effort guards installed by the bootstrap before the candidate loads: socket
         connects raise, and `builtins.open` / `io.open` refuse paths outside the candidate's
         directory, the temp cwd, the meta_harness package and the Python stdlib/site-packages
         (so the repo-root `.env` is refused). Any `.env*` file and anything under ~/.ssh is
         refused even inside an allowed root. Candidates run from their run-dir copy
         (candidates/<id>/harness.py), so only that directory is allowed, never its parent. These guards are BYPASSABLE (ctypes, os.open,
         _socket, subprocess): the child still runs as the same OS user. `docker` is the real
         security boundary.
docker   `docker run -i --rm --network none --cap-drop ALL --read-only --tmpfs /tmp
         --user 65534:65534 --memory <mb>m --cpus 1` with only the `meta_harness` package and
         the candidate's directory mounted read-only. `preflight()` runs first; a failed
         preflight raises `IsolationError` with the install hint.

Timeouts are preemptive: when a task overruns, the child (or container) is killed, the call
raises `IsolationTimeout`, and the next call starts a fresh child from the candidate source.
`python -I` ignores PYTHONPATH, so the bootstrap inserts the package root into `sys.path`
explicitly instead of relying on `-m` (which would also pick up any installed copy).
Native `model.chat()` is not proxied: inside the child it raises a clear error.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

MODES = ("none", "process", "docker")
DEFAULT_IMAGE = "python:3.12-slim"
ENV_ALLOW = ("PATH", "SYSTEMROOT", "COMSPEC", "PATHEXT", "SYSTEMDRIVE", "WINDIR", "TEMP", "TMP",
             "TMPDIR", "HOME", "USERPROFILE", "LANG")
ENV_ALLOW_PREFIXES = ("LC_",)
STDERR_TAIL = 4000
MAX_FRAME_BYTES = 8 * 1024 * 1024
MODEL_KWARGS = ("temperature", "max_tokens", "stop", "top_p")
DEFAULT_MAX_TOKENS = 32768
RESERVED_FIELDS = ("event", "time", "payload")
CHAT_UNAVAILABLE = "chat() is unavailable under --isolation process; use --isolation none or docker"
BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); "
             "from meta_harness.isolation import child_main; raise SystemExit(child_main(sys.argv[2:]))")
GUARDS_ENV = "META_HARNESS_ISOLATION_GUARDS"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_OVERFLOW = object()


class IsolationError(RuntimeError):
    """The child failed: harness exception, crash (memory limit), or protocol breakage."""


class IsolationTimeout(IsolationError, TimeoutError):
    """The task overran its limit; the child was killed."""


def scrubbed_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Allowlist, not denylist: only what an interpreter needs to start survives."""
    source = os.environ if environ is None else environ
    return {k: v for k, v in source.items()
            if k.upper() in ENV_ALLOW or k.upper().startswith(ENV_ALLOW_PREFIXES)}


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def model_kwargs(kwargs: Mapping[str, Any], max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    clean = {k: v for k, v in kwargs.items() if k in MODEL_KWARGS}
    if "max_tokens" in clean:
        try:
            clean["max_tokens"] = max(1, min(int(clean["max_tokens"]), int(max_tokens)))
        except (TypeError, ValueError):
            del clean["max_tokens"]
    return clean


# -- POSIX limits ---------------------------------------------------------------------------------

def _posix_limits(memory_mb: int, cpu_seconds: int):
    def apply() -> None:  # runs in the child between fork and exec
        import resource
        memory = int(memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds) + 1))
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        cap = 256 if hard == resource.RLIM_INFINITY else min(256, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (cap, cap))
    return apply


# -- Windows Job Object ---------------------------------------------------------------------------

def _windows_job(process: subprocess.Popen, memory_mb: int, cpu_seconds: int) -> Any:
    """Assign `process` to a new Job Object; returns the job handle (closing it kills the child)."""
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    info = EXTENDED()
    # JOB_OBJECT_LIMIT_PROCESS_TIME | JOB_OBJECT_LIMIT_PROCESS_MEMORY | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    info.BasicLimitInformation.LimitFlags = 0x2 | 0x100 | 0x2000
    info.BasicLimitInformation.PerProcessUserTimeLimit = int(cpu_seconds) * 10_000_000  # 100 ns units
    info.ProcessMemoryLimit = int(memory_mb) * 1024 * 1024
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(job, int(process._handle)):  # type: ignore[attr-defined]
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise OSError(error, "AssignProcessToJobObject failed")
    return (kernel32, job)


# -- parent side ----------------------------------------------------------------------------------

class IsolatedHarness:
    """One candidate harness living in a child process (or container) across tasks.

    The harness keeps its state between `run` calls (online learning) until a timeout or crash
    kills the child; the next call then reloads it from source (`restarts` counts these)."""

    def __init__(self, source: Path, mode: str = "process", memory_mb: int = 2048, cpu_seconds: int = 600,
                 image: str = DEFAULT_IMAGE, max_tokens: int = DEFAULT_MAX_TOKENS,
                 inherit_env: bool = False):
        """`inherit_env` passes the parent environment through unfiltered: only for the
        `--isolation none` validation child, which promises no isolation (no guards either)."""
        if mode not in ("process", "docker"):
            raise ValueError(f"isolation mode must be 'process' or 'docker', got {mode!r}")
        self.source = Path(source).resolve()
        self.mode = mode
        self.memory_mb = int(memory_mb)
        self.cpu_seconds = int(cpu_seconds)
        self.image = image
        self.max_tokens = int(max_tokens)
        self.inherit_env = bool(inherit_env)
        self.restarts = 0
        self.warnings: list[str] = []
        self._proc: subprocess.Popen | None = None
        self._job: Any = None
        self._lines: queue.Queue = queue.Queue()
        self._stderr: list[str] = []
        self._container: str | None = None
        self._workdir: str | None = None
        self._started = False

    # lifecycle ---------------------------------------------------------------------------------
    def _argv(self) -> list[str]:
        if self.mode == "process":
            return [sys.executable, "-I", "-c", BOOTSTRAP, str(PACKAGE_ROOT), str(self.source)]
        self._container = f"mh-iso-{uuid.uuid4().hex[:12]}"
        return docker_argv(self._container, self.source, self.memory_mb, self.cpu_seconds, self.image)

    def start(self, timeout: float = 120.0) -> None:
        if self.mode == "docker" and not self._started:
            from .sandboxes import preflight
            check = preflight()
            if not check.ok:
                raise IsolationError(f"docker isolation unavailable: {check.problem}. {check.hint}")
        self._started = True
        kwargs: dict[str, Any] = {}
        env = dict(os.environ) if self.inherit_env else scrubbed_env()
        env[GUARDS_ENV] = "0" if self.inherit_env else "1"
        if self.mode == "process":
            self._workdir = tempfile.mkdtemp(prefix="mh-iso-")
            kwargs["cwd"] = self._workdir
            for key in ("TEMP", "TMP", "TMPDIR"):
                env[key] = self._workdir
            if os.name == "posix":
                kwargs["preexec_fn"] = _posix_limits(self.memory_mb, self.cpu_seconds)
                kwargs["start_new_session"] = True
        env["PYTHONIOENCODING"] = "utf-8"
        self._lines = queue.Queue()
        self._stderr = []
        self._proc = subprocess.Popen(self._argv(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                                      env=env, bufsize=1, **kwargs)
        if self.mode == "process" and os.name == "nt":
            try:
                self._job = _windows_job(self._proc, self.memory_mb, self.cpu_seconds)
            except Exception as exc:  # noqa: BLE001 - degrade to timeout-only, visibly
                self._job = None
                self.warnings.append(f"job object unavailable ({exc}); memory/CPU limits not applied, "
                                     f"timeout-only")
        threading.Thread(target=self._pump_out, args=(self._proc, self._lines), daemon=True).start()
        threading.Thread(target=self._pump_err, args=(self._proc, self._stderr), daemon=True).start()
        self._exchange({"op": "load", "path": str(self.source) if self.mode == "process"
                        else f"/c/{self.source.name}"}, None, None, timeout=timeout)

    @property
    def limits_applied(self) -> bool:
        if self.mode == "docker" or os.name == "posix":
            return True
        return self._job is not None

    @staticmethod
    def _pump_out(proc: subprocess.Popen, lines: queue.Queue) -> None:
        try:
            while True:
                line = proc.stdout.readline(MAX_FRAME_BYTES + 1)  # type: ignore[union-attr]
                if not line:
                    break
                if len(line) > MAX_FRAME_BYTES:
                    lines.put(_OVERFLOW)
                    return
                lines.put(line)
        except (OSError, ValueError):
            pass
        lines.put(None)

    @staticmethod
    def _pump_err(proc: subprocess.Popen, tail: list[str]) -> None:
        try:
            while True:
                line = proc.stderr.readline(STDERR_TAIL)  # type: ignore[union-attr]
                if not line:
                    break
                tail.append(line)
                while sum(len(x) for x in tail) > STDERR_TAIL and len(tail) > 1:
                    tail.pop(0)
        except (OSError, ValueError):
            pass

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if self._container:
            try:
                subprocess.run(["docker", "kill", self._container], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=30)
            except (OSError, subprocess.SubprocessError):
                pass
            self._container = None
        if proc is not None:
            if os.name == "posix" and self.mode == "process":
                try:
                    import signal
                    os.killpg(proc.pid, signal.SIGKILL)  # the whole session: grandchildren too
                except (OSError, ProcessLookupError):
                    pass
            try:
                proc.kill()
            except OSError:
                pass
        if self._job is not None:
            kernel32, handle = self._job
            kernel32.CloseHandle(handle)  # kill-on-close takes any grandchildren with it
            self._job = None
        if proc is not None:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.stdin.write(json.dumps({"op": "exit"}) + "\n")  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
                self._proc.wait(timeout=2)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
        self._kill()
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None

    def __enter__(self) -> "IsolatedHarness":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _ensure(self, trace: Any = None) -> None:
        if self._proc is None or self._proc.poll() is not None:
            if self._started:
                self.restarts += 1
                if self._proc is not None:
                    self._kill()
                if self._workdir:
                    shutil.rmtree(self._workdir, ignore_errors=True)
                    self._workdir = None
            self.start()
            if trace is not None:
                for warning in self.warnings:
                    trace.event("isolation_warning", {"mode": self.mode, "warning": warning})
                self.warnings = []

    # protocol ----------------------------------------------------------------------------------
    def _send(self, message: Mapping[str, Any]) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
        self._proc.stdin.flush()

    def _died(self) -> IsolationError:
        proc = self._proc
        code = proc.poll() if proc is not None else None
        if proc is not None and code is None:
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        time.sleep(0.05)  # let the stderr pump catch the last lines
        detail = "".join(self._stderr).strip().splitlines()
        self._kill()
        tail = detail[-1] if detail else ""
        return IsolationError(f"harness process died (exit {code}; memory or CPU limit, or a crash)"
                              + (f": {tail}" if tail else ""))

    def _broken(self, why: str) -> IsolationError:
        self._kill()
        return IsolationError(f"protocol violation: {why}; harness process killed")

    def _exchange(self, message: Mapping[str, Any], model: Any, trace: Any,
                  timeout: float | None) -> Any:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        try:
            self._send(message)
        except (OSError, ValueError, AssertionError):
            raise self._died() from None
        while True:
            wait = None if deadline is None else deadline - time.monotonic()
            if wait is not None and wait <= 0:
                self._kill()
                raise IsolationTimeout(f"timeout: harness killed after {timeout:.1f}s")
            try:
                line = self._lines.get(timeout=wait)
            except queue.Empty:
                continue
            if line is None:
                raise self._died()
            if line is _OVERFLOW:
                raise self._broken(f"frame larger than {MAX_FRAME_BYTES} bytes")
            try:
                reply = json.loads(line)
            except ValueError:
                raise self._broken("frame is not JSON") from None
            if not isinstance(reply, dict):
                raise self._broken("frame is not a JSON object")
            op = reply.get("op")
            if op == "result":
                return reply.get("result")
            if op == "error":
                raise IsolationError(str(reply.get("error")))
            if op == "trace":
                if trace is not None:
                    fields = reply.get("fields")
                    fields = fields if isinstance(fields, dict) else {}
                    trace.event(f"harness.{reply.get('event')}", reply.get("payload"),
                                **{str(k): v for k, v in fields.items() if str(k) not in RESERVED_FIELDS})
            elif op == "learn_error":
                if trace is not None:
                    trace.event("learn_error", None, error=str(reply.get("error")))
            elif op == "model_call":
                self._serve_model(reply, model, trace)
            else:
                raise self._broken(f"unknown op {op!r}")

    def _serve_model(self, reply: Mapping[str, Any], model: Any, trace: Any) -> None:
        prompt = str(reply.get("prompt", ""))
        raw = reply.get("kwargs")
        kwargs = model_kwargs(raw if isinstance(raw, dict) else {}, self.max_tokens)
        try:
            if model is None:
                text, usage = "validation-response", {}
            elif callable(getattr(model, "call", None)):
                text, usage = model.call(prompt, **kwargs)
            else:
                text, usage = model(prompt, **kwargs), {}
            answer = {"op": "model_result", "text": str(text), "usage": _json_safe(dict(usage or {})),
                      "context_cost": float(getattr(trace, "context_cost", 0.0) or 0.0)}
        except Exception as exc:  # noqa: BLE001 - surfaced to the harness as an exception
            answer = {"op": "model_result", "error": f"{type(exc).__name__}: {exc}"}
        try:
            self._send(answer)
        except (OSError, ValueError, AssertionError):
            pass  # the read loop sees the EOF

    # public API --------------------------------------------------------------------------------
    def run(self, task: Mapping[str, Any], model: Any, trace: Any, learn: bool = False,
            timeout: float | None = None) -> dict:
        """Run one task in the child; with `learn` the child also calls `learn` afterwards.
        `timeout` is preemptive: on expiry the child is killed and IsolationTimeout raised."""
        self._ensure(trace)
        result = self._exchange({"op": "run", "task": _json_safe(dict(task)), "learn": bool(learn)},
                                model, trace, timeout)
        return dict(result) if isinstance(result, Mapping) else {"output": result}

    def snapshot(self, timeout: float | None = 60.0) -> None:
        self._ensure()
        self._exchange({"op": "snapshot"}, None, None, timeout)

    def restore(self, timeout: float | None = 60.0) -> None:
        if self._proc is None:
            return
        self._exchange({"op": "restore"}, None, None, timeout)

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def docker_argv(name: str, source: Path, memory_mb: int, cpu_seconds: int, image: str) -> list[str]:
    source = Path(source)
    return ["docker", "run", "-i", "--rm", "--network", "none", "--cap-drop", "ALL",
            "--read-only", "--tmpfs", "/tmp", "--user", "65534:65534",
            "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m", "--cpus", "1",
            "--pids-limit", "256", "--security-opt", "no-new-privileges", "--name", name,
            "--ulimit", f"cpu={cpu_seconds}", "-w", "/tmp",
            "-v", f"{PACKAGE_ROOT / 'meta_harness'}:/mh/meta_harness:ro",
            "-v", f"{source.parent}:/c:ro", "-e", "PYTHONDONTWRITEBYTECODE=1",
            image, "python", "-I", "-c", BOOTSTRAP, "/mh", f"/c/{source.name}"]


# -- child side -----------------------------------------------------------------------------------

def _always_refused(target: Path, ssh: Path) -> bool:
    """Dotenv files (`.env`, `.env.local`, ...; any case) and anything under ~/.ssh are refused
    whatever the allowed roots are."""
    if target.name.lower().startswith(".env"):
        return True
    return target == ssh or ssh in target.parents


def install_guards(candidate: Path) -> None:
    """Best-effort, BYPASSABLE guards (ctypes, os.open, _socket, subprocess all get around them):
    no socket connects, no open() outside the candidate dir, cwd, package and Python install."""
    import builtins
    import io
    import socket
    import sysconfig

    def refuse(*_: Any, **__: Any) -> Any:
        raise PermissionError("network access is blocked under isolation")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.socket.connect_ex = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]

    # Only the candidate's OWN directory (core evaluates the run-dir copy candidates/<id>/harness.py),
    # never its parent: a seed passed from a repo root must not open the repo's files.
    roots = {Path(candidate).resolve().parent, Path.cwd().resolve(), Path(__file__).resolve().parent,
             Path(tempfile.gettempdir()).resolve()}
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            roots.add(Path(sysconfig.get_paths()[key]).resolve())
        except (KeyError, OSError):
            pass
    for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix}:
        roots.add(Path(prefix).resolve() / ("Lib" if os.name == "nt" else "lib"))
    allowed = tuple(roots)
    ssh = (Path.home() / ".ssh").resolve()
    real_open = builtins.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, bytes, os.PathLike)):
            raw = os.fsdecode(file)
            target = Path(raw).resolve()
            if _always_refused(target, ssh):
                raise PermissionError(f"open() of a secrets file is refused: {raw}")
            if not any(target == root or root in target.parents for root in allowed):
                raise PermissionError(f"open() outside the sandbox is refused: {raw}")
        return real_open(file, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_open  # type: ignore[assignment]


class _Channel:
    def __init__(self, inp, out):
        self.inp, self.out = inp, out
        self.lock = threading.RLock()

    def send(self, message: Mapping[str, Any]) -> None:
        with self.lock:
            self.out.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
            self.out.flush()

    def receive(self) -> dict | None:
        line = self.inp.readline()
        return json.loads(line) if line else None


class _RemoteTrace:
    def __init__(self, channel: _Channel):
        self._channel = channel
        self.context_cost = 0.0
        self.count = 0

    def event(self, name: str, payload: Any = None, **fields: Any) -> None:
        self.count += 1
        self._channel.send({"op": "trace", "event": str(name), "payload": _json_safe(payload),
                            "fields": _json_safe(fields)})

    def add_context_cost(self, amount: float) -> None:
        self.event("harness_reported_cost", {"amount": amount})


class _RemoteModel:
    def __init__(self, channel: _Channel, trace: _RemoteTrace):
        self._channel = channel
        self._trace = trace

    def call(self, prompt: str, **kwargs: Any) -> tuple[str, dict]:
        with self._channel.lock:
            self._channel.send({"op": "model_call", "prompt": str(prompt), "kwargs": _json_safe(kwargs)})
            reply = self._channel.receive()
        if not reply or reply.get("op") != "model_result":
            raise RuntimeError("model channel closed")
        if reply.get("error"):
            raise RuntimeError(str(reply["error"]))
        self._trace.context_cost = float(reply.get("context_cost") or self._trace.context_cost)
        return str(reply.get("text", "")), dict(reply.get("usage") or {})

    def __call__(self, prompt: str, **kwargs: Any) -> str:
        return self.call(prompt, **kwargs)[0]

    def chat(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError(CHAT_UNAVAILABLE)


def _build(path: Path) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location("meta_harness_isolated_candidate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "build_harness"):
        harness = module.build_harness()
    elif hasattr(module, "Harness"):
        harness = module.Harness()
    else:
        raise ImportError("candidate must expose build_harness() or Harness")
    if not callable(getattr(harness, "run", None)):
        raise ImportError("harness must implement run(task, model, trace)")
    return harness


def child_main(argv: list[str]) -> int:
    from .protocol import call_learn, call_run

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    channel = _Channel(sys.stdin, sys.stdout)
    sys.stdout = sys.stderr  # a harness print() must not corrupt the protocol stream
    trace = _RemoteTrace(channel)
    model = _RemoteModel(channel, trace)
    harness: Any = None
    saved: Any = None
    default_path = Path(argv[0]) if argv else None
    if default_path is not None and os.environ.get(GUARDS_ENV, "1") != "0":
        install_guards(default_path)
    while True:
        message = channel.receive()
        if message is None or message.get("op") == "exit":
            return 0
        op = message.get("op")
        try:
            if op == "load":
                harness = _build(Path(message.get("path") or default_path))
                channel.send({"op": "result", "result": {"loaded": True}})
            elif op == "run":
                if harness is None:
                    raise RuntimeError("no harness loaded")
                task = dict(message.get("task") or {})
                result = call_run(harness, task, model, trace)
                if message.get("learn"):
                    try:
                        call_learn(harness, task, result, model)
                    except Exception as exc:  # the score stands; the failure is visible
                        channel.send({"op": "learn_error", "error": f"{type(exc).__name__}: {exc}"})
                channel.send({"op": "result", "result": _json_safe(result)})
            elif op == "snapshot":
                saved = copy.deepcopy(harness)
                channel.send({"op": "result", "result": {"snapshot": True}})
            elif op == "restore":
                if saved is not None:
                    harness, saved = saved, None
                channel.send({"op": "result", "result": {"restored": True}})
            else:
                channel.send({"op": "error", "error": f"unknown op {op!r}"})
        except BaseException as exc:  # noqa: BLE001 - MemoryError included: report, keep serving
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            channel.send({"op": "error", "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    raise SystemExit(child_main(sys.argv[1:]))
