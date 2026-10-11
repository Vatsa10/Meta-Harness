"""Sandbox protocol (exec/put/get) with temp-dir subprocess and Docker CLI implementations."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Mapping, Protocol

from .timeouts import Aborted, Watchdog

MAX_OUTPUT_BYTES = 1024 * 1024
TRUNCATION_NOTE = "\n[output truncated: showing the last 1 MB]\n"


@dataclass
class ExecResult:
    code: int
    stdout: str
    stderr: str
    timed_out: bool
    timeout_reason: str | None = None  # "idle" | "deadline" | None
    truncated: bool = False


class Sandbox(Protocol):
    def start(self) -> None: ...
    def exec(self, command: str, timeout: float, cwd: str | None = None,
             idle: float | None = None) -> ExecResult: ...
    def put(self, local: Path, remote: str) -> None: ...
    def get(self, remote: str, local: Path) -> None: ...
    def close(self) -> None: ...


class SandboxPathError(ValueError):
    pass


def _minimal_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", "")}
    if sys.platform == "win32":  # Windows processes cannot start without these
        for key in ("SYSTEMROOT", "COMSPEC", "PATHEXT"):
            if key in os.environ:
                env[key] = os.environ[key]
    return env


class _TailBuffer:
    """Keeps only the last `cap` bytes of a stream and remembers whether it dropped any."""

    def __init__(self, cap: int = MAX_OUTPUT_BYTES):
        self.cap = cap
        self.data = bytearray()
        self.truncated = False

    def add(self, chunk: bytes) -> None:
        self.data += chunk
        if len(self.data) > self.cap:
            del self.data[:len(self.data) - self.cap]
            self.truncated = True

    def text(self) -> str:
        out = bytes(self.data).decode("utf-8", errors="replace")
        return TRUNCATION_NOTE + out if self.truncated else out


def _pump(stream, buf: _TailBuffer, watchdog: Watchdog) -> None:
    try:
        while True:
            chunk = stream.read1(65536) if hasattr(stream, "read1") else stream.read(65536)
            if not chunk:
                return
            buf.add(chunk)
            watchdog.kick()
    except (OSError, ValueError):
        return


def _text_of(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value or ""


class TempDirSandbox:
    def __init__(self, env: Mapping[str, str] | None = None):
        self.env = dict(env) if env is not None else _minimal_env()
        self._dir: Path | None = None

    def start(self) -> None:
        if self._dir is None:
            self._dir = Path(tempfile.mkdtemp(prefix="mh-sandbox-")).resolve()

    @property
    def root(self) -> Path:
        self.start()
        assert self._dir is not None
        return self._dir

    def _resolve(self, remote: str) -> Path:
        root = self.root
        win = PureWindowsPath(remote)
        if (Path(remote).is_absolute() or PurePosixPath(remote).is_absolute()
                or win.is_absolute() or win.drive or win.root):
            raise SandboxPathError(f"absolute or drive-relative path not allowed: {remote}")
        target = (root / remote).resolve()
        if target != root and root not in target.parents:
            raise SandboxPathError(f"path escapes sandbox: {remote}")
        return target

    def exec(self, command: str, timeout: float, cwd: str | None = None,
             idle: float | None = None) -> ExecResult:
        workdir = self._resolve(cwd) if cwd else self.root
        kwargs: dict = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(command, shell=True, cwd=str(workdir), env=self.env,
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, **kwargs)
        watchdog = Watchdog(idle if idle is not None else timeout, timeout)
        out, err = _TailBuffer(), _TailBuffer()
        threads = [threading.Thread(target=_pump, args=(proc.stdout, out, watchdog), daemon=True),
                   threading.Thread(target=_pump, args=(proc.stderr, err, watchdog), daemon=True)]
        for t in threads:
            t.start()
        reason: str | None = None
        while proc.poll() is None:
            try:
                watchdog.check()
            except Aborted as aborted:
                reason = aborted.reason
                self._kill_tree(proc)
                break
            time.sleep(0.02)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        for t in threads:
            t.join(timeout=5)
        code = proc.returncode if proc.returncode is not None else -1
        return ExecResult(-1 if reason else code, out.text(), err.text(), reason is not None,
                          reason, out.truncated or err.truncated)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=10)
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.kill()
        except OSError:
            pass

    def _check_links(self, src: Path) -> None:
        """Reject any symlink under `src` that points outside the sandbox root."""
        root = self.root
        for dirpath, dirnames, filenames in os.walk(src):
            for name in dirnames + filenames:
                p = Path(dirpath) / name
                if p.is_symlink():
                    target = p.resolve()
                    if target != root and root not in target.parents:
                        raise SandboxPathError(f"symlink escapes sandbox: {p}")

    def put(self, local: Path, remote: str) -> None:
        dest = self._resolve(remote)
        local = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local.is_dir():
            shutil.copytree(local, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(local, dest)

    def get(self, remote: str, local: Path) -> None:
        src = self._resolve(remote)
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            self._check_links(src)
            shutil.copytree(src, local, dirs_exist_ok=True, symlinks=True)
        else:
            shutil.copy2(src, local)

    def close(self) -> None:
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None

    def __enter__(self) -> "TempDirSandbox":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DockerSandbox:
    def __init__(self, image: str, workdir: str = "/app", network: bool = False,
                 memory_mb: int | None = 2048, cpus: float | None = 2.0, runner=subprocess.run,
                 pids_limit: int | None = 512, shell: str = "sh"):
        self.image = image
        self.workdir = workdir
        self.network = network
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.shell = shell
        self._run = runner
        self.name: str | None = None
        self._started = False

    def _docker(self, args: list[str], timeout: float):
        return self._run(["docker", *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)

    def start(self) -> None:
        if self._started:
            return
        name = f"meta-harness-{uuid.uuid4().hex[:12]}"
        args = ["run", "-d", "--name", name, "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges"]
        if self.pids_limit is not None:
            args += ["--pids-limit", str(self.pids_limit)]
        if not self.network:
            args += ["--network", "none"]
        if self.memory_mb is not None:
            args += ["--memory", f"{self.memory_mb}m"]
        if self.cpus is not None:
            args += ["--cpus", str(self.cpus)]
        args += ["-w", self.workdir, self.image, "sleep", "infinity"]
        self.name = name
        try:
            done = self._docker(args, 300)
            if done.returncode != 0:
                raise RuntimeError(f"docker run failed: {(done.stderr or '').strip()}")
        except BaseException:
            self._remove(name)
            self.name = None
            raise
        self._started = True

    def _remove(self, name: str) -> None:
        try:
            self._docker(["rm", "-f", name], 60)
        except Exception:
            pass

    def _require(self) -> str:
        if not self._started or self.name is None:
            raise RuntimeError("sandbox not started")
        return self.name

    def exec(self, command: str, timeout: float, cwd: str | None = None,
             idle: float | None = None) -> ExecResult:
        """Runs under an in-container `timeout -s KILL`, so the work dies with the deadline,
        not just the local CLI. `idle` is accepted for protocol parity but only the overall
        deadline is enforced (the docker CLI call is not streamed)."""
        name = self._require()
        secs = max(1, int(timeout + 0.999))
        argv = ["exec", "-w", cwd or self.workdir, name, "timeout", "-s", "KILL", str(secs),
                self.shell, "-c", command]
        try:
            done = self._docker(argv, secs + 10)
        except subprocess.TimeoutExpired as exc:
            # The CLI itself hung past the in-container limit: kill the container outright.
            try:
                self._docker(["kill", name], 30)
            except Exception:
                pass
            self._started = False
            return ExecResult(-1, _text_of(exc.stdout), _text_of(exc.stderr), True, "deadline")
        stdout, stderr = done.stdout or "", done.stderr or ""
        truncated = False
        if len(stdout.encode("utf-8", "replace")) > MAX_OUTPUT_BYTES:
            stdout = TRUNCATION_NOTE + stdout[-MAX_OUTPUT_BYTES:]
            truncated = True
        if len(stderr.encode("utf-8", "replace")) > MAX_OUTPUT_BYTES:
            stderr = TRUNCATION_NOTE + stderr[-MAX_OUTPUT_BYTES:]
            truncated = True
        if done.returncode == 137:  # SIGKILL from the in-container timeout
            return ExecResult(-1, stdout, stderr, True, "deadline", truncated)
        return ExecResult(done.returncode, stdout, stderr, False, None, truncated)

    def put(self, local: Path, remote: str) -> None:
        name = self._require()
        done = self._docker(["cp", str(local), f"{name}:{remote}"], 300)
        if done.returncode != 0:
            raise RuntimeError(f"docker cp failed: {(done.stderr or '').strip()}")

    def get(self, remote: str, local: Path) -> None:
        name = self._require()
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        done = self._docker(["cp", f"{name}:{remote}", str(local)], 300)
        if done.returncode != 0:
            raise RuntimeError(f"docker cp failed: {(done.stderr or '').strip()}")

    def close(self) -> None:
        name, self.name, self._started = self.name, None, False
        if name is not None:
            self._remove(name)

    def __enter__(self) -> "DockerSandbox":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class Preflight:
    ok: bool
    problem: str
    hint: str


def preflight(runner=subprocess.run) -> Preflight:
    try:
        done = runner(["docker", "version", "--format", "{{.Server.Version}}"],
                      capture_output=True, text=True, encoding="utf-8", errors="replace",
                      timeout=30)
    except FileNotFoundError:
        hint = ("Install Docker Desktop (Windows/macOS) or the docker.io package (Linux), "
                "then re-run.")
        return Preflight(False, "docker executable not found", hint)
    except subprocess.TimeoutExpired:
        return Preflight(False, "docker did not respond", "Start the Docker daemon and retry.")
    if done.returncode != 0:
        return Preflight(False, (done.stderr or "docker daemon unreachable").strip(),
                         "Start the Docker daemon (open Docker Desktop or run "
                         "`sudo systemctl start docker`) and retry.")
    return Preflight(True, "", "")
