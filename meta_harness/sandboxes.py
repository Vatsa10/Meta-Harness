"""Sandbox protocol (exec/put/get) with temp-dir subprocess and Docker CLI implementations."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Mapping, Protocol


@dataclass
class ExecResult:
    code: int
    stdout: str
    stderr: str
    timed_out: bool


class Sandbox(Protocol):
    def start(self) -> None: ...
    def exec(self, command: str, timeout: float, cwd: str | None = None) -> ExecResult: ...
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
        if Path(remote).is_absolute() or PurePosixPath(remote).is_absolute():
            raise SandboxPathError(f"absolute path not allowed: {remote}")
        target = (root / remote).resolve()
        if target != root and root not in target.parents:
            raise SandboxPathError(f"path escapes sandbox: {remote}")
        return target

    def exec(self, command: str, timeout: float, cwd: str | None = None) -> ExecResult:
        workdir = self._resolve(cwd) if cwd else self.root
        proc = subprocess.Popen(command, shell=True, cwd=str(workdir), env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                encoding="utf-8", errors="replace")
        try:
            out, err = proc.communicate(timeout=timeout)
            return ExecResult(proc.returncode, out or "", err or "", False)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                out, err = "", ""
            return ExecResult(-1, out or "", err or "", True)

    @staticmethod
    def _kill_tree(proc: subprocess.Popen) -> None:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=10)
        proc.kill()

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
            shutil.copytree(src, local, dirs_exist_ok=True)
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
                 memory_mb: int | None = None, cpus: float | None = None, runner=subprocess.run):
        self.image = image
        self.workdir = workdir
        self.network = network
        self.memory_mb = memory_mb
        self.cpus = cpus
        self._run = runner
        self.container_id: str | None = None

    def _docker(self, args: list[str], timeout: float):
        return self._run(["docker", *args], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)

    def start(self) -> None:
        if self.container_id is not None:
            return
        args = ["run", "-d"]
        if not self.network:
            args += ["--network", "none"]
        if self.memory_mb is not None:
            args += ["--memory", f"{self.memory_mb}m"]
        if self.cpus is not None:
            args += ["--cpus", str(self.cpus)]
        args += ["-w", self.workdir, self.image, "sleep", "infinity"]
        done = self._docker(args, 300)
        if done.returncode != 0:
            raise RuntimeError(f"docker run failed: {(done.stderr or '').strip()}")
        self.container_id = (done.stdout or "").strip()

    def _require(self) -> str:
        if self.container_id is None:
            raise RuntimeError("sandbox not started")
        return self.container_id

    def exec(self, command: str, timeout: float, cwd: str | None = None) -> ExecResult:
        cid = self._require()
        try:
            done = self._docker(["exec", "-w", cwd or self.workdir, cid, "bash", "-lc", command],
                                timeout)
        except subprocess.TimeoutExpired as exc:
            def _text(v):
                return v.decode("utf-8", "replace") if isinstance(v, bytes) else (v or "")
            return ExecResult(-1, _text(exc.stdout), _text(exc.stderr), True)
        return ExecResult(done.returncode, done.stdout or "", done.stderr or "", False)

    def put(self, local: Path, remote: str) -> None:
        cid = self._require()
        done = self._docker(["cp", str(local), f"{cid}:{remote}"], 300)
        if done.returncode != 0:
            raise RuntimeError(f"docker cp failed: {(done.stderr or '').strip()}")

    def get(self, remote: str, local: Path) -> None:
        cid = self._require()
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        done = self._docker(["cp", f"{cid}:{remote}", str(local)], 300)
        if done.returncode != 0:
            raise RuntimeError(f"docker cp failed: {(done.stderr or '').strip()}")

    def close(self) -> None:
        cid, self.container_id = self.container_id, None
        if cid is not None:
            self._docker(["rm", "-f", cid], 60)

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
