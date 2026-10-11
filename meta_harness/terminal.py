"""Building blocks for TerminalBench-2 style agentic coding (paper section 4.3, App. B.3).

TerminalBench-2 task directory format (recorded for the loader, Step 1)
----------------------------------------------------------------------
Two layouts exist and `loaders/terminalbench.py` reads both:

* TB 1.x / early 2.0 (``tasks/<id>/``): ``task.yaml`` (``instruction``, ``difficulty``,
  ``category``, ``tags``, ``max_agent_timeout_sec``, ``max_test_timeout_sec``, ...),
  ``Dockerfile`` (and usually ``docker-compose.yaml``), ``run-tests.sh`` and ``tests/``
  (``test_outputs.py``), ``solution.sh``. Tests are copied to ``/tests`` only at scoring time.
* TB 2.0 harbor layout (``<id>/``): ``task.toml`` (``[agent] timeout_sec``,
  ``[verifier] timeout_sec``, ``[environment] docker_image`` / ``allow_internet``),
  ``instruction.md``, ``environment/Dockerfile``, ``tests/test.sh`` (writes
  ``/logs/verifier/reward.txt``) and ``solution/solve.sh``.

The agent works in the task's container (workdir ``/app``); scoring runs the task's test command
inside the very same container after the agent stops. Nothing here ever runs a model-authored
command on the host: commands go through the `sandboxes.Sandbox` protocol (`DockerSandbox` for
real tasks, `TempDirSandbox` only when a caller explicitly opts in for local tests).

This module holds the pieces both Terminus seeds share and the proposer may tune:

* `Container`              thin wrapper over `DockerSandbox`, used by the terminal domain
* `cap_output`             30KB cap per observation (Terminus-2)
* `apply_edit`/`FileTools` exact -> whitespace-normalised -> difflib (ratio >= 0.9) edit,
                           ambiguous matches refused, read-before-write content hash,
                           `py_compile` diagnostics appended, edits to one file serialised
* `StuckDetector`          reminder at 5 identical calls, stop at 8 identical failures
* `StagedCompactor`        elide bulky middle outputs at B1=0.6, summarise at B2=0.85
* `label_command`          rule-based Understand / Write / Verify / Execute / Other labels
* `Plan`                   the `planning` toggle's update_plan state
"""

from __future__ import annotations

import difflib
import hashlib
import json
import posixpath
import py_compile
import re
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .metrics import estimate_tokens
from .sandboxes import DockerSandbox, ExecResult, preflight

OBSERVATION_CAP_CHARS = 24_000  # tool-output cap from the coding-agent harness paper (refs section 2)
OBSERVATION_CAP_BYTES = OBSERVATION_CAP_CHARS  # legacy name; the cap counts characters
STOP_REASONS = ("done", "step_limit", "overflow", "stuck", "timeout", "error")


# --------------------------------------------------------------------------- policy / result
@dataclass
class ShellPolicy:
    timeout_seconds: float = 120.0
    max_output_chars: int = OBSERVATION_CAP_BYTES
    max_steps: int = 60
    cwd: str | None = None
    allowed_commands: tuple[str, ...] = ()

    def check(self, command: str) -> None:
        if len(command) > 20000:
            raise PermissionError("command exceeds maximum length")
        if self.allowed_commands:
            words = command.strip().split(maxsplit=1)
            program = words[0] if words else ""
            if program not in self.allowed_commands:
                raise PermissionError(f"command is not allowlisted: {program}")


@dataclass
class DockerPolicy(ShellPolicy):
    container: str = ""
    workdir: str = "/app"

    def wrap(self, command: str, shell: str = "bash") -> list[str]:
        flag = "-lc" if shell == "bash" else "-c"
        return ["docker", "exec", "-w", self.workdir, self.container, shell, flag, command]


@dataclass
class TerminalResult:
    completed: bool
    answer: str
    steps: int
    history: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "done"


# --------------------------------------------------------------------------- container
class TaskSandbox(DockerSandbox):
    """DockerSandbox for TerminalBench task containers.

    Deliberate difference from the default sandbox: no `--cap-drop ALL`, because task images and
    agents run apt / pip, which need Docker's default capabilities. `--security-opt
    no-new-privileges` and the pids / memory / cpu limits stay. Network follows the task.
    """

    def run_args(self, name: str) -> list[str]:
        args = ["run", "-d", "--name", name, "--security-opt", "no-new-privileges"]
        if self.pids_limit is not None:
            args += ["--pids-limit", str(self.pids_limit)]
        if not self.network:
            args += ["--network", "none"]
        if self.memory_mb is not None:
            args += ["--memory", f"{self.memory_mb}m"]
        if self.cpus is not None:
            args += ["--cpus", str(self.cpus)]
        # --entrypoint: `sleep infinity` runs directly whatever ENTRYPOINT the image sets, so
        # the container survives `docker restart` before scoring.
        return args + ["-w", self.workdir, "--entrypoint", "sleep", self.image, "infinity"]

    def start(self) -> None:
        if self._started:
            return
        name = f"meta-harness-{uuid.uuid4().hex[:12]}"
        self.name = name
        try:
            done = self._docker(self.run_args(name), 300)
            if done.returncode != 0:
                raise RuntimeError(f"docker run failed: {(done.stderr or '').strip()}")
        except BaseException:
            self._remove(name)
            self.name = None
            raise
        self._started = True


class Container:
    """A task container: `docker run -d` (network only when the task asks), exec, copy, rm.

    A thin wrapper over `TaskSandbox` (a `DockerSandbox`): every docker CLI call passes
    encoding="utf-8", errors="replace" and a timeout, and every exec (including scoring) runs
    under the sandbox's in-container `timeout -s KILL`. The shell is decided once by probing
    `command -v bash`: bash when present, else sh.
    """

    def __init__(self, image: str, workdir: str = "/app", network: bool = False,
                 runner: Callable[..., Any] = subprocess.run, sandbox: DockerSandbox | None = None):
        self.image = image
        self.workdir = workdir
        self.network = network
        self._run = runner
        self.sandbox = sandbox or TaskSandbox(image, workdir=workdir, network=network, runner=runner,
                                              shell="sh")
        self._shell_checked = False

    @staticmethod
    def available(runner: Callable[..., Any] = subprocess.run) -> bool:
        """True when `docker version` reaches a running daemon."""
        try:
            return preflight(runner).ok
        except Exception:
            return False

    @classmethod
    def attach(cls, container_id: str, workdir: str = "/app",
               runner: Callable[..., Any] = subprocess.run) -> "Container":
        """Wrap an already running container (created by the terminal domain)."""
        sandbox = TaskSandbox("", workdir=workdir, runner=runner, shell="sh")
        sandbox.name = str(container_id)
        sandbox._started = True
        return cls("", workdir=workdir, runner=runner, sandbox=sandbox)

    @property
    def id(self) -> str | None:
        return self.sandbox.name

    def start(self) -> str:
        self.sandbox.start()
        assert self.sandbox.name is not None
        return self.sandbox.name

    def _pick_shell(self) -> None:
        if self._shell_checked:
            return
        self._shell_checked = True
        self.sandbox.shell = "sh"
        probe = self.sandbox.exec("command -v bash", 20)
        if probe.code == 0 and probe.stdout.strip():
            self.sandbox.shell = "bash"
        self.shell = self.sandbox.shell

    def exec(self, command: str, timeout: float) -> tuple[int, str]:
        self._pick_shell()
        result = self.sandbox.exec(command, timeout)
        return _code_and_text(result)

    def put(self, local: Path, remote: str) -> None:
        self.sandbox.put(Path(local), remote)

    def exec_test(self, command: str, timeout: float) -> tuple[int, str]:
        """Run the task's test command in this container through `DockerSandbox.exec`
        (`docker exec -w <workdir> <id> timeout -s KILL <n> /bin/<bash|sh> -c <command>`; the
        container is killed if the CLI itself hangs). The shell is invoked by absolute path, so a
        binary the agent planted earlier on PATH is never the one that runs the tests. Never runs
        anything on the host."""
        self._pick_shell()
        shell = self.sandbox.shell
        self.sandbox.shell = "/bin/bash" if shell == "bash" else "/bin/sh"
        try:
            return _code_and_text(self.sandbox.exec(command, timeout))
        finally:
            self.sandbox.shell = shell

    def restart(self, timeout: float = 120.0) -> None:
        """`docker restart -t 0`: kills every process the agent left behind (a background loop
        rewriting the reward file or the tests) before scoring."""
        done = self._run(["docker", "restart", "-t", "0", str(self.id)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)
        if done.returncode != 0:
            raise RuntimeError(f"docker restart failed: {(done.stderr or '').strip()[-500:]}")
        alive = self._run(["docker", "exec", str(self.id), "true"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
        if alive.returncode != 0:
            raise RuntimeError(f"container {self.id} is not running after docker restart; refusing to "
                               f"score: {(alive.stderr or '').strip()[-500:]}")

    def diff(self, timeout: float = 120.0) -> list[tuple[str, str]]:
        """`docker diff`: (kind, path) for every change against the image; kind is A, C or D."""
        done = self._run(["docker", "diff", str(self.id)], capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=timeout)
        if done.returncode != 0:
            raise RuntimeError(f"docker diff failed: {(done.stderr or '').strip()[-500:]}")
        changes = []
        for line in (done.stdout or "").splitlines():
            kind, _, path = line.strip().partition(" ")
            if kind and path:
                changes.append((kind, path.strip()))
        return changes

    def remove(self) -> None:
        self.sandbox.close()


def _code_and_text(result: ExecResult) -> tuple[int, str]:
    text = (result.stdout or "") + (result.stderr or "")
    if result.timed_out:
        return 124, text + f"\n[command timed out ({result.timeout_reason})]"
    return result.code, text


def run_in_sandbox(sandbox: Any, command: str, timeout: float, cwd: str | None = None) -> tuple[int, str]:
    if isinstance(sandbox, Container):
        return sandbox.exec(command, timeout)
    return _code_and_text(sandbox.exec(command, timeout, cwd=cwd))


# --------------------------------------------------------------------------- output cap
def cap_output(text: str, limit: int = OBSERVATION_CAP_CHARS) -> str:
    """Keep the head and tail of an observation so it is at most `limit` characters."""
    if len(text) <= limit:
        return text
    note = "\n[... output truncated: {} chars omitted ...]\n"
    room = limit - len(note.format(len(text))) - 2
    head, tail = room // 2, room - room // 2
    return text[:head] + note.format(len(text) - head - tail) + text[len(text) - tail:]


# --------------------------------------------------------------------------- fuzzy edit
class EditError(ValueError):
    pass


FUZZY_RATIO = 0.9


def _norm(line: str) -> str:
    return " ".join(line.split())


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip(" \t"))]


def _splice(lines: list[str], start: int, size: int, old_lines: list[str], new: str) -> str:
    block = lines[start:start + size]
    newline = "\r\n" if block and block[0].endswith("\r\n") else "\n"
    trailing = bool(block) and block[-1].endswith(("\n", "\r"))
    old_first = next((l for l in old_lines if l.strip()), "")
    got_first = next((l for l in block if l.strip()), "")
    old_ind, got_ind = _indent(old_first), _indent(got_first.rstrip("\r\n"))
    out = []
    for line in new.splitlines():
        if line.strip() and line.startswith(old_ind):
            line = got_ind + line[len(old_ind):]
        out.append(line)
    replacement = newline.join(out) + (newline if trailing and out else "")
    return "".join(lines[:start]) + replacement + "".join(lines[start + size:])


def apply_edit(text: str, old: str, new: str) -> tuple[str, str]:
    """Replace `old` with `new` in `text`. Returns (new_text, mode).

    1. exact substring, must be unique;
    2. whitespace-normalised line window, must be unique (re-indents `new` to the match);
    3. best `difflib.SequenceMatcher` line window with ratio >= 0.9; refused when another
       non-overlapping window also reaches 0.9 (ambiguous).
    """
    if not old:
        raise EditError("old text is empty")
    count = text.count(old)
    if count == 1:
        return text.replace(old, new, 1), "exact"
    if count > 1:
        raise EditError(f"old text matches {count} places; include more surrounding lines")
    lines = text.splitlines(keepends=True)
    old_lines = old.splitlines()
    while old_lines and not old_lines[0].strip():
        old_lines.pop(0)
    while old_lines and not old_lines[-1].strip():
        old_lines.pop()
    if not old_lines:
        raise EditError("old text is blank")
    k = len(old_lines)
    normalised = [_norm(l) for l in lines]
    target = [_norm(l) for l in old_lines]
    hits = [i for i in range(len(lines) - k + 1) if normalised[i:i + k] == target]
    if len(hits) > 1:
        raise EditError(f"old text matches {len(hits)} places after whitespace normalisation; "
                        "include more surrounding lines")
    if len(hits) == 1:
        return _splice(lines, hits[0], k, old_lines, new), "whitespace"
    wanted = "\n".join(target)
    scored: list[tuple[float, int, int]] = []
    best_seen = 0.0
    for size in sorted({max(1, k - 1), k, k + 1}):
        for start in range(len(lines) - size + 1):
            candidate = "\n".join(normalised[start:start + size])
            matcher = difflib.SequenceMatcher(None, wanted, candidate, autojunk=False)
            if matcher.real_quick_ratio() < FUZZY_RATIO or matcher.quick_ratio() < FUZZY_RATIO:
                continue
            ratio = matcher.ratio()
            best_seen = max(best_seen, ratio)
            if ratio >= FUZZY_RATIO:
                scored.append((ratio, start, size))
    if not scored:
        raise EditError(f"old text not found (best fuzzy ratio {best_seen:.2f} < {FUZZY_RATIO})")
    best = max(scored, key=lambda s: (s[0], -s[2]))
    rivals = [s for s in scored if s[1] >= best[1] + best[2] or s[1] + s[2] <= best[1]]
    if rivals:
        raise EditError("old text is ambiguous: several places match it fuzzily; "
                        "include more surrounding lines")
    return _splice(lines, best[1], best[2], old_lines, new), f"fuzzy {best[0]:.2f}"


class FileTools:
    """read / edit / write through a sandbox's get/put, with read-before-write hashing.

    `edit`/`write` on an existing file are refused unless the file was fully read in this session
    and its content hash is unchanged. After a `.py` file changes, `py_compile` diagnostics are
    appended to the result. Edits to one path are serialised by a per-path lock.
    """

    def __init__(self, sandbox: Any, workdir: str | None = None):
        self.sandbox = sandbox.sandbox if isinstance(sandbox, Container) else sandbox
        self.workdir = workdir
        self._hashes: dict[str, str] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _remote(self, path: str) -> str:
        path = str(path)
        if self.workdir and not path.startswith("/"):
            return posixpath.join(self.workdir, path)
        return path

    def _lock(self, path: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(path, threading.Lock())

    def _fetch(self, remote: str) -> str | None:
        with tempfile.TemporaryDirectory(prefix="mh-file-") as td:
            local = Path(td) / "file"
            try:
                self.sandbox.get(remote, local)
            except FileNotFoundError:
                return None
            except RuntimeError as exc:
                if re.search(r"no such|could not find|not found", str(exc), re.IGNORECASE):
                    return None
                raise
            if local.is_dir():
                raise EditError(f"{remote} is a directory")
            return local.read_bytes().decode("utf-8", "replace")

    def _store(self, remote: str, text: str) -> str:
        """Write `text` and return py_compile diagnostics for .py files."""
        with tempfile.TemporaryDirectory(prefix="mh-file-") as td:
            local = Path(td) / (posixpath.basename(remote) or "file")
            local.write_text(text, encoding="utf-8", newline="")
            self.sandbox.put(local, remote)
            if not remote.endswith(".py"):
                return ""
            try:
                py_compile.compile(str(local), cfile=str(Path(td) / "check.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                return "\n[py_compile] " + str(exc.msg).strip()
            return "\n[py_compile] OK"

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()

    def read(self, path: str, cap: int = OBSERVATION_CAP_BYTES) -> str:
        remote = self._remote(path)
        text = self._fetch(remote)
        if text is None:
            raise FileNotFoundError(f"{path}: no such file")
        shown = cap_output(text, cap)
        if shown == text:  # only a full read licenses a later write
            self._hashes[remote] = self._digest(text)
        return shown

    def _check_fresh(self, remote: str, current: str, path: str) -> None:
        known = self._hashes.get(remote)
        if known is None:
            raise EditError(f"{path} exists but was not read in full this session; read it first")
        if known != self._digest(current):
            raise EditError(f"{path} changed since you read it; read it again before editing")

    def edit(self, path: str, old: str, new: str) -> str:
        remote = self._remote(path)
        with self._lock(remote):
            current = self._fetch(remote)
            if current is None:
                raise FileNotFoundError(f"{path}: no such file")
            self._check_fresh(remote, current, path)
            updated, mode = apply_edit(current, old, new)
            diagnostics = self._store(remote, updated)
            self._hashes[remote] = self._digest(updated)
            return f"edited {path} ({mode} match){diagnostics}"

    def write(self, path: str, content: str) -> str:
        remote = self._remote(path)
        with self._lock(remote):
            current = self._fetch(remote)
            if current is not None:
                self._check_fresh(remote, current, path)
            diagnostics = self._store(remote, content)
            self._hashes[remote] = self._digest(content)
            return f"wrote {path} ({len(content)} chars){diagnostics}"


# --------------------------------------------------------------------------- stuck detection
REMIND_AT = 5
STUCK_AT = 8
REPEAT_REMINDER = ("[reminder] You have issued the same call {n} times in a row without progress. "
                   "Change approach instead of repeating it.")
FAIL_REMINDER = ("[reminder] The same call keeps failing ({n} times in a row). Read the error, "
                 "then change the command or the approach.")


class StuckDetector:
    """Tracks streaks of byte-identical (tool, args).

    One reminder per streak at `remind_at` identical calls ("keeps failing" when all of them
    failed, else "repeating"); `stuck` becomes True at `stop_at` identical failing calls.
    Permission denials neither count nor break a streak; any different argument breaks it.
    """

    def __init__(self, remind_at: int = REMIND_AT, stop_at: int = STUCK_AT):
        self.remind_at, self.stop_at = remind_at, stop_at
        self.key: str | None = None
        self.count = 0
        self.failures = 0
        self.reminded = False
        self.stuck = False

    def observe(self, tool: str, args: Any, failed: bool, denied: bool = False) -> str | None:
        if denied:
            return None
        key = json.dumps([tool, args], sort_keys=True, default=str)
        if key != self.key:
            self.key, self.count, self.failures, self.reminded = key, 0, 0, False
        self.count += 1
        self.failures = self.failures + 1 if failed else 0
        if self.failures >= self.stop_at:
            self.stuck = True
            return None
        if self.count >= self.remind_at and not self.reminded:
            self.reminded = True
            template = FAIL_REMINDER if self.failures >= self.count else REPEAT_REMINDER
            return template.format(n=self.count)
        return None


# --------------------------------------------------------------------------- staged compaction
B1 = 0.6
B2 = 0.85
RECENT_FRACTION = 0.3
MIN_RECENT_TURNS = 2
ELIDE_MIN_CHARS = 400
SUMMARY_MARKER = "[Summary of earlier work]"
CONTINUE_LINE = "Continue from the summary; do not redo work already done."
ELIDED_STUB = "[tool output elided: {lines} lines / {chars} chars. Re-read or re-run to get it again.]"
SUMMARY_PROMPT = """You are compacting the history of a terminal agent's session so it can keep \
working without it. Write a summary using exactly these headings:

## Goal
## Files touched
## Done
## Pending
## Errors & fixes
## Current state
## Next step

Update the previous summary with the new events rather than discarding anything still true.

Previous summary:
{previous}

Events to fold in:
{events}
"""


def message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        content = message.get("text", "")
    calls = message.get("tool_calls") or []
    rendered = [f"{getattr(c, 'name', '')}({json.dumps(getattr(c, 'arguments', {}), default=str)})"
                for c in calls]
    return str(content) + ("\n" + "\n".join(rendered) if rendered else "")


def message_tokens(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(estimate_tokens(message_text(m)) + 4 for m in messages)


@dataclass
class CompactionInfo:
    elided: int = 0
    summarised: int = 0
    summary_calls: int = 0
    tokens_before: int = 0
    tokens_after: int = 0


class StagedCompactor:
    """Elide-then-summarise context management (App. Alg. 1 of the coding-agent harness paper).

    The preamble (first `preamble` messages: the task) and a recent window (`recent_fraction`
    of the window, at least `min_recent_turns` whole turns, cut at an assistant-turn boundary)
    stay byte-identical. At `b1` of the window, bulky middle tool outputs become stubs; only if
    still at or above `b2` are the middle events folded into a running summary (fixed headings,
    previous summary included) inserted right after the preamble with the continuation line.
    """

    def __init__(self, context_window: int, b1: float = B1, b2: float = B2,
                 recent_fraction: float = RECENT_FRACTION, min_recent_turns: int = MIN_RECENT_TURNS,
                 preamble: int = 1, elide_min_chars: int = ELIDE_MIN_CHARS):
        self.window = int(context_window)
        self.b1, self.b2 = b1, b2
        self.recent_fraction = recent_fraction
        self.min_recent_turns = min_recent_turns
        self.preamble = preamble
        self.elide_min_chars = elide_min_chars

    def _recent_start(self, messages: list[dict]) -> int:
        starts = [i for i in range(self.preamble, len(messages)) if messages[i].get("role") == "assistant"]
        if not starts:
            return len(messages)
        budget = self.recent_fraction * self.window
        chosen = len(messages)
        turns = 0
        for start in reversed(starts):
            size = message_tokens(messages[start:])
            if turns >= self.min_recent_turns and size > budget:
                break
            chosen, turns = start, turns + 1
        return chosen

    def _is_bulky_output(self, message: Mapping[str, Any]) -> bool:
        if message.get("role") != "tool" and message.get("kind") != "observation":
            return False
        content = str(message.get("content", ""))
        return len(content) >= self.elide_min_chars and not content.startswith("[tool output elided")

    def compact(self, messages: Sequence[Mapping[str, Any]], summarize: Callable[[str], str],
                system: str = "") -> tuple[list[dict], CompactionInfo]:
        messages = [dict(m) for m in messages]
        extra = estimate_tokens(system)
        info = CompactionInfo(tokens_before=message_tokens(messages) + extra)
        if info.tokens_before < self.b1 * self.window:
            info.tokens_after = info.tokens_before
            return messages, info
        head = messages[:self.preamble]
        rest = messages[self.preamble:]
        previous = ""
        if rest and rest[0].get("kind") == "summary":
            previous = str(rest[0]["content"]).split("\n", 1)[-1].replace(CONTINUE_LINE, "").strip()
            rest = rest[1:]
        body = head + rest
        cut = self._recent_start(body)
        middle, recent = body[self.preamble:cut], body[cut:]
        for message in middle:
            if self._is_bulky_output(message):
                content = str(message["content"])
                message["content"] = ELIDED_STUB.format(lines=content.count("\n") + 1, chars=len(content))
                info.elided += 1
        summary_msg = ([{"role": "user", "kind": "summary",
                         "content": f"{SUMMARY_MARKER}\n{previous}\n\n{CONTINUE_LINE}"}] if previous else [])
        out = head + summary_msg + middle + recent
        total = message_tokens(out) + extra
        if total >= self.b2 * self.window and (middle or previous):
            events = "\n\n".join(f"[{m.get('role')}] {message_text(m)}" for m in middle)
            prompt = SUMMARY_PROMPT.format(previous=previous or "(none)", events=events or "(none)")
            text = str(summarize(prompt)).strip()
            info.summary_calls += 1
            info.summarised = len(middle)
            out = head + [{"role": "user", "kind": "summary",
                           "content": f"{SUMMARY_MARKER}\n{text}\n\n{CONTINUE_LINE}"}] + recent
            total = message_tokens(out) + extra
        info.tokens_after = total
        return out, info


# --------------------------------------------------------------------------- phase labels
UNDERSTAND, WRITE, VERIFY, EXECUTE, OTHER = "understand", "write", "verify", "execute", "other"
_PRIORITY = {WRITE: 4, VERIFY: 3, EXECUTE: 2, UNDERSTAND: 1, OTHER: 0}
_READERS = {"cat", "ls", "grep", "egrep", "rg", "find", "head", "tail", "less", "more", "wc", "pwd",
            "tree", "file", "stat", "which", "whoami", "env", "ps", "du", "df", "id", "uname", "type"}
_VERIFIERS = {"pytest", "diff", "cmp", "test", "[", "shellcheck", "mypy", "ruff", "flake8"}
_RUNNERS = {"python", "python3", "node", "bash", "sh", "make", "gcc", "g++", "cc", "clang",
            "cargo", "go", "java", "javac", "rustc", "pip", "pip3", "apt-get", "apt", "npm", "uv",
            "curl", "wget", "ruby", "perl", "cmake", "git", "tar", "unzip", "chmod"}


def _segment_label(segment: str) -> str:
    words = segment.strip().split()
    while words and (re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", words[0]) or words[0] in ("sudo", "time", "env")):
        words = words[1:]
    if not words:
        return OTHER
    prog = words[0].rsplit("/", 1)[-1]
    text = " ".join(words)
    if (re.search(r"<<-?\s*['\"]?\w+", text) and re.search(r">\s*[\w./~-]", text)) \
            or re.search(r"\bsed\s+(-\w*\s+)*-i", text) or prog in ("tee", "patch") \
            or (prog in ("echo", "printf", "cat") and re.search(r"(?<![0-9&])>>?\s*(?!/dev/null)[\w./~-]", text)):
        return WRITE
    if prog in _VERIFIERS or re.search(r"(?:^|\s)-m\s+(pytest|unittest|py_compile)\b", text) \
            or re.search(r"\b(make|npm|cargo|go|yarn|pnpm)\s+test\b", text):
        return VERIFY
    if prog in _READERS or (prog == "sed" and "-n" in words) or prog == "cd":
        return UNDERSTAND
    if prog in _RUNNERS or words[0].startswith("./"):
        return EXECUTE
    return OTHER


def label_command(command: str) -> str:
    """Rule-based phase label for one shell command. `cd` only navigates, so it never decides the
    label when anything else is chained (`cd /app && python solve.py` is execute)."""
    segments = [s for s in re.split(r"&&|\|\||;|\n|\|", command) if s.strip()]
    work = [s for s in segments if s.strip().split()[0] != "cd"]
    if not work:
        return UNDERSTAND if segments else OTHER
    return max((_segment_label(s) for s in work), key=lambda label: _PRIORITY[label])


def label_action(tool: str, args: Mapping[str, Any]) -> str:
    if tool in ("bash", "keystrokes"):
        return label_command(str(args.get("command", args.get("keystrokes", ""))))
    if tool in ("read", "read_file"):
        return UNDERSTAND
    if tool in ("edit", "write", "write_file"):
        return WRITE
    return OTHER


def phase_summary(labels: Sequence[str]) -> dict[str, Any]:
    work = [l for l in labels if l != OTHER]
    return {"phases": list(labels), "ended_without_writing_code": WRITE not in labels,
            "final_phase": work[-1] if work else OTHER,
            "ended_in_understand": bool(work) and work[-1] == UNDERSTAND}


# --------------------------------------------------------------------------- planning toggle
PLAN_REMINDER = ("[plan] You have no plan yet. Call update_plan with the full list of steps, "
                 "exactly one of them in_progress, before doing anything else.")
PLAN_STATUSES = ("pending", "in_progress", "completed")


class Plan:
    """State of the `update_plan` tool: each call replaces the whole list."""

    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def update(self, items: Any) -> str:
        if not isinstance(items, list) or not items:
            raise ValueError("update_plan needs a non-empty list of {step, status}")
        cleaned = []
        for item in items:
            if not isinstance(item, Mapping) or not str(item.get("step", "")).strip():
                raise ValueError("each plan item needs a 'step'")
            status = str(item.get("status", "pending"))
            if status not in PLAN_STATUSES:
                raise ValueError(f"status must be one of {PLAN_STATUSES}")
            cleaned.append({"step": str(item["step"]).strip(), "status": status})
        active = sum(1 for i in cleaned if i["status"] == "in_progress")
        unfinished = any(i["status"] != "completed" for i in cleaned)
        if unfinished and active != 1:
            raise ValueError(f"exactly one step must be in_progress (got {active})")
        self.items = cleaned
        return f"plan updated ({len(cleaned)} steps)"

    def block(self) -> str:
        """The per-turn injection; never stored in the history."""
        if not self.items:
            return PLAN_REMINDER
        marks = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        return "[plan]\n" + "\n".join(f"{marks[i['status']]} {i['step']}" for i in self.items)


def __getattr__(name: str) -> Any:
    if name == "TerminalAgent":  # the agent loop lives in terminal_harness
        from .terminal_harness import TerminalAgent
        return TerminalAgent
    raise AttributeError(name)


__all__ = ["B1", "B2", "CONTINUE_LINE", "Container", "DockerPolicy", "EditError", "FileTools",
           "OBSERVATION_CAP_BYTES", "OBSERVATION_CAP_CHARS", "Plan", "TaskSandbox", "ShellPolicy", "StagedCompactor", "StuckDetector",
           "TerminalAgent", "TerminalResult", "apply_edit", "cap_output", "label_action",
           "label_command", "phase_summary", "run_in_sandbox"]
