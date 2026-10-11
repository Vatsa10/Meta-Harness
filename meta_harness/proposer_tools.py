"""Proposer tools confined to one run directory.

The proposer may list, read and grep files under the run directory and write exactly one
kind of file: a candidate harness plus its rationale. Everything outside the root, and every
forbidden subtree inside it (the test split, proposals, views), is invisible. Paths are
POSIX-style and relative; absolute paths, drive letters, UNC paths and '..' are refused
before resolution, and the resolved target (symlinks and junctions followed) is re-checked.

Limitation: hard links cannot be detected by path checks. A hard link inside the store that
shares an inode with a file outside it (for example the test split) is indistinguishable from
an ordinary file, so store writers must never create hard links into the run directory.

Every result the model sees, and every spill file written to disk, passes through
meta_harness.redact first. Long results are capped and the full (redacted) text is spilled
once per source file to <root>/_spill/<n>.txt so the model can page through it with read.
grep runs in a child process under a wall-clock budget so a catastrophic regex cannot hang
the engine.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import re
import time
from pathlib import Path
from typing import Any, Sequence

from .chat_types import ToolCall, ToolSpec
from .redact import redact

MAX_RESULT_TOKENS = 6000
MAX_RESULT_CHARS = MAX_RESULT_TOKENS * 4
MAX_RESULT_LINES = 2000
READ_MAX_LINES = 2000
READ_MAX_LINE_CHARS = 2000
READ_MAX_BYTES = 50_000
READ_MAX_FILE_BYTES = 50 * 1024 * 1024
GREP_MAX_FILE_BYTES = 2 * 1024 * 1024
GREP_MAX_PATTERN_CHARS = 500
GREP_MAX_LINE_CHARS = 2000
GREP_BUDGET_SECONDS = 2.0
SPILL_DIR = "_spill"
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class ToolRefused(Exception):
    """A tool call the sandbox will not perform; reported to the model as an error result."""


def _grep_worker(pattern: str, files: list[tuple[str, str]], conn: Any) -> None:
    """Child process: send one (rel, [(n, line)]) message per file, then None."""
    regex = re.compile(pattern)
    for rel, real in files:
        try:
            data = Path(real).read_bytes()
        except OSError:
            continue
        if b"\x00" in data[:8192]:
            continue
        found = []
        for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
            if regex.search(line[:GREP_MAX_LINE_CHARS]):
                found.append((n, line))
        conn.send((rel, found))
    conn.send(None)
    conn.close()


class ToolSandbox:
    """Read-only tools over one directory, plus the single write. Paths are POSIX-style relative."""

    MAX_READ_CHARS = 50_000
    MAX_GREP_HITS = 200
    MAX_LIST_ENTRIES = 500

    def __init__(self, root: Path, output_dir: Path, forbidden: Sequence[Path] = (),
                 grep_budget: float = GREP_BUDGET_SECONDS):
        self.root = Path(root).resolve()
        self.output_dir = Path(output_dir)  # where write_candidate puts files; NOT readable via tools
        self.forbidden = [Path(p).resolve() for p in forbidden]
        self.forbidden.append(self.output_dir.resolve())
        self.grep_budget = grep_budget
        self._spill_seq = 0
        self._spilled: dict[str, str] = {}  # source key -> spill rel path

    # -- confinement -------------------------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        if not isinstance(relative, str):
            raise ToolRefused("path must be a string")
        if "\x00" in relative:
            raise ToolRefused("path contains a NUL byte")
        text = (relative or ".").replace("\\", "/")
        if text.startswith("/") or re.match(r"^[A-Za-z]:", text) or text.startswith("//"):
            raise ToolRefused("absolute paths are not allowed")
        if any(part == ".." for part in text.split("/")):
            raise ToolRefused("'..' is not allowed")
        target = (self.root / text).resolve()  # follows symlinks and junctions
        if target != self.root and self.root not in target.parents:
            raise ToolRefused("path is outside the run directory")
        for blocked in self.forbidden:
            if target == blocked or blocked in target.parents:
                raise ToolRefused("path is not visible to the proposer")
        return target

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def _visible(self, path: Path) -> Path | None:
        """The resolved path if it may be shown, else None (escapes and forbidden subtrees)."""
        try:
            return self.resolve(path.relative_to(self.root).as_posix())
        except (ToolRefused, ValueError, OSError):
            return None

    def _is_spill(self, path: Path) -> bool:
        spill_dir = (self.root / SPILL_DIR).resolve()
        return spill_dir in path.parents

    # -- spill -------------------------------------------------------------------------------

    def _new_spill_path(self) -> Path:
        spill_dir = self.root / SPILL_DIR
        spill_dir.mkdir(exist_ok=True)
        while True:
            self._spill_seq += 1
            target = spill_dir / f"{self._spill_seq:04d}.txt"
            if not target.exists():
                return target

    def _spill_text(self, text: str, key: str | None = None) -> str:
        """Write already-redacted text to a spill file; reuse the spill for the same key."""
        if key is not None and key in self._spilled:
            return self._spilled[key]
        target = self._new_spill_path()
        target.write_text(text, encoding="utf-8")
        rel = f"{SPILL_DIR}/{target.name}"
        if key is not None:
            self._spilled[key] = rel
        return rel

    def _spill_file(self, source: Path) -> str:
        """Spill a redacted copy of a source file, streamed, once per (path, mtime, size)."""
        st = source.stat()
        key = hashlib.sha256(f"{source}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8")).hexdigest()
        if key in self._spilled:
            return self._spilled[key]
        target = self._new_spill_path()
        with source.open("r", encoding="utf-8", errors="replace") as src, \
                target.open("w", encoding="utf-8") as dst:
            for line in src:
                dst.write(redact(line))
        rel = f"{SPILL_DIR}/{target.name}"
        self._spilled[key] = rel
        return rel

    @staticmethod
    def _pointer(a: int, b: int, n: int, spill: str, unit: str = "lines") -> str:
        return (f"[truncated: showing {unit} {a}-{b} of {n}. Full output saved to {spill} - "
                f"narrow with read(path, offset, limit) or grep(pattern, path)]")

    def _cap(self, items: list[str], total: int, full: list[str]) -> str:
        """Keep the head of an item list within MAX_RESULT_LINES and MAX_RESULT_CHARS."""
        kept: list[str] = []
        used = 0
        cut = len(items) > MAX_RESULT_LINES or total > len(items)
        for line in items[:MAX_RESULT_LINES]:
            cost = len(line) + 1
            if used + cost > MAX_RESULT_CHARS:
                if not kept:  # one oversized item: cut on a character boundary
                    kept.append(line[: MAX_RESULT_CHARS - 1])
                cut = True
                break
            kept.append(line)
            used += cost
        if not cut:
            return "\n".join(items)
        spill = self._spill_text("\n".join(full))
        return "\n".join(kept + [self._pointer(1, len(kept), total, spill, unit="items")])

    # -- tools -------------------------------------------------------------------------------

    def list(self, path: str = ".") -> str:
        target = self.resolve(path)
        if not target.is_dir():
            raise ToolRefused(f"not a directory: {path}")
        entries: list[str] = []
        for child in sorted(target.iterdir(), key=lambda p: p.name):
            if self._visible(child) is None:
                continue
            if child.is_dir():
                entries.append(redact(f"{child.name}/"))
            else:
                try:
                    entries.append(redact(f"{child.name}  {child.stat().st_size}"))
                except OSError:
                    continue
        if not entries:
            return "(empty)"
        return self._cap(entries[: self.MAX_LIST_ENTRIES], len(entries), entries)

    def read(self, path: str, offset: int = 0, limit: int = 400) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolRefused(f"not a file: {path}")
        size = target.stat().st_size
        if size > READ_MAX_FILE_BYTES:
            raise ToolRefused(f"file is {size} bytes, over the {READ_MAX_FILE_BYTES} byte read limit; "
                              f"use grep(pattern, path) to locate content")
        offset = max(0, int(offset))
        limit = max(1, int(limit))
        window = min(limit, READ_MAX_LINES)
        out: list[str] = []
        used = 0
        cut = False
        full = False  # byte budget exhausted
        total = 0
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for total, raw in enumerate(handle, start=1):  # streamed: only the window is kept
                if total <= offset:
                    continue
                if total > offset + window:
                    if total <= offset + limit:
                        cut = True  # the caller asked for more than the line cap allows
                    continue
                if full:
                    continue
                line = redact(raw.rstrip("\r\n"))
                if len(line) > READ_MAX_LINE_CHARS:
                    line = line[:READ_MAX_LINE_CHARS] + "...[line truncated]"
                    cut = True
                row = f"{total}\t{line}"
                cost = len(row.encode("utf-8")) + 1
                if used + cost > READ_MAX_BYTES - 600:  # room for the header and the pointer
                    cut = full = True
                    continue
                out.append(row)
                used += cost
        rel = self._rel(target)
        first = offset + 1 if out else offset
        last = offset + len(out)
        text = "\n".join([f"{rel}: lines {first}-{last} of {total}"] + out)
        if cut:
            spill = rel if self._is_spill(target) else self._spill_file(target)
            text += "\n" + self._pointer(first, last, total, spill)
        return text

    def grep(self, pattern: str, path: str = ".") -> str:
        if not isinstance(pattern, str):
            raise ToolRefused("pattern must be a string")
        if len(pattern) > GREP_MAX_PATTERN_CHARS:
            raise ToolRefused(f"pattern is longer than {GREP_MAX_PATTERN_CHARS} characters")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ToolRefused(f"bad regex: {exc}") from None
        start = self.resolve(path)
        candidates: list[Path] = []
        if start.is_file():
            candidates.append(start)
        else:
            for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
                here = Path(dirpath)
                dirnames[:] = sorted(d for d in dirnames
                                     if self._visible(here / d) is not None
                                     and not (here == self.root and d == SPILL_DIR))
                candidates.extend(here / f for f in sorted(filenames))
        files: list[tuple[str, str]] = []
        for file in candidates:
            real = self._visible(file)  # re-check: a symlinked file pointing out is skipped
            if real is None or not real.is_file():
                continue
            try:
                if real.stat().st_size > GREP_MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            files.append((self._rel(file), str(real)))
        hits, stopped = self._run_grep(pattern, files)
        note = (f"[stopped: grep exceeded its {self.grep_budget:g}s time budget; results are partial - "
                f"simplify the pattern or narrow the path]") if stopped else None
        if not hits:
            return note or "(no matches)"
        text = self._cap(hits[: self.MAX_GREP_HITS], len(hits), hits)
        return f"{text}\n{note}" if note else text

    def _run_grep(self, pattern: str, files: list[tuple[str, str]]) -> tuple[list[str], bool]:
        if not files:
            return [], False
        ctx = multiprocessing.get_context("spawn")
        recv, send = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_grep_worker, args=(pattern, files, send), daemon=True)
        proc.start()
        send.close()
        deadline = time.monotonic() + self.grep_budget
        hits: list[str] = []
        stopped = True
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not recv.poll(remaining):
                    break
                try:
                    message = recv.recv()
                except EOFError:
                    stopped = False
                    break
                if message is None:
                    stopped = False
                    break
                rel, found = message
                hits.extend(redact(f"{rel}:{n}: {line}") for n, line in found)
        finally:
            if proc.is_alive():
                proc.terminate()
            proc.join(5)
            recv.close()
        return hits, stopped

    def write_candidate(self, source: str, rationale: str, name: str) -> Path:
        if not isinstance(source, str) or not isinstance(rationale, str):
            raise ToolRefused("source and rationale must be strings")
        if not _NAME_RE.match(name or ""):
            raise ToolRefused(f"bad candidate name: {name!r}")
        try:
            compile(source, name, "exec")
        except SyntaxError as exc:
            raise ToolRefused(f"syntax error: {exc}") from None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        code = self.output_dir / f"{name}.py"
        code.write_text(source, encoding="utf-8")
        (self.output_dir / f"{name}.reasoning.md").write_text(rationale, encoding="utf-8")
        return code


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec("list", "List a directory in the run store: 'name/' for directories, 'name  <bytes>' for files.",
             _schema({"path": {"type": "string", "description": "Relative POSIX path; '.' is the root."}},
                     ["path"])),
    ToolSpec("read", "Read a file as numbered lines. The header states the exact total line count.",
             _schema({"path": {"type": "string"},
                      "offset": {"type": "integer", "minimum": 0, "description": "Lines to skip."},
                      "limit": {"type": "integer", "minimum": 1, "description": "Lines to return (max 2000)."}},
                     ["path"])),
    ToolSpec("grep", "Search files under a path with a Python regex; prints 'path:line: text'.",
             _schema({"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"])),
    ToolSpec("write_candidate", "Submit the new harness source and the reasoning behind it.",
             _schema({"source": {"type": "string", "description": "Complete Python module."},
                      "rationale": {"type": "string"}}, ["source", "rationale"])),
]


def dispatch(sandbox: ToolSandbox, call: ToolCall, name: str) -> tuple[str, Path | None]:
    """Run one tool call. Refusals and bad arguments become 'error: ...' results, never exceptions."""
    args = call.arguments if isinstance(call.arguments, dict) else {}
    if "_raw" in args:
        return "error: tool arguments were not valid JSON", None
    try:
        if call.name == "list":
            return sandbox.list(args.get("path", ".")), None
        if call.name == "read":
            return sandbox.read(args["path"], args.get("offset", 0), args.get("limit", 400)), None
        if call.name == "grep":
            return sandbox.grep(args["pattern"], args.get("path", ".")), None
        if call.name == "write_candidate":
            written = sandbox.write_candidate(args["source"], args["rationale"], name)
            return f"wrote {written.name}", written
        return f"error: unknown tool {call.name}", None
    except ToolRefused as exc:
        return f"error: {redact(str(exc))}", None
    except KeyError as exc:
        return f"error: missing argument {exc.args[0]}", None
    except (TypeError, ValueError, OSError) as exc:
        return f"error: {type(exc).__name__}: {redact(str(exc))}", None


__all__ = ["ToolRefused", "ToolSandbox", "TOOL_SPECS", "dispatch", "MAX_RESULT_TOKENS"]
