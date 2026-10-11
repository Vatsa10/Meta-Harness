"""Proposer tools confined to one run directory.

The proposer may list, read and grep files under the run directory and write exactly one
kind of file: a candidate harness plus its rationale. Everything outside the root, and every
forbidden subtree inside it (the test split, proposals, views), is invisible. Paths are
POSIX-style and relative; absolute paths, drive letters, UNC paths and '..' are refused
before resolution, and the resolved target (symlinks and junctions followed) is re-checked.

Long results are capped and the full text is spilled to <root>/_spill/<n>.txt so the
model can page through it with read.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Sequence

from .chat_types import ToolCall, ToolSpec

MAX_RESULT_TOKENS = 6000
MAX_RESULT_CHARS = MAX_RESULT_TOKENS * 4
MAX_RESULT_LINES = 2000
READ_MAX_LINES = 2000
READ_MAX_LINE_CHARS = 2000
READ_MAX_BYTES = 50_000
GREP_MAX_FILE_BYTES = 2 * 1024 * 1024
SPILL_DIR = "_spill"
_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")


class ToolRefused(Exception):
    """A tool call the sandbox will not perform; reported to the model as an error result."""


class ToolSandbox:
    """Read-only tools over one directory, plus the single write. Paths are POSIX-style relative."""

    MAX_READ_CHARS = 50_000
    MAX_GREP_HITS = 200
    MAX_LIST_ENTRIES = 500

    def __init__(self, root: Path, output_dir: Path, forbidden: Sequence[Path] = ()):
        self.root = Path(root).resolve()
        self.output_dir = Path(output_dir)  # where write_candidate puts files; NOT readable via tools
        self.forbidden = [Path(p).resolve() for p in forbidden]
        self.forbidden.append(self.output_dir.resolve())
        self._spill_seq = 0

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

    # -- spill -------------------------------------------------------------------------------

    def _spill(self, full: str) -> str:
        spill_dir = self.root / SPILL_DIR
        spill_dir.mkdir(exist_ok=True)
        while True:
            self._spill_seq += 1
            target = spill_dir / f"{self._spill_seq:04d}.txt"
            if not target.exists():
                break
        target.write_text(full, encoding="utf-8")
        return f"{SPILL_DIR}/{target.name}"

    @staticmethod
    def _pointer(a: int, b: int, n: int, spill: str) -> str:
        return (f"[truncated: showing lines {a}-{b} of {n}. Full output saved to {spill} - "
                f"narrow with read(path, offset, limit) or grep(pattern, path)]")

    def _cap(self, lines: list[str], total: int | None = None, full: str | None = None) -> str:
        """Keep the head of a line list within MAX_RESULT_LINES and MAX_RESULT_CHARS."""
        kept: list[str] = []
        used = 0
        cut = len(lines) > MAX_RESULT_LINES or (total is not None and total > len(lines))
        for line in lines[:MAX_RESULT_LINES]:
            cost = len(line) + 1
            if used + cost > MAX_RESULT_CHARS:
                if not kept:  # one oversized line: cut on a character boundary
                    kept.append(line[: MAX_RESULT_CHARS - 1])
                cut = True
                break
            kept.append(line)
            used += cost
        if not cut:
            return "\n".join(lines)
        spill = self._spill(full if full is not None else "\n".join(lines))
        n = total if total is not None else len(lines)
        return "\n".join(kept + [self._pointer(1, len(kept), n, spill)])

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
                entries.append(f"{child.name}/")
            else:
                try:
                    entries.append(f"{child.name}  {child.stat().st_size}")
                except OSError:
                    continue
        if not entries:
            return "(empty)"
        return self._cap(entries[: self.MAX_LIST_ENTRIES], total=len(entries), full="\n".join(entries))

    def read(self, path: str, offset: int = 0, limit: int = 400) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise ToolRefused(f"not a file: {path}")
        offset = max(0, int(offset))
        limit = max(1, int(limit))
        full = target.read_text(encoding="utf-8", errors="replace")
        all_lines = full.splitlines()
        total = len(all_lines)
        window = all_lines[offset: offset + limit]
        cut = len(window) > READ_MAX_LINES
        window = window[:READ_MAX_LINES]
        out: list[str] = []
        used = 0
        for i, line in enumerate(window, start=offset + 1):
            if len(line) > READ_MAX_LINE_CHARS:
                line = line[:READ_MAX_LINE_CHARS] + "...[line truncated]"
                cut = True
            row = f"{i}\t{line}"
            cost = len(row.encode("utf-8")) + 1
            if used + cost > READ_MAX_BYTES - 600:  # room for the header and the pointer
                cut = True
                break
            out.append(row)
            used += cost
        rel = self._rel(target)
        first = offset + 1 if out else offset
        last = offset + len(out)
        text = "\n".join([f"{rel}: lines {first}-{last} of {total}"] + out)
        if cut:
            text += "\n" + self._pointer(first, last, total, self._spill(full))
        return text

    def grep(self, pattern: str, path: str = ".") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolRefused(f"bad regex: {exc}") from None
        start = self.resolve(path)
        files: list[Path] = []
        if start.is_file():
            files.append(start)
        else:
            for dirpath, dirnames, filenames in os.walk(start, followlinks=False):
                here = Path(dirpath)
                dirnames[:] = sorted(d for d in dirnames
                                     if self._visible(here / d) is not None
                                     and not (here == self.root and d == SPILL_DIR))
                files.extend(here / f for f in sorted(filenames))
        hits: list[str] = []
        for file in files:
            real = self._visible(file)  # re-check: a symlinked file pointing out is skipped
            if real is None or not real.is_file():
                continue
            try:
                if real.stat().st_size > GREP_MAX_FILE_BYTES:
                    continue
                data = real.read_bytes()
            except OSError:
                continue
            if b"\x00" in data[:8192]:
                continue
            rel = self._rel(file)
            for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), start=1):
                if regex.search(line):
                    hits.append(f"{rel}:{n}: {line}")
        if not hits:
            return "(no matches)"
        return self._cap(hits[: self.MAX_GREP_HITS], total=len(hits), full="\n".join(hits))

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
        return f"error: {exc}", None
    except KeyError as exc:
        return f"error: missing argument {exc.args[0]}", None
    except (TypeError, ValueError, OSError) as exc:
        return f"error: {type(exc).__name__}: {exc}", None


__all__ = ["ToolRefused", "ToolSandbox", "TOOL_SPECS", "dispatch", "MAX_RESULT_TOKENS"]
