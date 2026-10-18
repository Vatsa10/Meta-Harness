"""Find the tunable knobs of an agent project by reading its source, never running it.

``discover`` walks a project directory and records every prompt, model name,
sampling setting, limit, tool description and settings constant it can see
statically:

* Python files are parsed with :mod:`ast` (nothing is imported or executed).
* ``*.json`` config files are scanned with a span-tracking JSON reader.
* ``*.toml`` config files are read line by line (``[section]`` headers and
  ``key = value`` pairs, including ``\"\"\"`` multi-line strings) and every
  value is checked with :mod:`tomllib`.
* ``*.yaml``/``*.yml`` config files are read with a small safe subset reader:
  nested maps of scalars (plain, single- or double-quoted) and ``|``/``|-``
  block strings. Lists, anchors, flow collections and ``>`` folding are
  ignored, never guessed at.

Only config files whose name mentions prompt/config/settings/agent/llm/graph
are scanned, so ``package.json`` or ``pyproject.toml`` do not flood the
manifest.

Each knob records the exact source span of its literal, so ``apply`` can
rewrite only those bytes in a copy of the project.
"""

from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import json
import math
import re
import shutil
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

KINDS = ("prompt", "model", "temperature", "limit", "tool_description", "constant", "config")

MANIFEST_VERSION = 1

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", ".tox", ".nox", "venv", "env", ".idea",
    ".vscode", "site-packages", ".eggs",
}
_MESSAGE_CALLS = {"SystemMessage", "HumanMessage", "AIMessage", "SystemMessagePromptTemplate",
                  "HumanMessagePromptTemplate", "AIMessagePromptTemplate"}
_TEMPLATE_CALLS = {"from_template", "PromptTemplate", "ChatPromptTemplate"}
_ROLES = {"system", "human", "user", "ai", "assistant", "placeholder"}
_PROMPT_KWARGS = {"prompt", "system_prompt", "instructions", "system_message", "template"}
_MODEL_KWARGS = {"model", "model_name"}
_LIMIT_KWARGS = {"max_tokens", "max_iterations", "recursion_limit"}
_SETTINGS_MODULE = re.compile(r"prompt|config|settings", re.I)
_CONFIG_FILE = re.compile(r"prompt|config|settings|agent|llm|graph", re.I)
_UPPER = re.compile(r"^_?[A-Z][A-Z0-9_]*$")


class KnobError(ValueError):
    """A knob could not be discovered, matched or applied."""


@dataclass
class Knob:
    id: str
    file: str              # project-relative posix path
    line: int              # 1-based
    col: int               # 0-based character column
    end_line: int
    end_col: int
    offset: int            # byte offset of the literal in the file
    end_offset: int
    kind: str
    value: Any
    node: str | None = None    # enclosing function: the likely graph node
    qualname: str = "<module>"
    key: str | None = None     # kwarg / constant / config key that names the knob
    dynamic: bool = False      # f-string with expressions: visible, not tunable
    frozen: bool = False

    @property
    def tunable(self) -> bool:
        return not self.dynamic and not self.frozen

    def value_hash(self) -> str:
        blob = json.dumps([self.file, self.kind, self.value], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class Manifest:
    project_dir: str
    knobs: list[Knob] = field(default_factory=list)
    version: int = MANIFEST_VERSION

    def get(self, knob_id: str) -> Knob:
        for k in self.knobs:
            if k.id == knob_id:
                return k
        raise KnobError(f"unknown knob id: {knob_id}")

    def by_kind(self, kind: str) -> list[Knob]:
        return [k for k in self.knobs if k.kind == kind]

    def to_dict(self) -> dict:
        return {"version": self.version, "project_dir": self.project_dir,
                "knobs": [asdict(k) for k in self.knobs]}

    @classmethod
    def from_dict(cls, data: dict) -> "Manifest":
        if data.get("version") != MANIFEST_VERSION:
            raise KnobError(f"unsupported manifest version: {data.get('version')!r}")
        return cls(project_dir=data["project_dir"], knobs=[Knob(**k) for k in data["knobs"]])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Manifest":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- walking

def _skip_dir(name: str) -> bool:
    return name in _SKIP_DIRS or name.startswith(".venv") or name.endswith(".egg-info")


def _skip_file(name: str) -> bool:
    return name.startswith(".env")


def iter_project_files(root: Path):
    """Yield (path, relpath) for every file outside VCS/venv/cache dirs and .env files."""
    stack = [root]
    while stack:
        d = stack.pop()
        for p in sorted(d.iterdir(), key=lambda p: p.name):
            if p.is_dir():
                if not _skip_dir(p.name) and not p.is_symlink():
                    stack.append(p)
            elif not _skip_file(p.name):
                yield p, p.relative_to(root).as_posix()


def _matches(rel: str, patterns) -> bool:
    return any(fnmatch.fnmatch(rel, pat) for pat in patterns)


# --------------------------------------------------------------------------- spans

class _Lines:
    """Map ast (line, utf-8 byte col) positions to absolute byte offsets and char cols."""

    def __init__(self, data: bytes):
        self.data = data
        self.starts = [0]
        for i, b in enumerate(data):
            if b == 0x0A:
                self.starts.append(i + 1)

    def offset(self, line: int, bcol: int) -> int:
        return self.starts[line - 1] + bcol

    def char_col(self, line: int, bcol: int) -> int:
        start = self.starts[line - 1]
        return len(self.data[start:start + bcol].decode("utf-8", "replace"))

    def pos(self, offset: int) -> tuple[int, int]:
        lo, hi = 0, len(self.starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, self.char_col(lo + 1, offset - self.starts[lo])


# --------------------------------------------------------------------------- python

def _literal(node: ast.AST):
    """Return (value, dynamic) for a str/int/float literal node, or None."""
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or not isinstance(v, (str, int, float)):
            return None
        return v, False
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) \
            and isinstance(node.operand, ast.Constant) \
            and isinstance(node.operand.value, (int, float)) \
            and not isinstance(node.operand.value, bool):
        v = node.operand.value
        return (-v if isinstance(node.op, ast.USub) else v), False
    if isinstance(node, ast.JoinedStr):
        if any(isinstance(v, ast.FormattedValue) for v in node.values):
            return ast.unparse(node), True
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant)), False
    return None


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_tool_decorator(dec: ast.AST) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    return _call_name(target) == "tool"


class _PyScanner(ast.NodeVisitor):
    def __init__(self, rel: str, data: bytes, settings_module: bool):
        self.rel = rel
        self.lines = _Lines(data)
        self.settings_module = settings_module
        self.stack: list[tuple[str, bool]] = []   # (name, is_function)
        self.found: list[tuple[ast.AST, str, str | None, str, str | None]] = []
        self.seen: set[tuple[int, int]] = set()

    # context ----------------------------------------------------------
    def _qual(self) -> str:
        return ".".join(n for n, _ in self.stack) or "<module>"

    def _func(self) -> str | None:
        for name, is_func in reversed(self.stack):
            if is_func:
                return name
        return None

    def _add(self, node: ast.AST, kind: str, key: str | None, want=(str,)) -> None:
        lit = _literal(node)
        if lit is None:
            return
        value, dynamic = lit
        if not dynamic and not isinstance(value, want):
            return
        if isinstance(value, bool):
            return
        pos = (node.lineno, node.col_offset)
        if pos in self.seen:
            return
        self.seen.add(pos)
        self.found.append((node, kind, key, self._qual(), self._func()))

    # scopes -------------------------------------------------------------
    def _visit_func(self, node):
        if any(_is_tool_decorator(d) for d in node.decorator_list):
            doc = node.body[0] if node.body else None
            if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant) \
                    and isinstance(doc.value.value, str):
                self.stack.append((node.name, True))
                self._add(doc.value, "tool_description", "__doc__")
                self.stack.pop()
        for d in node.decorator_list:
            self.visit(d)
        for d in node.args.defaults + [d for d in node.args.kw_defaults if d is not None]:
            self.visit(d)
        self.stack.append((node.name, True))
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_ClassDef(self, node):
        for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
            self.visit(d)
        self.stack.append((node.name, False))
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    # module constants -------------------------------------------------------
    def _module_const(self, target, value):
        if self.stack or not self.settings_module or value is None:
            return
        if isinstance(target, ast.Name) and _UPPER.match(target.id):
            self._add(value, "constant", target.id, want=(str, int, float))

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            self._module_const(node.targets[0], node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        self._module_const(node.target, node.value)
        self.generic_visit(node)

    # calls --------------------------------------------------------------------
    def visit_Call(self, node):
        name = _call_name(node.func)
        if name in _MESSAGE_CALLS:
            if node.args:
                self._add(node.args[0], "prompt", name)
        if name in _TEMPLATE_CALLS or name == "from_messages":
            if node.args and name != "from_messages":
                self._add(node.args[0], "prompt", name)
        for kw in node.keywords:
            k = kw.arg
            if k is None:
                continue
            if k == "content" and name in _MESSAGE_CALLS:
                self._add(kw.value, "prompt", name)
            elif k in _PROMPT_KWARGS:
                self._add(kw.value, "prompt", k)
            elif k in _MODEL_KWARGS:
                self._add(kw.value, "model", k)
            elif k == "temperature":
                self._add(kw.value, "temperature", k, want=(int, float))
            elif k in _LIMIT_KWARGS:
                self._add(kw.value, "limit", k, want=(int,))
            elif k == "description":
                self._add(kw.value, "tool_description", k)
        self.generic_visit(node)

    def visit_Tuple(self, node):
        if len(node.elts) == 2 and isinstance(node.elts[0], ast.Constant) \
                and node.elts[0].value in _ROLES:
            self._add(node.elts[1], "prompt", node.elts[0].value)
        self.generic_visit(node)

    def visit_Dict(self, node):
        for k, v in zip(node.keys, node.values):
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                if k.value in _LIMIT_KWARGS:
                    self._add(v, "limit", k.value, want=(int,))
                elif k.value == "temperature":
                    self._add(v, "temperature", k.value, want=(int, float))
                elif k.value in _MODEL_KWARGS:
                    self._add(v, "model", k.value)
                elif k.value == "content" or k.value in _PROMPT_KWARGS:
                    self._add(v, "prompt", k.value)
        self.generic_visit(node)


def _scan_python(rel: str, data: bytes) -> list[dict]:
    tree = ast.parse(data, filename=rel)
    stem = rel.rsplit("/", 1)[-1][:-3]
    scanner = _PyScanner(rel, data, bool(_SETTINGS_MODULE.search(stem)))
    scanner.visit(tree)
    out = []
    lines = scanner.lines
    for node, kind, key, qual, func in scanner.found:
        value, dynamic = _literal(node)
        out.append(dict(
            file=rel, line=node.lineno, col=lines.char_col(node.lineno, node.col_offset),
            end_line=node.end_lineno, end_col=lines.char_col(node.end_lineno, node.end_col_offset),
            offset=lines.offset(node.lineno, node.col_offset),
            end_offset=lines.offset(node.end_lineno, node.end_col_offset),
            kind=kind, value=value, node=func, qualname=qual, key=key, dynamic=dynamic))
    out.sort(key=lambda d: d["offset"])
    return out


# --------------------------------------------------------------------------- json

_JSON_NUM = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?")
_JSON_WS = re.compile(r"[ \t\n\r]*")


def _scan_json_text(text: str) -> list[tuple[str, int, int, Any]]:
    """Return (dotted path, char start, char end, value) for every str/int/float scalar."""
    out: list[tuple[str, int, int, Any]] = []
    decoder = json.JSONDecoder()

    def ws(i):
        return _JSON_WS.match(text, i).end()

    def value(i, path):
        i = ws(i)
        c = text[i:i + 1]
        if c == "{":
            i = ws(i + 1)
            if text[i] == "}":
                return i + 1
            while True:
                if text[i] != '"':
                    raise KnobError("bad JSON object key")
                key, i = json.decoder.scanstring(text, i + 1)
                i = ws(i)
                if text[i] != ":":
                    raise KnobError("expected ':' in JSON")
                i = value(i + 1, f"{path}.{key}" if path else key)
                i = ws(i)
                if text[i] == ",":
                    i = ws(i + 1)
                    continue
                if text[i] == "}":
                    return i + 1
                raise KnobError("bad JSON object")
        if c == "[":
            i = ws(i + 1)
            if text[i] == "]":
                return i + 1
            n = 0
            while True:
                i = ws(value(i, f"{path}[{n}]"))
                n += 1
                if text[i] == ",":
                    i += 1
                    continue
                if text[i] == "]":
                    return i + 1
                raise KnobError("bad JSON array")
        if c == '"':
            s, end = json.decoder.scanstring(text, i + 1)
            out.append((path, i, end, s))
            return end
        m = _JSON_NUM.match(text, i)
        if m and m.end() > i:
            v = json.loads(m.group())
            out.append((path, i, m.end(), v))
            return m.end()
        v, end = decoder.raw_decode(text, i)   # true/false/null
        return end

    end = ws(value(0, ""))
    if end != len(text):
        raise KnobError("trailing data after JSON document")
    return out


# --------------------------------------------------------------------------- toml

_TOML_HEADER = re.compile(r"^\s*\[([A-Za-z0-9_.\- \"]+)\]\s*(?:#.*)?$")
_TOML_KV = re.compile(r"^(\s*)([A-Za-z0-9_\-]+)\s*=\s*")


def _toml_value(raw: str):
    return tomllib.loads("x = " + raw)["x"]


def _scan_toml_text(text: str) -> list[tuple[str, int, int, Any]]:
    tomllib.loads(text)   # the whole file must be valid TOML
    out = []
    section = ""
    pos = 0
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        line = lines[i]
        line_start = pos
        pos += len(line)
        i += 1
        body = line.rstrip("\r\n")
        h = _TOML_HEADER.match(body)
        if h and not body.lstrip().startswith("[["):
            section = h.group(1).replace('"', "").replace(" ", "")
            continue
        m = _TOML_KV.match(body)
        if not m:
            continue
        key = f"{section}.{m.group(2)}" if section else m.group(2)
        vstart = line_start + m.end()
        rest = body[m.end():]
        for q in ('"""', "'''"):
            if rest.startswith(q):
                close = text.find(q, vstart + 3)
                if close < 0:
                    raise KnobError("unterminated TOML multi-line string")
                end = close + 3
                while text[end:end + 1] == q[0]:   # up to two extra quotes belong to the value
                    end += 1
                raw = text[vstart:end]
                out.append((key, vstart, end, _toml_value(raw)))
                consumed = text[line_start:end]
                extra = consumed.count("\n") - (1 if line.endswith("\n") else 0)
                for _ in range(extra):
                    pos += len(lines[i])
                    i += 1
                break
        else:
            # single-line value: strip a trailing comment outside strings
            raw = rest
            for cut in range(1, len(rest) + 1):
                cand = rest[:cut].rstrip()
                try:
                    v = _toml_value(cand)
                except tomllib.TOMLDecodeError:
                    continue
                tail = rest[cut:].strip()
                if tail and not tail.startswith("#"):
                    continue
                raw = cand
                break
            else:
                continue
            if isinstance(v, bool) or not isinstance(v, (str, int, float)):
                continue
            out.append((key, vstart, vstart + len(raw), v))
    return out


# --------------------------------------------------------------------------- yaml subset

_YAML_KEY = re.compile(r"^(\s*)([A-Za-z0-9_\-.]+|\"[^\"]*\"|'[^']*')\s*:(?:\s+|$)")
_YAML_SPECIAL = re.compile(r"^(?:true|false|yes|no|on|off|null|~|[-+]?\d[\d_.eE+-]*|\.inf|\.nan)$", re.I)


def _yaml_scalar(raw: str):
    """Parse one YAML scalar of the supported subset. Returns (value, style) or None."""
    s = raw.strip()
    if not s:
        return None
    if s[0] == '"':
        try:
            return json.loads(s), "double"
        except json.JSONDecodeError:
            return None
    if s[0] == "'":
        if len(s) >= 2 and s[-1] == "'":
            return s[1:-1].replace("''", "'"), "single"
        return None
    if s[0] in "[{&*!|>%@`":
        return None
    low = s.lower()
    if low in ("true", "false", "yes", "no", "on", "off", "null", "~"):
        return None
    try:
        return int(s.replace("_", "")), "plain"
    except ValueError:
        pass
    try:
        return float(s), "plain"
    except ValueError:
        pass
    return s, "plain"


def _strip_yaml_comment(rest: str) -> str:
    if rest[:1] in ('"', "'"):
        q = rest[0]
        j = 1
        while j < len(rest):
            if q == '"' and rest[j] == "\\":
                j += 2
                continue
            if rest[j] == q:
                if q == "'" and rest[j + 1:j + 2] == "'":
                    j += 2
                    continue
                return rest[:j + 1]
            j += 1
        return rest
    m = re.search(r"\s#", rest)
    return (rest[:m.start()] if m else rest).rstrip()


def _scan_yaml_text(text: str) -> list[tuple[str, int, int, Any]]:
    out = []
    lines = text.splitlines(keepends=True)
    starts = []
    p = 0
    for ln in lines:
        starts.append(p)
        p += len(ln)
    path: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        body = lines[i].rstrip("\r\n")
        stripped = body.strip()
        if not stripped or stripped.startswith("#") or stripped in ("---", "..."):
            i += 1
            continue
        m = _YAML_KEY.match(body)
        if not m:
            i += 1        # list items and anything else outside the subset
            continue
        indent = len(m.group(1))
        key = m.group(2).strip("\"'")
        while path and path[-1][0] >= indent:
            path.pop()
        dotted = ".".join([k for _, k in path] + [key])
        rest = body[m.end():]
        vstart = starts[i] + m.end()
        value_text = _strip_yaml_comment(rest)
        if not value_text:
            path.append((indent, key))
            i += 1
            continue
        if value_text in ("|", "|-"):
            j = i + 1
            block: list[str] = []
            bindent = None
            last = i
            while j < len(lines):
                raw = lines[j].rstrip("\r\n")
                if raw.strip() == "":
                    block.append("")
                    j += 1
                    continue
                ind = len(raw) - len(raw.lstrip(" "))
                if ind <= indent:
                    break
                if bindent is None:
                    bindent = ind
                if ind < bindent:
                    break
                block.append(raw[bindent:])
                last = j
                j += 1
            if bindent is None:
                i = j
                continue
            block = block[:last - i]
            val = "\n".join(block)
            if value_text == "|":
                val += "\n"
            end = starts[last] + len(lines[last].rstrip("\r\n"))
            out.append((dotted, vstart, end, val))
            i = last + 1
            continue
        parsed = _yaml_scalar(value_text)
        if parsed is not None:
            out.append((dotted, vstart, vstart + len(value_text), parsed[0]))
        i += 1
    return out


def _yaml_style(src: str) -> str:
    if src.startswith("|"):
        return "block"
    if src.startswith('"'):
        return "double"
    if src.startswith("'"):
        return "single"
    return "plain"


# --------------------------------------------------------------------------- config dispatch

def _config_scanner(name: str):
    low = name.lower()
    if not _CONFIG_FILE.search(low.rsplit(".", 1)[0]):
        return None
    if low.endswith(".json"):
        return _scan_json_text
    if low.endswith(".toml"):
        return _scan_toml_text
    if low.endswith((".yaml", ".yml")):
        return _scan_yaml_text
    return None


def _scan_config(rel: str, data: bytes) -> list[dict]:
    scanner = _config_scanner(rel.rsplit("/", 1)[-1])
    text = data.decode("utf-8")
    lines = _Lines(data)
    out = []
    for key, start, end, value in scanner(text):
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            continue
        boff = len(text[:start].encode("utf-8"))
        eoff = boff + len(text[start:end].encode("utf-8"))
        line, col = lines.pos(boff)
        end_line, end_col = lines.pos(eoff)
        out.append(dict(file=rel, line=line, col=col, end_line=end_line, end_col=end_col,
                        offset=boff, end_offset=eoff, kind="config", value=value, node=None,
                        qualname=key, key=key, dynamic=False))
    return out


# --------------------------------------------------------------------------- discover

def _assign_ids(raw: list[dict]) -> list[Knob]:
    counts: dict[tuple, int] = {}
    knobs = []
    for d in raw:
        base = (d["file"], d["qualname"], d["kind"])
        n = counts.get(base, 0)
        counts[base] = n + 1
        knobs.append(Knob(id=f"{d['file']}:{d['qualname']}:{d['kind']}:{n}", **d))
    return knobs


def discover(project_dir: str | Path, include=None, exclude=None,
             previous: Manifest | None = None) -> Manifest:
    """Statically scan ``project_dir`` and return its knob manifest.

    ``include``/``exclude`` are glob patterns over project-relative posix paths.
    ``previous`` carries user ``frozen`` flags over (matched by id, then by
    file + value hash). Files that fail to parse are skipped, never executed.
    """
    root = Path(project_dir).resolve()
    if not root.is_dir():
        raise KnobError(f"not a directory: {project_dir}")
    raw: list[dict] = []
    for path, rel in iter_project_files(root):
        if include and not _matches(rel, include):
            continue
        if exclude and _matches(rel, exclude):
            continue
        name = path.name
        try:
            if name.endswith(".py"):
                raw.extend(_scan_python(rel, path.read_bytes()))
            elif _config_scanner(name):
                raw.extend(_scan_config(rel, path.read_bytes()))
        except (SyntaxError, ValueError, UnicodeDecodeError, tomllib.TOMLDecodeError,
                IndexError):
            continue
    manifest = Manifest(project_dir=str(root), knobs=_assign_ids(raw))
    if previous is not None:
        carry_frozen(manifest, previous)
    return manifest


def carry_frozen(manifest: Manifest, previous: Manifest) -> Manifest:
    """Copy ``frozen`` flags from ``previous``: by id first, else by file + value hash."""
    by_id = {k.id: k.frozen for k in previous.knobs}
    by_hash: dict[str, bool] = {}
    for k in previous.knobs:
        by_hash.setdefault(k.value_hash(), k.frozen)
    for k in manifest.knobs:
        if k.id in by_id:
            k.frozen = by_id[k.id]
        elif k.value_hash() in by_hash:
            k.frozen = by_hash[k.value_hash()]
    return manifest


def rediscover(project_dir: str | Path, manifest_path: str | Path, include=None,
               exclude=None) -> Manifest:
    """Re-scan and save, keeping the user's frozen edits from the manifest on disk."""
    mp = Path(manifest_path)
    previous = Manifest.load(mp) if mp.exists() else None
    manifest = discover(project_dir, include=include, exclude=exclude, previous=previous)
    manifest.save(mp)
    return manifest


# =========================================================================== apply

_LIMIT_KEYS = _LIMIT_KWARGS | {"max_retries", "max_steps"}


@dataclass
class ApplyResult:
    out_dir: str
    diff: str
    changed: list[str]


def _leaf(key: str | None) -> str:
    if not key:
        return ""
    return re.split(r"[.\[]", key.rsplit(".", 1)[-1])[0].lower()


def _check_value(k: Knob, new: Any) -> Any:
    """Type- and bounds-check ``new`` for knob ``k``; return the value to write."""
    if isinstance(new, bool):
        raise KnobError(f"{k.id}: booleans are not knob values")
    leaf = _leaf(k.key)
    numeric = isinstance(k.value, (int, float))
    if k.kind == "temperature" or (numeric and leaf == "temperature"):
        if not isinstance(new, (int, float)) or not math.isfinite(new) or not 0 <= new <= 2:
            raise KnobError(f"{k.id}: temperature must be a number in [0, 2], got {new!r}")
        return float(new) if isinstance(k.value, float) else new
    if k.kind == "limit" or (numeric and leaf in _LIMIT_KEYS):
        if not isinstance(new, int) or new < 1:
            raise KnobError(f"{k.id}: limit must be an integer >= 1, got {new!r}")
        return new
    if isinstance(k.value, str):
        if not isinstance(new, str):
            raise KnobError(f"{k.id}: expected a string, got {type(new).__name__}")
        if (k.kind == "model" or leaf in _MODEL_KWARGS) and (not new.strip() or "\n" in new):
            raise KnobError(f"{k.id}: model must be a non-empty single-line string")
        return new
    if isinstance(k.value, int):
        if not isinstance(new, int):
            raise KnobError(f"{k.id}: expected an integer, got {new!r}")
        return new
    if not isinstance(new, (int, float)) or not math.isfinite(new):
        raise KnobError(f"{k.id}: expected a finite number, got {new!r}")
    return float(new)


def _num(v) -> str:
    return repr(v) if isinstance(v, float) else str(v)


# ---- python literals

_PY_PREFIX = re.compile(r"^([A-Za-z]*)('''|\"\"\"|'|\")")


def _py_escape(s: str, quote: str, triple: bool) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == quote:
            out.append("\\" + ch)
        elif ch == "\n":
            out.append("\n" if triple else "\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif o < 0x20 or o == 0x7F or 0xD800 <= o <= 0xDFFF:
            out.append(f"\\x{o:02x}" if o < 0x100 else f"\\u{o:04x}")
        else:
            out.append(ch)
    return "".join(out)


def _render_py(original_src: str, new: Any) -> str:
    if not isinstance(new, str):
        return _num(new)
    m = _PY_PREFIX.match(original_src)
    quote = m.group(2) if m else '"'
    lit = quote + _py_escape(new, quote[0], len(quote) == 3) + quote   # r/u/f prefixes dropped
    if ast.literal_eval(lit) != new:
        raise KnobError("internal: python literal round-trip failed")
    return lit


# ---- config literals

def _render_json(_src: str, new: Any) -> str:
    return json.dumps(new, ensure_ascii=False) if isinstance(new, str) else _num(new)


def _toml_basic(s: str, multiline: bool) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\n" if multiline else "\\n")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    return "".join(out)


def _render_toml(src: str, new: Any) -> str:
    if not isinstance(new, str):
        return _num(new)
    if src.startswith(('"""', "'''")):
        lead = src[3:5] if src[3:5] == "\r\n" else ("\n" if src[3:4] == "\n" else "")
        return '"""' + lead + _toml_basic(new, True) + '"""'
    if src.startswith("'") and "'" not in new \
            and not any(ord(c) < 0x20 or ord(c) == 0x7F for c in new):
        return "'" + new + "'"
    return '"' + _toml_basic(new, False) + '"'


_YAML_PLAIN_OK = re.compile(r"^[A-Za-z0-9_./(][^\n#:'\"]*$")


def _render_yaml(src: str, new: Any) -> str:
    if not isinstance(new, str):
        return _num(new)
    style = _yaml_style(src)
    if style == "block":
        body = new[:-1] if new.endswith("\n") else new
        eol = "\r\n" if "\r\n" in src else "\n"
        lines = src.split("\n")
        indent = next((ln[:len(ln) - len(ln.lstrip(" "))] for ln in lines[1:] if ln.strip()),
                      "  ")
        if body and not body.startswith((" ", "\n")) and not body.endswith(("\n", " ")) \
                and "\r" not in body and all(ord(c) >= 0x20 or c in "\n\t" for c in body):
            header = "|" if new.endswith("\n") else "|-"
            return header + eol + eol.join(indent + ln if ln else "" for ln in body.split("\n"))
        style = "double"
    if style == "plain" and _YAML_PLAIN_OK.match(new) and new == new.strip() \
            and not _YAML_SPECIAL.match(new):
        return new
    if style == "single" and all(ord(c) >= 0x20 for c in new):
        return "'" + new.replace("'", "''") + "'"
    return json.dumps(new, ensure_ascii=False)


def _renderer(rel: str):
    low = rel.lower()
    if low.endswith(".py"):
        return _render_py
    if low.endswith(".json"):
        return _render_json
    if low.endswith(".toml"):
        return _render_toml
    return _render_yaml


def _scan_file(rel: str, data: bytes) -> dict[str, Knob]:
    raw = _scan_python(rel, data) if rel.endswith(".py") else _scan_config(rel, data)
    return {k.id: k for k in _assign_ids(raw)}


def _same(a, b) -> bool:
    return type(a) is type(b) and a == b


def _copy_ignore(directory, names):
    base = Path(directory)
    return [n for n in names
            if (_skip_dir(n) if (base / n).is_dir() else _skip_file(n))]


def apply(project_dir: str | Path, out_dir: str | Path, manifest: Manifest,
          values: dict[str, Any]) -> ApplyResult:
    """Copy ``project_dir`` to ``out_dir`` and write ``values`` (knob id -> new value).

    Only the recorded source span of each knob is rewritten, keeping its
    quoting style where the new value allows it. Everything is validated
    before anything is written: unknown, frozen or dynamic ids, mistyped or
    out-of-bounds values, and spans whose source no longer holds the
    manifest's value raise :class:`KnobError`. Each rewritten file is parsed
    again and every new value must read back exactly. The original project is
    never modified. The copy skips VCS dirs, virtualenvs, caches and ``.env*``.
    """
    root = Path(project_dir).resolve()
    out = Path(out_dir).resolve()
    if out == root or root in out.parents:
        raise KnobError("out_dir must be outside the project directory")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise KnobError(f"out_dir exists and is not an empty directory: {out}")

    edits: dict[str, list[tuple[Knob, Any]]] = {}
    for knob_id, new in values.items():
        k = manifest.get(knob_id)
        if k.frozen:
            raise KnobError(f"{knob_id}: knob is frozen")
        if k.dynamic:
            raise KnobError(f"{knob_id}: knob is dynamic (f-string with expressions)")
        edits.setdefault(k.file, []).append((k, _check_value(k, new)))

    new_files: dict[str, bytes] = {}
    diffs: list[str] = []
    for rel in sorted(edits):
        src_path = root / rel
        if not src_path.is_file():
            raise KnobError(f"{rel}: file is missing")
        data = src_path.read_bytes()
        current = _scan_file(rel, data)
        render = _renderer(rel)
        result = data
        ordered = sorted(edits[rel], key=lambda e: e[0].offset, reverse=True)
        prev_start = None
        for k, new in ordered:
            cur = current.get(k.id)
            if cur is None or (cur.offset, cur.end_offset) != (k.offset, k.end_offset) \
                    or not _same(cur.value, k.value):
                raise KnobError(f"{k.id}: source changed since discovery; run discover again")
            if prev_start is not None and k.end_offset > prev_start:
                raise KnobError(f"{k.id}: overlapping knob spans")
            prev_start = k.offset
            old_src = data[k.offset:k.end_offset].decode("utf-8")
            lit = render(old_src, new).encode("utf-8")
            result = result[:k.offset] + lit + result[k.end_offset:]
        try:
            if rel.endswith(".py"):
                ast.parse(result, filename=rel)
            after = _scan_file(rel, result)
        except (SyntaxError, ValueError, UnicodeDecodeError, tomllib.TOMLDecodeError,
                IndexError) as e:
            raise KnobError(f"{rel}: rewritten file does not parse: {e}") from e
        for k, new in edits[rel]:
            got = after.get(k.id)
            if got is None or not _same(got.value, new):
                raise KnobError(f"{k.id}: rewritten value did not read back as {new!r}")
        new_files[rel] = result
        a = data.decode("utf-8").splitlines(keepends=True)
        b = result.decode("utf-8").splitlines(keepends=True)
        for line in difflib.unified_diff(a, b, fromfile=f"a/{rel}", tofile=f"b/{rel}"):
            diffs.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")

    if out.exists():
        out.rmdir()
    shutil.copytree(root, out, ignore=_copy_ignore, symlinks=True)
    for rel, blob in new_files.items():
        (out / rel).write_bytes(blob)
    return ApplyResult(out_dir=str(out), diff="".join(diffs), changed=sorted(new_files))
