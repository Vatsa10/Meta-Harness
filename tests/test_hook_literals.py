"""Static check: every `on(...)` in a shipped hooks module passes a function LITERAL as its hook.

Claude Code's hooks loader rejects a module whose `on("<event>", hook)` hook argument is not a
function literal (or the name of one) written in the call; `on('tool.call', afterCall(...))`
failed the whole module in a real session, so no hook ran. The unit fakes accept any function
value and cannot catch this, so it is checked here against the TypeScript source itself.
"""

from __future__ import annotations

import re
from pathlib import Path

HOOKS = Path(__file__).resolve().parent.parent / "hooks"

LITERAL = re.compile(
    r"^(async\s+)?(\([^()]*(\([^()]*\)[^()]*)*\)\s*(:\s*[^=]+?)?\s*=>|[A-Za-z_$][\w$]*\s*=>|function\b)"
)


def _strip_comments(src: str) -> str:
    """Blank out comments and regex literals, keeping string contents and offsets intact."""
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in src[i:j]))
            i = j
        elif c in "'\"`":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i : j + 1])
            i = j + 1
        elif c == "/" and re.search(r"(^|[=(,:;!&|?{}\[]|\breturn|\btypeof)\s*$", "".join(out[-40:])):
            # a regex literal: blanked, since its body may hold quotes or slashes
            j, in_class = i + 1, False
            while j < n and (src[j] != "/" or in_class):
                if src[j] == "\\":
                    j += 1
                elif src[j] == "[":
                    in_class = True
                elif src[j] == "]":
                    in_class = False
                j += 1
            out.append(" " * (j + 1 - i))
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _args(src: str, start: int) -> list[str]:
    """Top-level comma-separated arguments of the call whose '(' is at `start`."""
    depth, i, parts, cur = 0, start, [], start + 1
    while i < len(src):
        c = src[i]
        if c in "'\"`":
            j = i + 1
            while j < len(src) and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            i = j
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                parts.append(src[cur:i])
                return [p.strip() for p in parts]
        elif c == "," and depth == 1:
            parts.append(src[cur:i])
            cur = i + 1
        i += 1
    raise AssertionError("unterminated on( call")


def on_calls(src: str) -> list[tuple[int, list[str]]]:
    clean = _strip_comments(src)
    calls = []
    for m in re.finditer(r"(?<![\w$.])on\s*\(", clean):
        calls.append((clean.count("\n", 0, m.start()) + 1, _args(clean, m.end() - 1)))
    return calls


def violations(src: str) -> list[str]:
    bad = []
    for line, args in on_calls(src):
        if len(args) != 2 or not re.fullmatch(r"(['\"]).+\1", args[0]):
            bad.append(f"line {line}: not on('<event>', hook): {args!r}")
        elif not LITERAL.match(args[1]):
            bad.append(f"line {line}: hook is not a function literal: {args[1][:80]!r}")
    return bad


def _modules() -> list[Path]:
    return [p for p in HOOKS.glob("*.ts") if ".test." not in p.name]


def test_every_on_hook_is_a_function_literal():
    found = 0
    for path in _modules():
        src = path.read_text(encoding="utf-8")
        found += len(on_calls(src))
        assert violations(src) == [], f"{path.name}: {violations(src)}"
    assert found >= 8, "expected the harness's registrations to be found"


def test_checker_rejects_wrapper_calls():
    bad = "on('tool.call', afterCall('tool.call', async (d, e, o) => {}));"
    assert violations(bad)
    assert violations("on('x', safely('x', async () => 1));")
    good = "on('tool.call', async (dollar: any, event: any, next: any) => guard(next));"
    assert violations(good) == []
    assert violations("register(on); // on(x)") == []
