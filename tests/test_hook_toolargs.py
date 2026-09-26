"""tool.call readers must read the tool's arguments through `toolArgs`, never `event.input`.

On `tool.call` Claude Code puts the arguments at the top level of the event; only `tool.check`
nests them under `input`. Reading `event.input` on tool.call is always empty live: read-tracking
recorded nothing and the built-in read-before-edit rule denied every Edit.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_hook_literals import _args, _strip_comments

ROOT = Path(__file__).resolve().parent.parent


def tool_call_bodies(src: str) -> list[str]:
    """The hook of every `add('tool.call', ...)` / `on('tool.call', ...)` registration."""
    clean = _strip_comments(src)
    bodies = []
    for m in re.finditer(r"(?<![\w$.])(?:add|on)\s*\(\s*(['\"])tool\.call\1", clean):
        paren = clean.index("(", m.start())
        bodies.append(_args(clean, paren)[1])
    return bodies


def direct_input_reads(src: str) -> list[str]:
    return [body[:80] for body in tool_call_bodies(src) if re.search(r"\.\s*input\b|\[\s*['\"]input['\"]\s*\]", body)]


def test_no_tool_call_reader_reads_event_input_directly():
    src = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert len(tool_call_bodies(src)) >= 3, "expected the observer, read-tracking and drift tool.call hooks"
    assert direct_input_reads(src) == []


def test_the_static_check_catches_a_direct_read():
    bad = "add('tool.call', async (io, event, next) => guardAfter('x', io, event, next, async () => { f(event?.input); }));"
    assert direct_input_reads(bad)
    good = "add('tool.call', async (io, event, next) => guardAfter('x', io, event, next, async () => { f(toolArgs(event)); }));"
    assert direct_input_reads(good) == []


def test_tool_call_shape_behaviour_via_node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.toolargs.test.mts behavioural test")
    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.toolargs.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
