import json
import shutil
import subprocess
from pathlib import Path

import pytest

from meta_harness.replay import ERROR_PATTERNS

ROOT = Path(__file__).resolve().parent.parent


def test_hooks_json_registers_the_module():
    config = json.loads((ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    assert config["modules"] == ["./harness.ts"]


def test_hook_module_exports_register():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "export const register" in source


def test_every_hook_is_wrapped_so_it_fails_open():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # A learning system that can break a session is worse than no learning system.
    for event in ("tool.call", "tool.check", "prompt.section", "turn.complete"):
        if f"'{event}'" in source:
            assert "safely(" in source, f"{event} must be wrapped"


def test_plugin_declares_the_hooks_directory():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert "hooks" in manifest
    assert manifest["hooks"] == "./hooks/hooks.json"


def test_observer_writes_one_line_per_failure():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # One file per session (not a single shared file): $.fs has no append/lock primitive, so a
    # shared file would need a non-atomic exists/read/write cycle that concurrent sessions race.
    assert "observed-" in source and ".jsonl" in source
    assert "'tool.call'" in source


def test_observer_records_repeats_as_well_as_errors():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "tool_error" in source and "repeat" in source


def test_ts_cause_patterns_match_python_error_patterns():
    """The interface promises {ts, tool, kind, cause, input}; `cause` must be the same slug
    meta_harness.replay._cause() would compute from the same text, in the same pattern order
    (UnicodeEncodeError before the decode/charmap/cp1252 pattern is load-bearing), or dedupe
    between hook-observed and replay-mined episodes silently breaks.
    """
    import re

    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    match = re.search(r"CAUSE_PATTERNS[^=]*=\s*\[(.*?)\];", source, re.DOTALL)
    assert match, "CAUSE_PATTERNS array not found in hooks/harness.ts"
    pairs = re.findall(
        r"\[\s*'((?:[^'\\]|\\.)*)'\s*,\s*'((?:[^'\\]|\\.)*)'\s*\]", match.group(1)
    )
    assert pairs, "CAUSE_PATTERNS array parsed empty"
    ts_patterns = [(pattern.replace("\\'", "'"), slug) for pattern, slug in pairs]
    assert ts_patterns == list(ERROR_PATTERNS)


def test_observer_behaviour_via_node():
    """Substring greps over the source cannot prove observe() ever performs a real write: a
    reviewer commented out the dollar.fs.write call inside observe() and every grep-based test
    in this file still passed. hooks/harness.observer.test.mts drives registerObserver() with a
    fake `$` whose fs methods record calls, and asserts on the actual JSON line written.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.ts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.observer.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.observer.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
