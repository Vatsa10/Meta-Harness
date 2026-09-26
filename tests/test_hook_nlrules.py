import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_parse_stop_instruction_is_exported():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    assert "export function parseStopInstruction" in source
    assert "export function addSessionRule" in source


def test_session_rules_are_written_to_the_pending_proposal_file():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "pending-session-rules.json" in source


def test_nlrules_behaviour_via_node():
    """A grep over the source cannot prove parseStopInstruction discriminates a real instruction
    from an ordinary question, or that a session rule actually denies a matching call while
    never reaching installed.json: it would pass just as well against a parser that matches
    anywhere in the text, or a tool.check that skips the session-rule check entirely.
    hooks/harness.nlrules.test.mts drives the real registerRules() handlers with a fake `$` and
    asserts on the actual decisions and on-disk writes.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.nlrules.test.mts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.nlrules.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.nlrules.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
