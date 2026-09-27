import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_rules_module_exists_and_exports_the_evaluator():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    assert "export function evaluateRule" in source
    assert "export async function loadRules" in source


def test_no_built_in_read_before_edit_rule_ships():
    # A live test proved Claude Code's own engine already denies an Edit/Write of a file not
    # read this session, before the plugin ever sees the call. A plugin-side copy of that check
    # only added failure modes (every Edit denied, every new-file Write denied, Write-then-Edit
    # denied) on top of an enforcement that already existed, so it is retired.
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    assert "kind: 'read-before-edit'" not in source
    assert "readPaths" not in source


def test_deny_carries_a_reason_and_the_artifact_id():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    assert "reason" in source
    assert "artifactId" in source or "artifact_id" in source


def test_tool_check_is_registered_and_wrapped():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "'tool.check'" in source
    assert "guardBefore('tool.check'" in source


def test_rules_behaviour_via_node():
    """A grep over the source cannot prove evaluateRule/loadRules/tool.check actually deny or
    allow anything: it would pass just as well against a rules.ts whose functions always return
    {deny: false}, or a harness.ts that registers 'tool.check' but never calls the evaluator.
    hooks/harness.rules.test.mts drives the real registerRules() handlers with a fake `$` and a
    fake installed-artifact registry and asserts on the actual decisions returned.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.rules.test.mts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.rules.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.rules.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
