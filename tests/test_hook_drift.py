import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_drift_module_exports_the_config_reader():
    source = (ROOT / "hooks" / "drift.ts").read_text(encoding="utf-8")
    assert "export async function loadDriftConfig" in source
    assert "export function shouldWarn" in source


def test_drift_never_touches_prompt_section():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    body = source.split("export function registerDrift", 1)[1].split("export function registerBootstrap", 1)[0]
    assert "'prompt.section'" not in body
    assert "registerDrift(add)" in source


def test_no_enabling_drift_config_ships():
    """The ship gate refused every judge (tools/tune_drift.py), so no drift.json that enables the
    feature may be bundled anywhere in the repository."""
    for path in ROOT.rglob("drift.json"):
        if ".git" in path.parts or "worktrees" in path.parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        assert not (isinstance(data, dict) and data.get("enabled") is True), f"{path} enables drift"


def test_drift_behaviour_via_node():
    """Drives the real registerDrift() handlers, through the full register() chain, with events
    shaped as the claude-code-2.1.283 declarations define them: the note appears once per
    stretch, only when drift.json explicitly enables it, and next is called exactly once."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.drift.test.mts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.drift.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.drift.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
