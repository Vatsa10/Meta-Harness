import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_bootstrap_is_registered_on_prompt_submit_and_wrapped():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "export function registerBootstrap" in source
    # prompt.section has no event.text at all and returning {text} from it replaces a
    # system-prompt section; the line is attached as prompt.submit context instead.
    assert "safely('prompt.submit:bootstrap'" in source
    assert "registerBootstrap(on)" in source


def test_bootstrap_reads_the_real_waste_report_schema():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # meta_harness/waste.py's waste_report() nests calls_burned under corrections; sessions and
    # calls_burned are NOT both top-level.
    assert "waste?.corrections?.calls_burned" in source


def test_bootstrap_only_marks_itself_done_after_attaching():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # The marker write must happen after next(...) resolves, not before, so a missing report or
    # a transient failure never gets marked "done" without the line ever having been said.
    assert "const result = await next(" in source


def test_bootstrap_gates_on_a_marker_file_not_memory():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "bootstrap.json" in source


def test_bootstrap_behaviour_via_node():
    """A grep over the source cannot prove registerBootstrap() actually gates on bootstrap.json,
    reads waste.json, or fails open on a throw: it would pass just as well against a handler
    that always returns the line, never writes the marker, or drops the `safely` wrapper.
    hooks/harness.bootstrap.test.mts drives the real registerBootstrap() handler with a fake `$`
    and asserts on the actual returned text and the marker file it writes.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.bootstrap.test.mts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.bootstrap.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.bootstrap.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
