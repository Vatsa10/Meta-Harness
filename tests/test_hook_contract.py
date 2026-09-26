import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_contract_behaviour_via_node():
    """Two cross-cutting contracts every hook in this plugin must honour, driven with real-shaped
    events per the recovered claude-code-2.1.283.d.ts declarations:

    1. register()'s full handler set must never change a prompt.section it does not own. Before
       fix round 1, registerBootstrap and registerInjection both listened on prompt.section with
       no section-name filter and could overwrite or delete any section of the system prompt
       (CRITICAL 1 and CRITICAL 3). This is a regression test: it fails against the pre-fix
       commit and passes here.
    2. Multiple tool.call handlers from different registerX functions compose in production
       (each runs, the tool underneath runs exactly once), driven through a genuine chaining
       fake rather than the single-slot fakes other test files here started from.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.contract.test.mts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.contract.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.contract.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
