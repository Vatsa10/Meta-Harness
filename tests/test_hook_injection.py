import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_prompt_section_is_registered_and_wrapped():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "'prompt.section'" in source
    assert "safely('prompt.section'" in source


def test_injection_returns_null_when_nothing_matches():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # Returning null leaves the section out entirely, which is why this costs nothing
    # on turns where no artifact is relevant.
    assert "text: null" in source


def test_injection_records_what_it_injected():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # The hook knows what it placed, so use is recorded by construction rather than by
    # asking the model to self-report a retrieval it might forget.
    assert "injected" in source


def test_injection_never_injects_non_injection_artifact_types():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "entry.type !== 'injection'" in source


def test_injection_behaviour_via_node():
    """A grep over the source cannot prove registerInjection() actually reads the registry,
    matches triggers, and returns {text: null} vs {text: ...}: it would pass just as well
    against a handler that always returns {text: null}, or one that injects a 'rule' or
    'skill' artifact's payload. hooks/harness.injection.test.mts drives the real
    registerInjection() handler with a fake `$` and a fake installed-artifact registry and
    asserts on the actual returned text and on the observe() side effect.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.injection.test.mts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.injection.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.injection.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
