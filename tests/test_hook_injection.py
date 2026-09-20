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
    harness_source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    rules_source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    # The type filter lives once, in the shared loadInstalled() reader (rules.ts), and
    # harness.ts asks it for 'injection' rows specifically rather than re-filtering itself.
    assert "entry.type !== type" in rules_source
    assert "loadInstalled(dollar, home, 'injection')" in harness_source


def test_injection_reuses_the_shared_registry_reader():
    harness_source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    rules_source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    # A second, divergent installed.json reader in harness.ts would drift from rules.ts's
    # fail-open semantics on the next edit to either; both layers must call the one reader.
    assert "export async function loadInstalled" in rules_source
    assert "loadInstalled" in harness_source
    assert "JSON.parse(await dollar.fs.read(registry))" not in harness_source


def test_injection_text_is_capped_per_artifact_and_in_total():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "INJECTION_TEXT_CAP" in source
    assert "INJECTION_TOTAL_CAP" in source
    assert "truncate(" in source


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
