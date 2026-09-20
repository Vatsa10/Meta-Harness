"""The seam: an artifact built by the real producer must be enforced by the real hooks.

Every other test on either side of this seam supplies its own fixture, and that is how the loop
shipped broken: `propose_artifact` emitted an `origin` with no `tools` and no `triggers`, the hook
tests hand-wrote both, and each side passed while no learned artifact could ever fire. This test
runs the real `propose_artifact`, installs the result through the real `HarnessStore`, and then
shells out to `hooks/harness.loop.test.mts`, which drives the real hook handlers against that
on-disk home.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from meta_harness.harness_store import HarnessStore
from meta_harness.learn import FailureClass, propose_artifact

ROOT = Path(__file__).resolve().parent.parent

REPLAY = {
    "instruction": "run the script",
    "files": {"broken.py": "print(1)"},
    "expect": {"no_thrash": {"tool": "Bash", "window": 6, "threshold": 4}},
    "_origin": {"kind": "thrash", "session": "sess-1", "turn": 4, "signature": "thrash:Bash"},
}


def _produce(artifact_type: str, payload: str, signature: str, kind: str):
    failure = FailureClass(signature=signature, kind=kind, tool="Bash", count=9)
    replay = dict(REPLAY)
    replay["_origin"] = dict(REPLAY["_origin"], signature=signature, kind=kind)
    model = lambda prompt: f"TYPE: {artifact_type}\nPAYLOAD:\n```\n{payload}\n```"  # noqa: E731
    return propose_artifact(failure, replay, [], model)


def test_producer_emits_the_keys_the_hooks_match_on():
    rule = _produce("rule", "READ THE FILE FIRST", "thrash:Bash", "thrash")
    assert rule.origin["tools"] == ["Bash"]
    # The matcher key is not the episode kind, and the episode kind is still carried.
    assert rule.origin["rule_kind"] == "repeat-call"
    assert rule.origin["kind"] == "thrash"

    injection = _produce("injection", "PASS AN EXPLICIT TIMEOUT", "tool_error:Bash:timeout",
                         "tool_error")
    assert "bash" in injection.origin["triggers"]
    assert "timeout" in injection.origin["triggers"]


def test_produced_artifacts_are_enforced_by_the_real_hooks(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run hooks/harness.loop.test.mts")

    store = HarnessStore(tmp_path)
    for artifact in (_produce("rule", "READ THE FILE FIRST", "thrash:Bash", "thrash"),
                     _produce("injection", "PASS AN EXPLICIT TIMEOUT",
                              "tool_error:Bash:timeout", "tool_error")):
        store.stage(artifact)
        store.accept(artifact.id)

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.loop.test.mts"),
         str(tmp_path).replace("\\", "/")],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.loop.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    assert "all assertions passed" in result.stdout
