import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from cause_fixtures import CAUSE_FIXTURES
from meta_harness.replay import _cause

ROOT = Path(__file__).resolve().parent.parent


def _run_ts_cause(texts: list[str]) -> list[str]:
    """Feed `texts` through hooks/harness.ts's cause() via node and return the resulting slugs,
    in the same order. Raises pytest.skip if node is unavailable.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.ts cause() parity check")

    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "input.json"
        output_path = Path(tmp) / "output.json"
        input_path.write_text(json.dumps(texts), encoding="utf-8")

        result = subprocess.run(
            [
                node, "--experimental-strip-types", "--no-warnings",
                "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
                str(ROOT / "hooks" / "harness.cause_parity.test.mts"),
                str(input_path), str(output_path),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=ROOT,
        )
        assert result.returncode == 0, (
            f"hooks/harness.cause_parity.test.mts failed:\nstdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        return json.loads(output_path.read_text(encoding="utf-8"))


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


def test_ts_and_python_cause_agree_on_fixtures():
    """The interface promises {ts, tool, kind, cause, input}; `cause` must be the same slug
    meta_harness.replay._cause() would compute from the same text, or dedupe between
    hook-observed and replay-mined episodes silently breaks.

    A prior version of this test compared the *source text* of CAUSE_PATTERNS against
    ERROR_PATTERNS. That can only prove the two lists look alike; it cannot prove they classify
    text the same way, because hooks/harness.ts's CAUSE_PATTERNS entries are plain JS string
    literals, and JS silently drops a backslash in front of any character it does not recognize
    as a string escape (`\\d`, `\\S`, `\\[`, `\\]`, `\\(`, `\\)` are not recognized escapes). A
    single-backslash TS mirror of a Python pattern containing any of those can type-check, pass
    a text-comparison test, and still classify differently at runtime. This test instead
    executes both classifiers -- Python's _cause() directly, TypeScript's cause() via node -- on
    the identical fixture texts and asserts they produce the identical slugs.
    """
    texts = [text for text, _expected in CAUSE_FIXTURES]
    expected = [expected for _text, expected in CAUSE_FIXTURES]
    py_slugs = [_cause(text) for text in texts]
    assert py_slugs == expected, "a CAUSE_FIXTURES entry no longer matches its expected slug on the Python side"

    ts_slugs = _run_ts_cause(texts)
    mismatches = [
        (text[:80], py, ts)
        for text, py, ts in zip(texts, py_slugs, ts_slugs)
        if py != ts
    ]
    assert not mismatches, f"Python and TypeScript disagree on {len(mismatches)} fixture(s): {mismatches}"


def test_ts_and_python_cause_agree_across_the_mined_corpus():
    """Runs the same parity check as test_ts_and_python_cause_agree_on_fixtures, but over every
    tool_error episode's assistant_text in the mined history, not just the hand-picked fixtures.
    Skips (rather than fails) when the corpus file has not been mined in this environment --
    that file is generated by `python -m meta_harness mine`, not checked in.
    """
    episodes_path = ROOT / ".meta-harness" / "history-text" / "episodes.jsonl"
    if not episodes_path.exists():
        pytest.skip(f"{episodes_path} does not exist; run the mine command first")

    texts = []
    with episodes_path.open(encoding="utf-8") as handle:
        for line in handle:
            episode = json.loads(line)
            if episode.get("kind") == "tool_error":
                texts.append(episode.get("assistant_text", ""))
    assert texts, "no tool_error episodes found in the mined corpus"

    py_slugs = [_cause(text) for text in texts]
    ts_slugs = _run_ts_cause(texts)
    mismatches = [
        (text[:80], py, ts)
        for text, py, ts in zip(texts, py_slugs, ts_slugs)
        if py != ts
    ]
    assert not mismatches, (
        f"Python and TypeScript disagree on {len(mismatches)} of {len(texts)} mined episodes; "
        f"first few: {mismatches[:5]}"
    )


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
