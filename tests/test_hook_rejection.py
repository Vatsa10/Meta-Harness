import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_rejection_memory_is_exported():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    assert "export function rememberRejection" in source
    assert "export function wasRejected" in source


def test_tool_check_checks_rejection_before_the_rule_loop():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "wasRejected(state" in source


def test_mention_clearing_reads_prompt_submit_not_prompt_section():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # prompt.section has no event.text/event.prompt at all; only prompt.submit's event carries
    # the human's actual words, which is what mention-clearing must read.
    assert "guardBefore('prompt.submit:nlrules'" in source
    assert "event?.text" in source


def test_rejection_text_is_read_from_every_outcome_field():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # A real ToolCallResult carries the rejection on `deny`, or on `text`/`result` when
    # `isError` is true; checking `result` alone missed the other two.
    assert "function rejectionAnnouncement" in source
    assert "outcome?.deny" in source
    assert "outcome?.text" in source


def test_rejection_never_fires_on_a_successful_result():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # A successful Read whose file merely CONTAINS the rejection phrase (this file, for one)
    # must never be recorded as a rejection: only `deny`, or `text`/`result` when `isError` is
    # true, may be checked — never a successful result's own content.
    assert "outcome?.isError !== true" in source


def test_rejection_behaviour_via_node():
    """A grep over the source cannot prove a rejected call is ever remembered or denied: it
    would pass just as well against a rememberRejection that is a no-op, or a tool.check that
    ignores wasRejected's verdict. hooks/harness.rejection.test.mts drives the real
    registerRules() handlers with a fake `$` and asserts on the actual decisions returned.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; cannot run the hooks/harness.rejection.test.mts behavioural test")

    result = subprocess.run(
        [node, "--experimental-strip-types", "--no-warnings",
         "--import", (ROOT / "hooks" / "loaders" / "preload.mjs").as_uri(),
         str(ROOT / "hooks" / "harness.rejection.test.mts")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"hooks/harness.rejection.test.mts failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "all assertions passed" in result.stdout
