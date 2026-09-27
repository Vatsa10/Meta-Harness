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


def test_nlrules_reads_prompt_submit_not_prompt_section():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # prompt.section has no event.text/event.prompt at all; only prompt.submit's event carries
    # the human's actual words.
    assert "guardBefore('prompt.submit:nlrules'" in source


def test_stop_instruction_requires_an_actionable_verb():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    # The action verb (running/using/calling/doing/touching/editing/deleting, or its plain
    # imperative) is mandatory, not optional: an earlier version made it optional and matched
    # "don't worry about it", "never mind" and other acknowledgements with no actionable target.
    assert "STOP_VERB" in source
    assert "touching|editing|deleting" in source


def test_stop_instruction_accepts_the_plain_imperative_too():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    # Fix round 2 finding 4: only the -ing form was accepted, so "don't run pytest" and "don't
    # use git push --force" (no gerund) returned null and were silently missed.
    assert "run|use|call|do|touch|edit|delete|push" in source


def test_stop_instruction_rejects_function_words_as_the_object():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    # Fix round 2 finding 4: "Stop doing that" and "stop using the" must not become rules that
    # deny every call containing the English word "that" or "the".
    assert "FUNCTION_WORDS" in source
    assert "isPlausibleObject" in source


def test_stop_instruction_supports_a_requires_flag():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    # Fix round 2 finding 6 (reversing an earlier ruling): "without <flag>" is now captured as a
    # `requires` flag the rule allows, instead of denying the pattern in every form — denying
    # `pytest -q` outright (the command the human asked to KEEP) was the wrong call.
    assert "requires" in source
    harness_source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "rule.requires" in harness_source


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
