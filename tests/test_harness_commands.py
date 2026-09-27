from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMAND = ROOT / "commands" / "harness.md"


def test_the_command_file_exists_with_frontmatter():
    text = COMMAND.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "description:" in text


def test_all_three_modes_are_documented():
    text = COMMAND.read_text(encoding="utf-8")
    for mode in ("waste", "pending", "why"):
        assert f"/meta-harness:harness {mode}" in text


def test_every_command_it_names_exists_in_the_cli():
    from meta_harness.__main__ import build_parser
    text = COMMAND.read_text(encoding="utf-8")
    known = set(build_parser()._subparsers._group_actions[0].choices)
    import re
    for invoked in re.findall(r"python -m meta_harness (\w+)", text):
        assert invoked in known, f"{invoked} is documented but not a real subcommand"


# --- final fix wave --------------------------------------------------------------------------

def test_the_command_defers_to_the_report_caveat_rather_than_restating_a_figure():
    text = (ROOT / "commands" / "harness.md").read_text(encoding="utf-8")
    assert "52.4" not in text and "11 of 21" not in text
    assert "`caveat` field verbatim" in text


def test_the_command_does_not_claim_a_session_rule_waits_for_acceptance():
    text = (ROOT / "commands" / "harness.md").read_text(encoding="utf-8")
    assert "until then it has no effect" not in text
    assert "IMMEDIATELY" in text
    assert "not\n   implemented" in text or "not implemented" in text
