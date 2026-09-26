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
        assert f"/harness {mode}" in text


def test_every_command_it_names_exists_in_the_cli():
    from meta_harness.__main__ import build_parser
    text = COMMAND.read_text(encoding="utf-8")
    known = set(build_parser()._subparsers._group_actions[0].choices)
    import re
    for invoked in re.findall(r"python -m meta_harness (\w+)", text):
        assert invoked in known, f"{invoked} is documented but not a real subcommand"
