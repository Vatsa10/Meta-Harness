import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "learning-from-failures" / "SKILL.md"


def test_skill_exists_with_frontmatter():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: learning-from-failures" in text
    assert "description: Use when" in text


def test_description_states_triggers_not_workflow():
    text = SKILL.read_text(encoding="utf-8")
    description = text.split("description:", 1)[1].split("\n---", 1)[0]
    # A description that summarises the workflow gets followed instead of the skill body.
    assert "mine" not in description.lower()
    assert "step" not in description.lower()


def test_skill_names_the_staging_gate():
    text = SKILL.read_text(encoding="utf-8")
    # The safety property itself: nothing installs on a score alone.
    assert "Nothing is adopted on a score alone." in text
    # How a human actually does it: --accept installs.
    assert "learn --accept <id>          # install it" in text
    assert "Never accept without reading the payload." in text


def test_plugin_version_bumped():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["version"] >= "0.6.0"
