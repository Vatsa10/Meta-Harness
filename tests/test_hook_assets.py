import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


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
    assert manifest.get("hooks", "./hooks/hooks.json").endswith("hooks.json")
