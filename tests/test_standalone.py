import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GONE = ["cc_harness", "cc_history", "claude_cli", "learn", "waste", "drift", "receipts",
        "replay", "temporal", "harness_store"]


@pytest.mark.parametrize("name", GONE)
def test_claude_code_modules_are_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"meta_harness.{name}")


@pytest.mark.parametrize("path", ["hooks", "commands", ".claude-plugin", "agents", "skills",
                                  "tsconfig.json", "tools/tb_run.py", "baselines/cc_default.py"])
def test_plugin_files_are_gone(path):
    assert not (ROOT / path).exists()


def test_help_lists_engine_commands_only():
    completed = subprocess.run([sys.executable, "-m", "meta_harness", "--help"], cwd=ROOT,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert completed.returncode == 0, completed.stderr
    for gone in ("learn", "waste", "receipts", "mine", "compare"):
        assert f" {gone} " not in completed.stdout and f"{{{gone}" not in completed.stdout
    assert "run" in completed.stdout and "inspect" in completed.stdout


def test_package_imports_without_claude_binary(monkeypatch):
    monkeypatch.setenv("PATH", "")
    import meta_harness
    importlib.reload(meta_harness)
    assert not hasattr(meta_harness, "ClaudeCliModel")
