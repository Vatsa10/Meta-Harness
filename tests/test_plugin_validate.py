"""Regression guard: Claude Code's own validator must accept the plugin and its hooks module.

`claude plugin validate` reads the hooks module the way the engine loads it and reports every
refusal (non-literal hooks, a repeated event, `$` passed out of a hook, ...). The unit fakes accept
modules the engine refuses, so this is the check that the module actually loads.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_claude_plugin_validate_passes():
    claude = shutil.which("claude")
    if claude is None:
        pytest.skip("the `claude` CLI is not on PATH, so the engine's own plugin validator cannot run")
    result = subprocess.run(
        [claude, "plugin", "validate", str(ROOT / ".claude-plugin" / "plugin.json")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        cwd=ROOT,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, f"claude plugin validate failed:\n{output}"
    assert "Validation passed" in output, output
