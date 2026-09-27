"""Every test runs against a throwaway harness home.

`meta_harness.harness_store.harness_home()` falls back to ~/.claude/harness when
META_HARNESS_HOME is unset, and `meta-harness waste` writes waste.json there. A test that
forgot to set the variable once overwrote a user's real waste.json with an empty report, so
the variable is set for EVERY test here, and the session fails if the real harness home was
touched anyway (a file created, removed or modified while the suite ran)."""
import os
from pathlib import Path

import pytest

REAL_HARNESS_HOME = Path.home() / ".claude" / "harness"


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    if not root.is_dir():
        return {}
    state = {}
    for path in root.rglob("*"):
        try:
            if path.is_file():
                stat = path.stat()
                state[str(path)] = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            continue
    return state


@pytest.fixture(autouse=True)
def isolated_harness_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("harness_home")
    monkeypatch.setenv("META_HARNESS_HOME", str(home))
    return home


@pytest.fixture(autouse=True, scope="session")
def real_harness_home_untouched():
    before = _snapshot(REAL_HARNESS_HOME)
    yield
    after = _snapshot(REAL_HARNESS_HOME)
    changed = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    assert not changed, f"the test suite wrote under the real harness home: {changed}"
