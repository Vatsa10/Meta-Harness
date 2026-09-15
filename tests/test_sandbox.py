from pathlib import Path

import pytest

from meta_harness.core import HarnessValidationError
from meta_harness.sandbox import validate_in_subprocess


def _write(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_valid_harness_passes(tmp_path: Path):
    source = _write(tmp_path, "good.py", '''
class Harness:
    def run(self, task, model, trace):
        return model(task["input"])
''')
    validate_in_subprocess(source, timeout=60.0)


def test_build_harness_factory_passes(tmp_path: Path):
    source = _write(tmp_path, "factory.py", '''
class _H:
    def run(self, task, model, trace):
        return "x"


def build_harness():
    return _H()
''')
    validate_in_subprocess(source, timeout=60.0)


def test_missing_interface_fails(tmp_path: Path):
    source = _write(tmp_path, "bad.py", "VALUE = 1\n")
    with pytest.raises(HarnessValidationError, match="build_harness"):
        validate_in_subprocess(source, timeout=60.0)


def test_import_error_fails(tmp_path: Path):
    source = _write(tmp_path, "boom.py", "raise RuntimeError('nope')\n")
    with pytest.raises(HarnessValidationError, match="nope"):
        validate_in_subprocess(source, timeout=60.0)


def test_infinite_loop_is_killed(tmp_path: Path):
    source = _write(tmp_path, "hang.py", '''
class Harness:
    def run(self, task, model, trace):
        while True:
            pass
''')
    with pytest.raises(HarnessValidationError, match="timed out"):
        validate_in_subprocess(source, timeout=4.0)
