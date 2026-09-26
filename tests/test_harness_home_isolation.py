import os
from pathlib import Path

from meta_harness.harness_store import harness_home

from conftest import REAL_HARNESS_HOME


def test_every_test_gets_a_throwaway_harness_home():
    assert os.environ.get("META_HARNESS_HOME")
    assert harness_home().resolve() != REAL_HARNESS_HOME.resolve()


def test_waste_command_writes_only_under_the_throwaway_home(tmp_path, capsys):
    from meta_harness.__main__ import main
    main(["waste", "--json", "--home", str(tmp_path / "nothing")])
    capsys.readouterr()
    assert (harness_home() / "waste.json").is_file()
    assert REAL_HARNESS_HOME.resolve() not in (harness_home() / "waste.json").resolve().parents
