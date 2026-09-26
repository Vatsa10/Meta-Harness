import pytest

from meta_harness.waste import Stretch
from tools.tune_drift import (
    MIN_CALLS_SAVED,
    PRECISION_BAR,
    GoldFileMissing,
    _join_gold_by_containment,
    _load_gold,
    gate,
    main,
)


def test_a_judge_below_the_precision_bar_does_not_ship():
    assert gate(precision=PRECISION_BAR - 0.01, median_calls_saved=20) is False


def test_a_judge_that_fires_too_late_does_not_ship():
    assert gate(precision=0.95, median_calls_saved=MIN_CALLS_SAVED - 1) is False


def test_a_judge_clearing_both_ships():
    assert gate(precision=PRECISION_BAR, median_calls_saved=MIN_CALLS_SAVED) is True


def test_the_bar_is_a_constant_not_a_parameter_callers_can_soften():
    # The gate exists to be able to refuse. It takes no threshold argument.
    import inspect
    assert list(inspect.signature(gate).parameters) == ["precision", "median_calls_saved"]


def test_a_gold_key_joins_to_the_stretch_that_contains_it_even_when_moved():
    # A stretch-splitter fix merged what used to be two stretches into one starting at turn 0.
    # The gold record's own start_turn (5) no longer matches any stretch exactly, but turn 5
    # still falls inside this merged stretch - an exact-match join would wrongly call this a
    # failed join (or, worse, silently drop a genuine correction).
    merged = Stretch("s1", "p", 0, calls=["Read"] * 10)
    gold = [{"session_id": "s1", "start_turn": 5, "calls": 3, "label": "a", "why": "x"}]
    result = _join_gold_by_containment(gold, [merged])
    assert result["matched"] == 1
    assert result["moved"] == 1
    assert id(merged) in result["positive_ids"]


def test_a_merged_stretch_containing_both_labels_is_positive():
    merged = Stretch("s1", "p", 0, calls=["Read"] * 10)
    gold = [
        {"session_id": "s1", "start_turn": 2, "calls": 3, "label": "c", "why": "x"},
        {"session_id": "s1", "start_turn": 6, "calls": 3, "label": "a", "why": "y"},
    ]
    result = _join_gold_by_containment(gold, [merged])
    assert id(merged) in result["positive_ids"]


def test_a_gold_record_with_no_containing_stretch_fails_to_join():
    later = Stretch("s1", "p", 10, calls=["Read"] * 10)
    gold = [{"session_id": "s1", "start_turn": 2, "calls": 3, "label": "a", "why": "z"}]
    result = _join_gold_by_containment(gold, [later])
    assert result["matched"] == 0
    assert result["failed"] == gold


def test_a_missing_gold_file_raises_instead_of_scoring_zero_positives(tmp_path):
    with pytest.raises(GoldFileMissing):
        _load_gold(tmp_path / "does-not-exist.json")


def test_main_exits_non_zero_and_reports_a_missing_gold_file(tmp_path, capsys):
    rc = main(["--gold", str(tmp_path / "does-not-exist.json"), "--home", str(tmp_path / "nothing")])
    assert rc != 0
    assert "gold file not found" in capsys.readouterr().err
