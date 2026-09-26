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


def _merged_stretch():
    # A stretch-splitter fix merged what used to be two stretches into one. It runs turns 0-9
    # (10 calls, one per turn), so its end_turn is 9.
    return Stretch("s1", "p", 0, calls=["Read"] * 10, end_turn=9)


def test_a_gold_key_joins_to_the_stretch_that_contains_it_even_when_moved():
    # The gold record's own start_turn (5) no longer matches any stretch exactly, but turn 5
    # still falls inside this merged stretch's span (0-9) - an exact-match join would wrongly
    # call this a failed join (or, worse, silently drop a genuine correction).
    merged = _merged_stretch()
    gold = [{"session_id": "s1", "start_turn": 5, "calls": 3, "label": "a", "why": "x"}]
    result = _join_gold_by_containment(gold, [merged])
    assert result["matched"] == 1
    assert result["moved"] == 1
    assert id(merged) in result["positive_ids"]


def test_a_merged_stretch_containing_both_labels_is_positive():
    merged = _merged_stretch()
    gold = [
        {"session_id": "s1", "start_turn": 2, "calls": 3, "label": "c", "why": "x"},
        {"session_id": "s1", "start_turn": 6, "calls": 3, "label": "a", "why": "y"},
    ]
    result = _join_gold_by_containment(gold, [merged])
    assert id(merged) in result["positive_ids"]


def test_a_gold_turn_in_the_gap_between_two_stretches_fails_to_join():
    # The reviewer's exact case: s1 covers turns 0-4, s2 covers turns 20-24, and a gold record
    # at turn 10 sits in the gap between them (a short, filtered-out stretch, or no tool calls
    # at all). "Largest start_turn <= gold_turn" alone would wrongly attach it to s1.
    s1 = Stretch("s1", "p", 0, calls=["Read"] * 5, end_turn=4)
    s2 = Stretch("s1", "p", 20, calls=["Read"] * 5, end_turn=24)
    gold = [{"session_id": "s1", "start_turn": 10, "calls": 3, "label": "a", "why": "gap"}]
    result = _join_gold_by_containment(gold, [s1, s2])
    assert result["matched"] == 0
    assert len(result["failed"]) == 1
    assert id(s1) not in result["positive_ids"]
    assert id(s2) not in result["positive_ids"]
    assert "reason" in result["failed"][0]


def test_a_gold_turn_at_a_stretchs_last_turn_joins():
    s1 = Stretch("s1", "p", 0, calls=["Read"] * 5, end_turn=4)
    gold = [{"session_id": "s1", "start_turn": 4, "calls": 3, "label": "a", "why": "x"}]
    result = _join_gold_by_containment(gold, [s1])
    assert result["matched"] == 1
    assert id(s1) in result["positive_ids"]


def test_a_gold_turn_just_past_a_stretchs_last_turn_does_not_join():
    s1 = Stretch("s1", "p", 0, calls=["Read"] * 5, end_turn=4)
    gold = [{"session_id": "s1", "start_turn": 5, "calls": 3, "label": "a", "why": "x"}]
    result = _join_gold_by_containment(gold, [s1])
    assert result["matched"] == 0
    assert id(s1) not in result["positive_ids"]


def test_a_gold_record_with_no_containing_stretch_fails_to_join():
    later = Stretch("s1", "p", 10, calls=["Read"] * 10, end_turn=19)
    gold = [{"session_id": "s1", "start_turn": 2, "calls": 3, "label": "a", "why": "z"}]
    result = _join_gold_by_containment(gold, [later])
    assert result["matched"] == 0
    assert len(result["failed"]) == 1
    assert result["failed"][0]["session_id"] == "s1"
    assert result["failed"][0]["start_turn"] == 2
    assert "reason" in result["failed"][0]


def test_a_missing_gold_file_raises_instead_of_scoring_zero_positives(tmp_path):
    with pytest.raises(GoldFileMissing):
        _load_gold(tmp_path / "does-not-exist.json")


def test_main_exits_non_zero_and_reports_a_missing_gold_file(tmp_path, capsys):
    rc = main(["--gold", str(tmp_path / "does-not-exist.json"), "--home", str(tmp_path / "nothing")])
    assert rc != 0
    assert "gold file not found" in capsys.readouterr().err
