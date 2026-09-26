import pytest

from tools.tune_drift import MIN_CALLS_SAVED, PRECISION_BAR, gate


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
