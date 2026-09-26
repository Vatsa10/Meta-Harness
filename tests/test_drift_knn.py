import collections

from meta_harness.drift import build_index, judge_knn, shape
from meta_harness.waste import Stretch


def stretch(calls):
    return Stretch("s", "p", 0, calls=list(calls))


def test_shape_counts_tool_name_ngrams():
    counted = shape(stretch(["Read", "Edit", "Read"]), n=2)
    assert counted["Read>Edit"] == 1 and counted["Edit>Read"] == 1


def test_an_empty_index_never_reports_drift():
    # A fresh install has no history. Firing here would be guessing.
    assert judge_knn(stretch(["Read"] * 12), at_call=12, index=[]).drifting is False


def test_a_shape_like_past_corrections_is_drifting():
    bad = shape(stretch(["Edit", "Bash", "Edit", "Bash"] * 3))
    index = [(bad, True)] * 8 + [(shape(stretch(["Read"] * 12)), False)]
    verdict = judge_knn(stretch(["Edit", "Bash"] * 6), at_call=12, index=index, k=9)
    assert verdict.drifting is True
    assert "correction" in verdict.reason


def test_a_shape_like_past_clean_stretches_is_not_drifting():
    good = shape(stretch(["Read"] * 12))
    index = [(good, False)] * 8 + [(shape(stretch(["Edit", "Bash"] * 6)), True)]
    assert judge_knn(stretch(["Read"] * 12), at_call=12, index=index, k=9).drifting is False


def test_a_short_stretch_never_drifts():
    index = [(shape(stretch(["Edit", "Bash"] * 6)), True)] * 9
    assert judge_knn(stretch(["Edit", "Bash"]), at_call=2, index=index, min_calls=8).drifting is False


def test_build_index_over_an_absent_home_is_empty_not_an_error(tmp_path):
    assert build_index(home=tmp_path / "nothing") == []
