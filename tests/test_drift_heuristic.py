from meta_harness.drift import Verdict, anchor_paths, judge_overlap
from meta_harness.waste import Stretch


def stretch(calls, paths):
    return Stretch("s", "p", 0, calls=list(calls), paths=list(paths))


def test_a_short_stretch_never_drifts_however_different():
    s = stretch(["Read"] * 4, ["a.py", "b.py", "z.py", "q.py"])
    assert judge_overlap(s, at_call=4, min_calls=8).drifting is False


def test_work_that_stays_on_the_anchor_files_is_not_drifting():
    s = stretch(["Read"] * 12, ["a.py"] * 12)
    assert judge_overlap(s, at_call=12, min_calls=8).drifting is False


def test_work_that_has_left_the_anchor_files_entirely_is_drifting():
    s = stretch(["Read"] * 12, ["a.py", "a.py", "a.py"] + ["far/away.py"] * 9)
    verdict = judge_overlap(s, at_call=12, min_calls=8)
    assert verdict.drifting is True
    assert "a.py" in verdict.reason or "overlap" in verdict.reason


def test_a_stretch_with_no_paths_at_all_is_not_called_drifting():
    # Shell-only work has no file anchor; claiming drift here would be guessing.
    s = stretch(["Bash"] * 12, [])
    assert judge_overlap(s, at_call=12, min_calls=8).drifting is False


def test_anchor_is_taken_from_the_first_calls_only():
    s = stretch(["Read"] * 10, ["first.py", "first.py", "first.py"] + ["later.py"] * 7)
    assert anchor_paths(s, anchor_calls=3) == {"first.py"}
