import pytest

from meta_harness.drift import Verdict, judge_model
from meta_harness.waste import Stretch


def stretch():
    return Stretch("s", "p", 0, calls=["Read", "Edit"] * 6,
                   paths=["/secret/path/config.py"] * 12)


def fallback():
    return Verdict(drifting=False, score=0.0, reason="fallback")


def test_absent_capability_falls_back():
    assert judge_model(stretch(), 12, "add a flag", None, fallback).reason == "fallback"


def test_a_raising_model_falls_back():
    def boom(prompt): raise RuntimeError("no model here")
    assert judge_model(stretch(), 12, "add a flag", boom, fallback).reason == "fallback"


def test_an_unparseable_answer_falls_back():
    assert judge_model(stretch(), 12, "r", lambda p: "I think maybe?", fallback).reason == "fallback"


@pytest.mark.parametrize("answer,expected", [("DRIFT: gone to unrelated files", True), ("OK: on task", False)])
def test_a_parseable_answer_is_used(answer, expected):
    verdict = judge_model(stretch(), 12, "add a flag", lambda p: answer, fallback)
    assert verdict.drifting is expected and verdict.reason != "fallback"


def test_the_prompt_carries_no_file_contents_and_no_full_paths():
    seen = {}
    def capture(prompt):
        seen["prompt"] = prompt
        return "OK: fine"
    judge_model(stretch(), 12, "add a flag", capture, fallback)
    assert "/secret/path/" not in seen["prompt"]
    assert "config.py" in seen["prompt"]
    assert "add a flag" in seen["prompt"]
