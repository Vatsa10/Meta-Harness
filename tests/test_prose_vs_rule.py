from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from prose_vs_rule import build_arms, summarize


def test_two_arms_differ_only_in_where_the_doctrine_lives(tmp_path: Path):
    doctrine = tmp_path / "d.md"
    doctrine.write_text("Read before you edit.", encoding="utf-8")
    arms = build_arms(doctrine, tmp_path / "tasks.jsonl")
    assert set(arms) == {"prose", "rule"}
    assert "Read before you edit." in arms["prose"]["append_system_prompt"]
    # The rule arm carries no doctrine text: enforcement is the hook's job.
    assert arms["rule"]["append_system_prompt"] == ""


def test_both_arms_share_the_same_task_set(tmp_path: Path):
    doctrine = tmp_path / "d.md"
    doctrine.write_text("x", encoding="utf-8")
    arms = build_arms(doctrine, tmp_path / "tasks.jsonl")
    assert arms["prose"]["tasks"] == arms["rule"]["tasks"]


def test_summary_names_which_objective_moved():
    text = summarize({"prose": {"score": 1.0, "context": 130000.0},
                      "rule": {"score": 1.0, "context": 96000.0}})
    assert "context" in text.lower()
    assert "26" in text or "34000" in text


def test_summary_refuses_to_claim_an_accuracy_win_when_scores_tie():
    text = summarize({"prose": {"score": 1.0, "context": 100.0},
                      "rule": {"score": 1.0, "context": 100.0}})
    assert "no difference" in text.lower()


def test_summary_reports_no_detectable_difference_for_small_sample():
    # Scores differ (0.6 vs 1.0) but only 3 tasks per arm: too small to trust.
    text = summarize({"prose": {"score": 0.6, "context": 100.0, "n": 3},
                      "rule": {"score": 1.0, "context": 100.0, "n": 3}})
    assert "no detectable difference" in text.lower()
    assert "wins" not in text.lower()


def test_summary_does_not_declare_winner_from_one_lucky_run():
    # A single run per arm (n defaults to 1 when omitted) is not evidence.
    text = summarize({"prose": {"score": 0.0, "context": 100.0},
                      "rule": {"score": 1.0, "context": 100.0}})
    assert "no detectable difference" in text.lower()


def test_summary_declares_significant_winner_with_adequate_sample():
    text = summarize({"prose": {"score": 0.5, "context": 100.0, "n": 60},
                      "rule": {"score": 0.9, "context": 100.0, "n": 60}})
    assert "no detectable difference" not in text.lower()
    assert "rule" in text.lower()


def test_summary_reports_when_arms_are_statistically_tied_despite_score_gap():
    # Small nominal gap, moderate n: not enough to clear the significance bar.
    text = summarize({"prose": {"score": 0.50, "context": 100.0, "n": 20},
                      "rule": {"score": 0.55, "context": 100.0, "n": 20}})
    assert "no detectable difference" in text.lower()
