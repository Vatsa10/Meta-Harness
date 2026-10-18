import json

import pytest

from meta_harness import experience_query as q
from meta_harness.metrics import diversity


def test_identical_sources_have_zero_diversity():
    assert diversity(["a = 1\nb = 2", "a = 1\nb = 2", "a = 1\nb = 2"]) == 0.0


def test_distinct_sources_have_positive_diversity():
    assert diversity(["alpha beta", "gamma delta"]) == 1.0
    assert diversity(["a b c d", "a b c e"]) == pytest.approx(1 - 3 / 5)
    assert 0 < diversity(["x = 1", "x = 2", "x = 1"]) < 1


def test_fewer_than_two_sources():
    assert diversity([]) == 0.0 and diversity(["x"]) == 0.0


def _cand(root, cid, source):
    d = root / "candidates" / cid
    d.mkdir(parents=True)
    (d / "harness.py").write_text(source, encoding="utf-8")
    (d / "receipt.json").write_text(json.dumps({"verdict": "accepted"}), encoding="utf-8")


def test_report_has_overall_and_per_iteration(tmp_path):
    _cand(tmp_path, "iteration-0001-00", "alpha beta")
    _cand(tmp_path, "iteration-0001-01", "alpha beta")
    _cand(tmp_path, "iteration-0002-00", "gamma delta")
    _cand(tmp_path, "iteration-0002-01", "epsilon zeta")
    div = q.report(tmp_path)["diversity"]
    assert div["by_iteration"] == {"0001": 0.0, "0002": 1.0}
    assert div["overall"] == pytest.approx((0 + 1 + 1 + 1 + 1 + 1) / 6)
