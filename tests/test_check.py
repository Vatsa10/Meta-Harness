import json

from meta_harness.__main__ import main

N = 12
TASKS = [{"id": f"t{i}", "input": f"q{i}", "label": "yes"} for i in range(N)]

GOOD = ("class H:\n    def run(self, task, model):\n        return {'output': model(task['input'])}\n\n"
        "def build_harness():\n    return H()\n")
# answers wrong ('no') on 10 of 12 tasks
BAD = ("class H:\n    def run(self, task, model):\n        out = model(task['input'])\n"
       "        return {'output': out if int(task['input'][1:]) < 2 else 'no'}\n\n"
       "def build_harness():\n    return H()\n")
# same answers, but ~40x the prompt tokens
PRICEY = ("class H:\n    def run(self, task, model):\n        return {'output': model(task['input'] + ' pad' * 400)}\n\n"
          "def build_harness():\n    return H()\n")
BROKEN = "def build_harness(:\n"


def setup(tmp_path, baseline=GOOD, candidate=GOOD):
    (tmp_path / "base.py").write_text(baseline, encoding="utf-8")
    (tmp_path / "cand.py").write_text(candidate, encoding="utf-8")
    (tmp_path / "data.jsonl").write_text("\n".join(json.dumps(t) for t in TASKS), encoding="utf-8")
    (tmp_path / "yes.jsonl").write_text(json.dumps({"default": "yes"}) + "\n", encoding="utf-8")


def check(tmp_path, *extra, data="data.jsonl"):
    argv = ["check", "--baseline", str(tmp_path / "base.py"), "--candidate", str(tmp_path / "cand.py"),
            "--data", str(tmp_path / data), "--metric", "exact", "--provider", "scripted",
            "--model", str(tmp_path / "yes.jsonl"), "--isolation", "none",
            "--out", str(tmp_path / "check.json"), "--markdown", str(tmp_path / "summary.md"), *extra]
    return main(argv)


def report(tmp_path):
    return json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))


def test_identical_harnesses_pass(tmp_path, capsys):
    setup(tmp_path)
    assert check(tmp_path, "--price", "1,1") == 0
    r = report(tmp_path)
    assert r["decision"]["passed"] and r["decision"]["score_delta"] == 0
    assert r["paired"]["wins"] == r["paired"]["losses"] == 0
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "PASS" in md and "| mean score |" in md and "| context tokens |" in md


def test_significant_regression_fails(tmp_path):
    setup(tmp_path, candidate=BAD)
    assert check(tmp_path) == 1
    r = report(tmp_path)
    assert r["paired"]["losses"] == 10 and r["paired"]["significant"]
    assert any("significant" in x for x in r["decision"]["reasons"])
    assert "FAIL" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_small_drop_not_significant_but_over_max_drop_fails_and_loose_threshold_passes(tmp_path):
    bad_one = BAD.replace("< 2", "< 11")  # wrong on 1 of 12: not significant, drop 0.083
    setup(tmp_path, candidate=bad_one)
    assert check(tmp_path, "--max-drop", "0.02") == 1
    r = report(tmp_path)
    assert not r["paired"]["significant"]
    assert all("significant" not in x for x in r["decision"]["reasons"])
    assert check(tmp_path, "--max-drop", "0.2") == 0


def test_cost_regression_fails_even_with_equal_quality(tmp_path):
    setup(tmp_path, candidate=PRICEY)
    assert check(tmp_path, "--price", "1,1") == 1
    r = report(tmp_path)
    assert r["decision"]["score_delta"] == 0
    assert r["decision"]["cost_increase"] > 1
    assert any("cost" in x for x in r["decision"]["reasons"])
    assert check(tmp_path, "--price", "1,1", "--max-cost-increase", "1000") == 0
    # context-token delta is reported even without a price
    assert report(tmp_path)["decision"]["context_increase"] > 1


def test_cost_gate_skipped_without_price(tmp_path, monkeypatch):
    monkeypatch.delenv("META_HARNESS_PRICES", raising=False)
    setup(tmp_path, candidate=PRICEY)
    assert check(tmp_path) == 0
    assert report(tmp_path)["decision"]["cost_increase"] is None


def test_setup_errors_exit_2(tmp_path, capsys):
    setup(tmp_path, candidate=BROKEN)
    assert check(tmp_path) == 2  # invalid harness
    assert "error:" in capsys.readouterr().err
    setup(tmp_path)
    assert check(tmp_path, data="missing.jsonl") == 2
    assert main(["check", "--baseline", str(tmp_path / "nope.py"), "--candidate", str(tmp_path / "cand.py"),
                 "--data", str(tmp_path / "data.jsonl"), "--metric", "exact", "--provider", "scripted",
                 "--model", str(tmp_path / "yes.jsonl")]) == 2
    assert check(tmp_path, "--price", "abc") == 2
