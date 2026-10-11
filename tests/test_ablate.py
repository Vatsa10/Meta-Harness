import json
import sys
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness.ablate import run_ablation
from meta_harness.stats import bh, compare, mcnemar_exact

FIX = Path(__file__).parent / "fixtures" / "smoke"

# view -> per-run (best search score, best test score)
CANNED = {"scores": [(0.1, 0.1), (0.2, 0.2), (0.9, 0.9)],
          "summary": [(0.5, 0.4), (0.5, 0.4), (0.5, 0.4)],
          "full": [(0.3, 0.3), (0.6, 0.6), (0.6, 0.6)]}
SEED = 0.25


def write_trace(path: Path, rewards):
    lines = []
    for i, reward in enumerate(rewards):
        lines.append({"event": "task_start", "payload": {"index": i}})
        lines.append({"event": "model_call", "payload": {"tokens": 1000, "usage": {"completion_tokens": 500}}})
        lines.append({"event": "task_end", "payload": {"index": i, "reward": reward, "stop_reason": "done"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


def fake_search(calls):
    counters = {}

    def search(argv):
        calls.append(list(argv))
        view, root = argv[argv.index("--view") + 1], Path(argv[argv.index("--root") + 1])
        i = counters[view] = counters.get(view, -1) + 1
        search_best, test_best = CANNED[view][i]
        root.mkdir(parents=True, exist_ok=True)
        (root / "frontier.json").write_text(json.dumps([{"candidate_id": "c", "score": search_best}]), encoding="utf-8")
        seed = root / "candidates" / "initial-0000"
        seed.mkdir(parents=True)
        (seed / "scores.json").write_text(json.dumps({"score": SEED}), encoding="utf-8")
        test = root.with_name(root.name + "-test")
        test.mkdir(parents=True)
        metrics = {"ended_without_writing_code_rate": 0.5} if view == "full" else {}
        (test / "test_results.json").write_text(json.dumps({"results": [
            {"candidate_id": "c", "score": test_best, "metrics": metrics},
            {"candidate_id": "worse", "score": test_best / 2, "metrics": {}}]}), encoding="utf-8")
        # "full" passes every task, the others pass the first 4 of 10: a clear paired difference
        write_trace(test / "traces" / "c.jsonl", [1.0] * 10 if view == "full" else [1.0] * 4 + [0.0] * 6)
        return 0
    return search


def test_medians_bests_above_seed_and_distinct_roots(tmp_path):
    calls = []
    report = run_ablation(["--x", "1"], ["scores", "summary", "full"], 3, tmp_path / "ab", fake_search(calls))
    scores = report["views"]["scores"]
    assert scores["median_search"] == 0.2 and scores["best_search"] == 0.9   # a mean would give 0.4
    assert scores["median_test"] == 0.2 and scores["best_test"] == 0.9
    assert scores["above_seed"] == 1 and scores["runs"] == 3
    assert report["views"]["summary"]["above_seed"] == 3
    assert report["views"]["full"]["median_search"] == 0.6 and report["views"]["full"]["above_seed"] == 3
    roots = [c[c.index("--root") + 1] for c in calls]
    assert len(calls) == 9 and len(set(roots)) == 9
    assert all(c[:2] == ["--x", "1"] for c in calls)
    assert json.loads((tmp_path / "ab" / "ablation.json").read_text(encoding="utf-8")) == report


def test_paired_stats_cost_and_phase_rate_in_report(tmp_path):
    argv = ["--model", "m", "--price", "scripted:m=1000000,1000000"]
    report = run_ablation(argv, ["scores", "full"], 3, tmp_path / "ab", fake_search([]))
    full = report["views"]["full"]
    assert full["vs_baseline"]["wins"] == 18 and full["vs_baseline"]["losses"] == 0
    assert full["vs_baseline"]["ties"] == 12
    assert full["vs_baseline"]["p"] == pytest.approx(mcnemar_exact(0, 18))
    assert full["vs_baseline"]["significant"] is True
    assert "vs_baseline" not in report["views"]["scores"]
    assert full["ended_without_writing_code_rate"] == 0.5
    assert report["views"]["scores"]["ended_without_writing_code_rate"] is None
    assert full["mean_cost_usd_per_task"] == pytest.approx(1500.0)   # 1000 in + 500 out at $1 per token


def test_view_or_root_in_search_args_exits_2(tmp_path, capsys):
    for bad in (["--view", "full"], ["--root=x"]):
        code = cli.main(["ablate", "--root", str(tmp_path / "ab"), "--", "--harness", "h.py", *bad])
        assert code == 2
    assert "must not include" in capsys.readouterr().out
    with pytest.raises(ValueError):
        run_ablation(["--view", "x"], ["full"], 1, tmp_path, lambda argv: 0)


def test_real_offline_run_writes_both_views(tmp_path, capsys):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"input": t, "label": l}) for t, l in [
        ("an apple", "fruit"), ("a pear", "fruit"), ("a fast car", "vehicle"),
        ("a red truck", "vehicle"), ("a banana", "fruit"), ("a blue van", "vehicle")]), encoding="utf-8")
    agent = tmp_path / "agent.py"
    agent.write_text(
        "import os\nfrom pathlib import Path\nout = Path(os.environ['META_HARNESS_OUTPUT'])\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'candidate-00.py').write_text(\"class Harness:\\n    def run(self, task, model, trace):\\n"
        "        return model(task['input'])\\n\")\n(out / 'candidate-00.reasoning.md').write_text('echo')\n",
        encoding="utf-8")
    root = tmp_path / "ab"
    code = cli.main(["ablate", "--views", "scores,full", "--runs", "1", "--root", str(root), "--",
                     "--harness", str(FIX / "seed.py"), "--data", str(tasks), "--provider", "scripted",
                     "--model", str(FIX / "base.jsonl"), "--iterations", "1", "--no-cache",
                     "--test-fraction", "0.5", "--proposer-command", f"{sys.executable} {agent}"])
    assert code == 0
    report = json.loads((root / "ablation.json").read_text(encoding="utf-8"))
    assert set(report["views"]) == {"scores", "full"}
    assert all(v["runs"] == 1 and len(v["search_best"]) == 1 and len(v["test_best"]) == 1
               for v in report["views"].values())
    assert (root / "scores" / "run-0" / "frontier.json").is_file()
    assert (root / "full" / "run-0" / "frontier.json").is_file()
    assert "scores" in capsys.readouterr().out


def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 10) == pytest.approx(2 / 1024)   # 0.001953125
    assert mcnemar_exact(10, 0) == mcnemar_exact(0, 10)
    assert mcnemar_exact(5, 5) == 1.0 and mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(2, 8) == pytest.approx(2 * 56 / 1024)


def test_bh_fixture_marks_only_the_first():
    q, flags = bh([0.01, 0.04, 0.03, 0.2], q=0.05)
    assert flags == [True, False, False, False]
    assert q == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.2])
    _, loose = bh([0.01, 0.02, 0.03, 0.2], q=0.05)   # 0.03 <= 3 * .05 / 4 passes, so ranks 1-3 do
    assert loose == [True, True, True, False]


def test_compare_counts_pairs_by_key():
    base = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 1.0}
    arm = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0, "extra": 1.0}
    got = compare(base, {"x": arm})["x"]
    assert (got["wins"], got["losses"], got["ties"]) == (2, 1, 1)
