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


def write_trace(path: Path, rewards, ids="t"):
    lines = []
    for i, reward in enumerate(rewards):
        lines.append({"event": "task_start", "payload": {"index": i, "task": {"id": f"{ids}{i}", "input": "x"}}})
        lines.append({"event": "model_call", "payload": {"tokens": 1000, "usage": {"completion_tokens": 500}}})
        lines.append({"event": "task_end", "payload": {"index": i, "reward": reward, "stop_reason": "done"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")


def fake_search(calls, ids="t", fail=()):
    counters = {}

    def search(argv):
        calls.append(list(argv))
        if len(calls) - 1 in fail:
            return 3
        view, root = argv[argv.index("--view") + 1], Path(argv[argv.index("--root") + 1])
        i = counters[view] = counters.get(view, -1) + 1
        search_best, test_best = CANNED[view][i]
        root.mkdir(parents=True, exist_ok=True)
        (root / "frontier.json").write_text(json.dumps([{"candidate_id": "c", "score": search_best, "context_cost": 5},
                                                {"candidate_id": "decoy", "score": search_best - 0.05, "context_cost": 1}]), encoding="utf-8")
        seed = root / "candidates" / "initial-0000"
        seed.mkdir(parents=True)
        (seed / "scores.json").write_text(json.dumps({"score": SEED}), encoding="utf-8")
        test = root.with_name(root.name + "-test")
        test.mkdir(parents=True)
        metrics = {"ended_without_writing_code_rate": 0.5} if view == "full" else {}
        (test / "test_results.json").write_text(json.dumps({"results": [
            {"candidate_id": "c", "score": test_best, "metrics": metrics},
            {"candidate_id": "decoy", "score": test_best + 0.05, "metrics": {}}]}), encoding="utf-8")
        # "full" passes every task, the others pass the first 4 of 10: a clear paired difference
        write_trace(test / "traces" / "c.jsonl", ([1.0] * 10 if view == "full" else [1.0] * 4 + [0.0] * 6), ids)
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
    assert full["vs_baseline"]["wins"] == 6 and full["vs_baseline"]["losses"] == 0
    assert full["vs_baseline"]["ties"] == 4
    assert full["vs_baseline"]["p"] == pytest.approx(mcnemar_exact(0, 6))
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


def test_test_score_is_the_search_selected_candidates_not_the_max_over_test(tmp_path):
    report = run_ablation([], ["scores"], 3, tmp_path / "ab", fake_search([]))
    # the decoy has the higher test score in every run but the lower search score
    assert report["views"]["scores"]["test_best"] == [0.1, 0.2, 0.9]
    assert report["views"]["scores"]["best_test"] == 0.9


def test_one_pair_per_task_and_mismatched_split_is_refused(tmp_path):
    report = run_ablation([], ["scores", "full"], 3, tmp_path / "a", fake_search([]))
    c = report["views"]["full"]["vs_baseline"]
    assert c["wins"] + c["losses"] + c["ties"] == 10   # 10 tasks x 3 runs collapse to 10 pairs
    inner = fake_search([])

    def shifted(argv):
        code = inner(argv)
        if "full" in argv:
            root = Path(argv[argv.index("--root") + 1])
            write_trace(root.with_name(root.name + "-test") / "traces" / "c.jsonl", [1.0] * 10, "other")
        return code
    with pytest.raises(ValueError, match="different tasks"):
        run_ablation([], ["scores", "full"], 1, tmp_path / "b", shifted)


def test_failed_runs_fail_the_command_and_are_not_counted(tmp_path, capsys):
    report = run_ablation([], ["scores"], 3, tmp_path / "ab", fake_search([], fail={1}))
    assert report["views"]["scores"]["runs"] == 2
    assert report["failed"] == [{"view": "scores", "run": 1, "exit_code": 3}]
    import argparse
    from meta_harness.commands import ablate as cmd
    parser = argparse.ArgumentParser()
    cmd.add_arguments(parser)
    args = parser.parse_args(["--views", "scores", "--runs", "2", "--root", str(tmp_path / "cli")])
    assert cmd.run(args, search=fake_search([], fail={0})) == 1
    assert "failed: view scores run 0" in capsys.readouterr().out


def test_non_empty_root_refused_and_data_sampled_deterministically(tmp_path):
    (tmp_path / "ab").mkdir()
    (tmp_path / "ab" / "old").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        run_ablation([], ["scores"], 1, tmp_path / "ab", fake_search([]))
    from meta_harness.ablate import cap_tasks
    data = tmp_path / "d.jsonl"
    data.write_text("\n".join(json.dumps({"input": str(i)}) for i in range(50)), encoding="utf-8")
    a = cap_tasks(["--data", str(data)], 20, tmp_path / "r1")
    b = cap_tasks(["--data", str(data)], 20, tmp_path / "r2")
    ra, rb = Path(a[1]).read_text(encoding="utf-8"), Path(b[1]).read_text(encoding="utf-8")
    assert ra == rb and len(ra.splitlines()) == 20


def test_bh_rejects_invalid_p_values():
    for bad in ([float("nan")], [1.5], [-0.1]):
        with pytest.raises(ValueError):
            bh(bad)


def test_majority_collapse_across_disagreeing_runs():
    from meta_harness.ablate import _collapse, _compare
    runs = [{"A": 1.0, "B": 0.0, "C": 1.0}, {"A": 1.0, "B": 1.0, "C": 0.0}, {"A": 0.0, "B": 0.0}]
    got = _collapse(runs)
    assert got == {"A": True, "B": False, "C": None}   # v[0] would give A, C pass and B fail
    base = {"A": False, "B": True, "C": False}
    c = _compare(base, {"x": got})["x"]
    assert (c["wins"], c["losses"], c["ties"]) == (1, 1, 1)
