import json
import random
from collections import Counter
from pathlib import Path

from meta_harness.core import EvaluationResult, FilesystemExperience
from meta_harness.domains import generic
from meta_harness.optimizers import OptimizerContext, all_optimizers, build
from meta_harness.optimizers import gepa

# 3 candidates x 4 tasks. A wins tasks 0-2, B wins task 3, C is best nowhere.
MATRIX = {"A": [1.0, 1.0, 1.0, 0.0], "B": [0.0, 0.0, 0.0, 1.0], "C": [0.5, 0.5, 0.5, 0.5]}
HARNESS = "class Harness:\n    def run(self, task, model, trace):\n        return 'x'\n"
REPLY = "```python\n" + HARNESS + "```"


def write_trace(path: Path, rewards):
    rows = []
    for i, reward in enumerate(rewards):
        rows.append({"event": "task_start", "payload": {"index": i, "task": {"input": f"in-{i}", "label": "l"}}})
        rows.append({"event": "task_end", "payload": {"index": i, "output": f"out-{i}", "reward": reward}})
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def make_experience(tmp_path, matrix):
    experience = FilesystemExperience(tmp_path / "store")
    for cid, rewards in matrix.items():
        directory = experience.candidate_dir(cid)
        directory.mkdir(parents=True)
        (directory / "harness.py").write_text(HARNESS + f"# {cid}\n", encoding="utf-8")
        write_trace(directory / "traces.jsonl", rewards)
        experience.write_result(EvaluationResult(cid, sum(rewards) / len(rewards)))
    return experience


def as_matrix(raw):
    return {cid: dict(enumerate(r)) for cid, r in raw.items()}


def test_registered():
    assert "gepa" in all_optimizers()


def test_pareto_set_is_best_on_at_least_one_task():
    assert gepa.pareto_set(as_matrix(MATRIX)) == ["A", "B"]
    wins = gepa.task_wins(as_matrix(MATRIX))
    assert wins == {"A": {0, 1, 2}, "B": {3}, "C": set()}


def test_parent_sampling_is_proportional_to_wins():
    wins = gepa.task_wins(as_matrix(MATRIX))
    rng = random.Random(7)
    drawn = Counter(gepa.sample_parent(rng, wins) for _ in range(300))
    assert drawn["C"] == 0
    assert 0.65 <= drawn["A"] / 300 <= 0.85      # expected 3/4
    assert drawn["B"] >= 40                      # the overall-best candidate is not always the parent


def test_minibatch_is_the_three_lowest_reward_tasks():
    assert gepa.lowest_tasks({0: 0.9, 1: 0.1, 2: 0.5, 3: 0.0, 4: 0.7}) == [3, 1, 2]


def test_proposal_four_is_a_merge_of_two_disjoint_parents(tmp_path):
    experience = make_experience(tmp_path, MATRIX)
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return REPLY

    proposer = gepa.GepaProposer(model, domain="generic", seed=3)
    paths = proposer(experience, 1, 4)
    parents = [json.loads(p.with_name(p.stem + ".parent.json").read_text(encoding="utf-8")) for p in paths]
    assert [p["operator"] for p in parents] == ["reflect", "reflect", "reflect", "merge"]
    assert sorted(parents[3]["parents"]) == ["A", "B"]
    assert all(len(p["parents"]) == 1 for p in parents[:3])
    assert len(prompts) == 3 * 2 + 1             # reflection then revision per reflect, one per merge
    assert all(p.is_file() and "class Harness" in p.read_text(encoding="utf-8") for p in paths)
    # the fourth proposal overall, whatever the iteration split
    later = proposer(experience, 2, 2)
    kinds = [json.loads(p.with_name(p.stem + ".parent.json").read_text(encoding="utf-8"))["operator"]
             for p in later]
    assert kinds == ["reflect", "merge"]


def test_reflection_shows_lowest_tasks_with_input_output_reward(tmp_path):
    experience = make_experience(tmp_path, {"A": [1.0, 0.9, 0.0, 0.1, 0.2, 1.0]})
    prompts = []
    gepa.GepaProposer(lambda p: prompts.append(p) or REPLY, seed=0)(experience, 1, 1)
    first = prompts[0]
    for shown in (2, 3, 4):
        assert f"in-{shown}" in first and f"out-{shown}" in first
    for hidden in (0, 1, 5):
        assert f"in-{hidden}\n" not in first
    assert "reward: 0.0" in first
    assert "Reflection on the current harness" in prompts[1]   # revision sees the reflection


def test_reads_only_candidate_traces_of_this_run(tmp_path, monkeypatch):
    experience = make_experience(tmp_path, MATRIX)
    # decoys: a repeat trace and the held-out split must never feed the reward matrix
    write_trace(experience.candidate_dir("C") / "traces-1.jsonl", [9, 9, 9, 9])
    test_root = tmp_path / "store-test"
    (test_root / "traces").mkdir(parents=True)
    write_trace(test_root / "traces" / "C.jsonl", [9, 9, 9, 9])
    opened = []
    real = Path.read_text

    def spy(self, *args, **kwargs):
        opened.append(Path(self))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    matrix = gepa.reward_matrix(experience.candidates)
    gepa.GepaProposer(lambda p: REPLY)(experience, 1, 1)
    assert matrix["C"] == {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5}
    assert opened and all(test_root not in p.parents for p in opened)
    assert all(p.name in ("traces.jsonl", "harness.py", "scores.json") for p in opened)


def test_build_from_context(tmp_path):
    experience = make_experience(tmp_path, MATRIX)
    ctx = OptimizerContext(lambda p: REPLY, generic, seed=5)
    (path,) = build("gepa", ctx)(experience, 1, 1)
    assert path.parent == experience.root / "proposals" / "iteration-0001"
    assert path.with_name("candidate-00.reasoning.md").is_file()


def test_dominated_candidate_is_pruned_and_never_sampled():
    matrix = as_matrix({"A": [1.0, 1.0], "D": [1.0, 0.5]})
    wins = gepa.task_wins(matrix)
    assert wins == {"A": {0, 1}, "D": set()}
    assert gepa.pareto_set(matrix) == ["A"]
    rng = random.Random(1)
    assert {gepa.sample_parent(rng, wins) for _ in range(100)} == {"A"}


def test_ties_give_every_tied_candidate_the_win():
    wins = gepa.task_wins(as_matrix({"A": [1.0, 0.0], "B": [1.0, 0.0]}))
    assert wins["A"] == {0, 1} and wins["B"] == {0, 1}      # identical rows: neither is strictly dominated
    wins = gepa.task_wins(as_matrix({"A": [1.0, 0.0], "B": [1.0, 0.5], "C": [0.0, 0.0]}))
    assert wins["B"] == {0, 1} and wins["A"] == set()


def test_partial_candidate_cannot_win_unscored_tasks():
    matrix = {"A": {0: 0.5, 1: 0.5, 2: 0.5}, "P": {0: 0.9}}      # P was killed after task 0
    wins = gepa.task_wins(matrix)
    assert wins == {"A": set(), "P": {0}}                        # tasks 1 and 2 are not commonly scored
    matrix = {"A": {0: 0.5, 1: 0.5}, "P": {0: 0.1, 2: 0.9}}
    assert gepa.task_wins(matrix) == {"A": {0}, "P": set()}      # only task 0 is common; A beats P there


def test_invalid_candidates_are_never_parents(tmp_path):
    experience = make_experience(tmp_path, {"A": [0.5, 0.5], "BAD": [1.0, 1.0]})
    experience.write_result(EvaluationResult("BAD", 1.0, valid=False))
    paths = gepa.GepaProposer(lambda p: REPLY)(experience, 1, 3)
    for path in paths:
        info = json.loads(path.with_name(path.stem + ".parent.json").read_text(encoding="utf-8"))
        assert info["parents"] == ["A"]
