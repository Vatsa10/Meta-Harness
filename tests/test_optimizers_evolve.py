import json
import random
from pathlib import Path

from meta_harness.core import EvaluationResult, FilesystemExperience
from meta_harness.domains import generic
from meta_harness.optimizers import OptimizerContext, all_optimizers, build
from meta_harness.optimizers import openevolve, ttt_discover
from meta_harness.providers import model_from_environment

REWRITE = "class Harness:\n    def run(self, task, model, trace):\n        return 'x'\n"


def model(tmp_path):
    path = tmp_path / "script.jsonl"
    path.write_text(json.dumps({"default": f"Rewrite.\n```python\n{REWRITE}```\n"}), encoding="utf-8")
    return model_from_environment("scripted", str(path))


def experience(tmp_path, scores, parents=None):
    exp = FilesystemExperience(tmp_path / "store")
    for i, (cid, score) in enumerate(scores.items()):
        directory = exp.candidate_dir(cid)
        directory.mkdir(parents=True)
        (directory / "harness.py").write_text(f"# {cid}\n" + REWRITE, encoding="utf-8")
        exp.write_result(EvaluationResult(cid, score, context_cost=float(i * 10)))
        if parents and cid in parents:
            (directory / "parent.json").write_text(json.dumps({"parents": [parents[cid]]}), encoding="utf-8")
    return exp


SIX = {"s0": 0.1, "s1": 0.25, "s2": 0.4, "s3": 0.55, "s4": 0.7, "s5": 0.9}


def ctx(tmp_path, **kw):
    return OptimizerContext(model(tmp_path), generic, seed=3, **kw)


def test_both_registered():
    assert "openevolve" in all_optimizers() and "ttt_discover" in all_optimizers()


def test_openevolve_parent_from_archive_and_parent_json(tmp_path):
    exp = experience(tmp_path, SIX)
    paths = build("openevolve", ctx(tmp_path))(exp, 1, 2)
    assert [p.name for p in paths] == ["candidate-00.py", "candidate-01.py"]
    archive_ids = {r.candidate_id for r in openevolve.build_archive(exp.results()).values()}
    for p in paths:
        meta = json.loads(p.with_name(p.stem + ".parent.json").read_text(encoding="utf-8"))
        assert meta["parents"][0] in archive_ids and meta["operator"] in ("exploit", "explore")
        assert p.with_name(p.stem + ".reasoning.md").is_file()
        assert "class Harness" in p.read_text(encoding="utf-8")


def test_openevolve_prompt_has_parent_and_two_inspirations(tmp_path):
    seen = []
    exp = experience(tmp_path, SIX)
    proposer = openevolve.OpenEvolve(lambda prompt: seen.append(prompt) or f"```python\n{REWRITE}```", seed=1)
    proposer(exp, 1, 1)
    assert seen[0].count("Inspiration 1 (") + seen[0].count("Inspiration 2 (") == 2 and "Inspiration 3" not in seen[0] and "Parent program" in seen[0]
    parent = [line for line in seen[0].splitlines() if line.startswith("Parent program")][0]
    pid = parent.split("id ")[1].split(")")[0]
    assert f"id {pid}," not in seen[0].split("Inspiration programs")[1]
    assert "s5" in seen[0] or pid == "s5"


def test_openevolve_exploit_ratio(tmp_path):
    exp = experience(tmp_path, SIX)
    archive = openevolve.build_archive(exp.results())
    members = {"s0", "s2", "s4", "s5"}
    modes = [openevolve.sample_parent(random.Random(i), archive, members)[1] for i in range(200)]
    assert 0.6 <= modes.count("exploit") / 200 <= 0.8


def test_openevolve_migration_at_iteration_five(tmp_path):
    exp = experience(tmp_path, SIX)
    proposer = build("openevolve", ctx(tmp_path))
    proposer(exp, 4, 1)
    state = tmp_path / "store" / "proposals" / "openevolve_state.json"
    assert json.loads(state.read_text(encoding="utf-8"))["migrations"] == {}
    proposer(exp, 5, 1)
    moves = json.loads(state.read_text(encoding="utf-8"))["migrations"]["5"]
    assert moves and all(0 <= island < 3 for island in moves.values())
    best = max((r for r in exp.results() if openevolve.island_of(r.candidate_id) == 0),
               key=lambda r: r.score, default=None)
    if best:
        assert moves[best.candidate_id] == 1


def tree(tmp_path, scores, parents):
    exp = experience(tmp_path, scores, parents)
    results = exp.results()
    return exp, results, ttt_discover.parent_map(exp.root, [r.candidate_id for r in results])


def test_ttt_prefers_unvisited_child(tmp_path):
    _, results, parents = tree(tmp_path, {"a-root": 0.5, "child-a": 0.8, "child-b": 0.8},
                               {"child-a": "a-root", "child-b": "a-root"})
    visits = {"a-root": 2, "child-a": 2}
    assert ttt_discover.select(results, parents, visits) == "child-b"


def test_ttt_exploration_term_scales_with_parent_visits(tmp_path):
    # the unvisited lower-Q child wins only because of the sqrt(N(parent)) factor
    _, results, parents = tree(tmp_path, {"a-root": 0.2, "child-a": 0.8, "child-b": 0.5},
                               {"child-a": "a-root", "child-b": "a-root"})
    assert ttt_discover.select(results, parents, {"a-root": 9, "child-a": 1}) == "child-b"
    assert ttt_discover.select(results, parents, {"a-root": 1, "child-a": 1}) == "child-a"


def test_ttt_higher_q_once_visits_equalise(tmp_path):
    _, results, parents = tree(tmp_path, {"a-root": 0.5, "child-a": 0.8, "child-b": 0.6},
                               {"child-a": "a-root", "child-b": "a-root"})
    visits = {"a-root": 4, "child-a": 2, "child-b": 2}
    assert ttt_discover.select(results, parents, visits) == "child-a"


def test_ttt_state_round_trips_and_expansion(tmp_path):
    exp = experience(tmp_path, {"r": 0.5, "c": 0.8}, {"c": "r"})
    proposer = build("ttt_discover", ctx(tmp_path))
    paths = proposer(exp, 1, 2)
    state = ttt_discover.load_state(exp.root)
    assert sum(state["visits"].values()) >= 2
    ttt_discover.save_state(exp.root, state)
    assert ttt_discover.load_state(exp.root) == state
    raw = json.loads((exp.root / "proposals" / "ttt_state.json").read_text(encoding="utf-8"))
    assert raw == state
    meta = json.loads(paths[0].with_name("candidate-00.parent.json").read_text(encoding="utf-8"))
    assert meta["parents"][0] in {"r", "c"} and meta["operator"]


def test_neither_opens_test_root(tmp_path, monkeypatch):
    test_root = tmp_path / "test_split"
    test_root.mkdir()
    (test_root / "secret.jsonl").write_text("x", encoding="utf-8")
    opened: list[Path] = []
    real_open, real_read = Path.open, Path.read_text

    def spy_open(self, *a, **k):
        opened.append(Path(self))
        return real_open(self, *a, **k)

    def spy_read(self, *a, **k):
        opened.append(Path(self))
        return real_read(self, *a, **k)

    exp = experience(tmp_path, SIX, {"s5": "s4", "s4": "s3"})
    monkeypatch.setattr(Path, "open", spy_open)
    monkeypatch.setattr(Path, "read_text", spy_read)
    for name in ("openevolve", "ttt_discover"):
        build(name, ctx(tmp_path, test_root=test_root))(exp, 5, 2)
    assert opened and not [p for p in opened if test_root in p.parents or p == test_root]
