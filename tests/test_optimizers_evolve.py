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


def _archives(exp):
    valid = exp.results()
    ids = [r.candidate_id for r in valid]
    bounds = (min(r.score for r in valid), max(r.score for r in valid), max(r.context_cost for r in valid))
    state = {"migrations": {}}
    return {i: openevolve.build_archive(valid, openevolve.OpenEvolve.members(state, ids, i), bounds)
            for i in range(3)}


def test_openevolve_exploit_ratio(tmp_path):
    archives = _archives(experience(tmp_path, SIX))
    island = next(i for i, a in archives.items() if len(a) >= 2)
    modes = [openevolve.sample_parent(random.Random(i), archives, island)[1] for i in range(200)]
    assert 0.6 <= modes.count("exploit") / 200 <= 0.8
    assert set(modes) == {"exploit", "explore"}


def test_openevolve_empty_island_fallback_is_labelled(tmp_path):
    archives = _archives(experience(tmp_path, SIX))
    empty = {0: {}, 1: archives[0] or archives[1] or archives[2]}
    modes = {openevolve.sample_parent(random.Random(i), empty, 0)[1] for i in range(100)}
    assert modes == {"exploit_fallback_explore", "explore"}


def test_openevolve_scores_outside_unit_range_stay_in_deciles(tmp_path):
    exp = experience(tmp_path, {"a": -5.0, "b": 3.0, "c": 40.0})
    cells = openevolve.build_archive(exp.results())
    assert all(0 <= x <= 9 and 0 <= y <= 9 for x, y in cells)


def test_openevolve_migration_at_iteration_five(tmp_path):
    exp = experience(tmp_path, SIX)
    proposer = build("openevolve", ctx(tmp_path))
    proposer(exp, 4, 1)
    state_path = tmp_path / "store" / "proposals" / "openevolve_state.json"
    assert json.loads(state_path.read_text(encoding="utf-8"))["migrations"] == {}
    proposer(exp, 5, 1)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    moves = state["migrations"]["5"]
    ids = [r.candidate_id for r in exp.results()]
    assert moves
    for cid, target in moves.items():
        assert cid in openevolve.OpenEvolve.members(state, ids, target)
        assert openevolve.island_of(cid) != target
    # the migrant is now sampleable from the target island's archive
    cid, target = next(iter(moves.items()))
    valid = exp.results()
    bounds = (min(r.score for r in valid), max(r.score for r in valid), max(r.context_cost for r in valid))
    members = openevolve.OpenEvolve.members(state, ids, target)
    assert any(r.candidate_id == cid for r in openevolve.build_archive(valid, members, bounds).values())


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


def test_ttt_selecting_a_leaf_bumps_every_ancestor(tmp_path):
    exp = experience(tmp_path, {"r": 0.5, "m": 0.6, "leaf": 0.7}, {"m": "r", "leaf": "m"})
    ttt_discover.save_state(exp.root, {"visits": {"r": 1, "m": 1}})
    build("ttt_discover", ctx(tmp_path))(exp, 1, 1)
    visits = ttt_discover.load_state(exp.root)["visits"]
    assert visits == {"r": 2, "m": 2, "leaf": 1}


def test_ttt_two_seed_roots_share_the_virtual_root(tmp_path):
    _, results, parents = tree(tmp_path, {"seed-a": 0.5, "seed-b": 0.5}, {})
    assert parents == {"seed-a": "", "seed-b": ""}
    # root-level N is the sum of root visits: the unvisited seed wins
    assert ttt_discover.select(results, parents, {"seed-a": 3}) == "seed-b"
    assert ttt_discover.select(results, parents, {"seed-a": 3, "seed-b": 3}) == "seed-a"


def test_ttt_no_visit_for_unparseable_reply(tmp_path):
    exp = experience(tmp_path, {"r": 0.5})
    ttt_discover.TTTDiscover(lambda prompt: "no code here")(exp, 1, 1)
    assert ttt_discover.load_state(exp.root)["visits"] == {}


def test_ttt_cyclic_lineage_is_treated_as_roots(tmp_path):
    _, results, parents = tree(tmp_path, {"x": 0.4, "y": 0.6}, {"x": "y", "y": "x"})
    assert parents == {"x": "", "y": ""}
    assert ttt_discover.select(results, parents, {}) in {"x", "y"}


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
    assert opened
    assert not [p for p in opened if test_root.resolve() in Path(p).resolve().parents]
    assert (test_root / "secret.jsonl").resolve() not in {Path(p).resolve() for p in opened}
