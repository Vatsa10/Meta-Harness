import importlib.util
import json
from pathlib import Path

import pytest

from meta_harness.core import TraceRecorder
from meta_harness.domains import get_domain
from meta_harness.math_harnesses import MathRetrievalHarness, adaptive_k, technique_stated_early
from meta_harness.sandbox import validate_in_subprocess

SEEDS = Path(__file__).resolve().parent.parent / "baselines" / "math"


class _Trace:
    def __init__(self):
        self.events = []

    def event(self, name, data=None):
        self.events.append((name, data))


def _item(i, problem, solution="Therefore the answer holds.", difficulty=3, source="MATH"):
    return {"id": f"x{i}", "problem": problem, "solution": solution, "difficulty": difficulty, "source": source}


ROUTE_TABLE = [
    ("In triangle ABC the angle at A is 40 degrees. Find angle B.", "geometry"),
    ("Let \\triangle ABC be isosceles; prove the circle is tangent.", "geometry"),
    ("Prove that for every prime p, p divides 2^p - 2.", "number_theory"),
    ("Find all integers n with n^2 congruent to 1 modulo 8.", "number_theory"),
    ("How many ways can 5 people choose subsets of a set?", "combinatorics"),
    ("Each vertex of K_6 is colored red or blue; prove a monochromatic triangle exists.", "combinatorics"),
    ("Solve x^2 - 5x + 6 = 0 for real x.", "default"),
    ("Evaluate \\frac{1}{2} + \\frac{1}{3}.", "default"),
]


@pytest.mark.parametrize("problem,route", ROUTE_TABLE)
def test_route_table(problem, route):
    assert MathRetrievalHarness.route(problem) == route


def test_geometry_hard_item_comes_from_separate_index_outside_main_top_10():
    corpus = [_item(i, "triangle ABC angle bisector circumcircle tangent chord") for i in range(12)]
    corpus.append(_item(99, "olympiad configuration with a circle", difficulty=8, source="NuminaMath"))
    harness = MathRetrievalHarness(corpus)
    query = "triangle ABC angle bisector circumcircle tangent chord"
    main_top10 = {r.item["id"] for r in harness.index.search(query, 10)}
    assert "x99" not in main_top10
    ids = [r.item["id"] for r in harness._retrieve(query, "geometry")]
    assert "x99" in ids
    assert len(ids) == 3 and len(set(ids)) == 3


def test_hard_index_excludes_easy_and_non_numina_items():
    corpus = [_item(1, "triangle one", difficulty=6, source="NuminaMath"),
              _item(2, "triangle two", difficulty=9, source="OpenMathReasoning"),
              _item(3, "triangle three", difficulty=7, source="NuminaMath")]
    harness = MathRetrievalHarness(corpus)
    assert [x["id"] for x in harness.hard_index.items] == ["x3"]


def test_no_duplicate_ids_when_hard_item_is_also_top_main_hit():
    corpus = [_item(1, "triangle angle bisector", difficulty=9, source="NuminaMath"),
              _item(2, "triangle angle foo"), _item(3, "triangle angle bar"), _item(4, "unrelated")]
    harness = MathRetrievalHarness(corpus)
    ids = [r.item["id"] for r in harness._retrieve("triangle angle bisector", "geometry")]
    assert ids.count("x1") == 1 and len(ids) == len(set(ids)) == 3


def test_adaptive_k_values():
    assert adaptive_k([10.0, 3.0, 2.0]) == 1
    assert adaptive_k([10.0, 9.5, 9.0, 8.9]) == 3
    assert adaptive_k([10.0, 9.5, 4.0]) == 2
    assert adaptive_k([]) == 0


def test_default_route_adaptive_k_end_to_end():
    close = [_item(i, "solve the quadratic equation for real roots") for i in range(3)]
    filler = [_item(10 + i, f"unrelated topic number {i} alpha{i}") for i in range(5)]
    assert len(MathRetrievalHarness(close + filler)._retrieve("solve the quadratic equation for real roots", "default")) == 3
    dominant = [_item(0, "solve the quadratic equation for real roots now"),
                _item(1, "quadratic"), _item(2, "equation")]
    result = MathRetrievalHarness(dominant + filler)._retrieve("solve the quadratic equation for real roots now", "default")
    assert [r.item["id"] for r in result] == ["x0"]


def test_technique_check():
    assert technique_stated_early("We use induction on n. Base case is trivial.")
    assert technique_stated_early("By the pigeonhole principle two of them agree.")
    assert not technique_stated_early("Therefore the result follows after computing.")
    assert not technique_stated_early("x" * 400 + " induction")


def test_number_theory_technique_bonus_changes_order():
    plain = _item(1, "prime divisibility integer problem", "Therefore, compute directly.")
    clever = _item(2, "prime divisibility integer problem", "Work modulo p and use Fermat's little theorem.")
    ids = [r.item["id"] for r in MathRetrievalHarness([plain, clever])._retrieve("prime divisibility integer problem", "number_theory")]
    assert ids[0] == "x2"


def test_long_items_never_retrieved_and_solutions_truncated():
    corpus = [_item(1, "triangle angle", "s" * 4001), _item(2, "p" * 4001 + " triangle angle"),
              _item(3, "triangle angle", "S" * 3500)]
    harness = MathRetrievalHarness(corpus)
    assert [x["id"] for x in harness.corpus] == ["x3"]
    prompts = []
    harness.run({"problem": "triangle angle"}, lambda p, **_: prompts.append(p) or "ok", _Trace())
    assert "S" * 3000 in prompts[0] and "S" * 3001 not in prompts[0]


def test_domain_contract_and_metric():
    domain = get_domain("math")
    assert domain.NAME == "math" and domain.PROTOCOL == "plain"
    assert domain.SEED_DIR == SEEDS and SEEDS.is_dir()
    metric = domain.METRICS["math_equivalence"]
    assert metric({"answer": "60"}, {"output": "60"}) == 1.0
    assert metric({"answer": "60"}, {"output": "61"}) == 0.0
    assert metric({"answer": "0.5"}, {"output": "0.50"}) == 1.0
    assert domain.adapt([{"problem": "p", "answer": "1"}]) == [{"problem": "p", "answer": "1"}]
    with pytest.raises(ValueError):
        domain.adapt([{"input": "no problem field"}])
    assert "geometry" in domain.PROMPT


def _load(name):
    spec = importlib.util.spec_from_file_location("seed_" + name, SEEDS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["router", "random_few_shot", "dense", "ace"])
def test_seed_passes_interface_validation(name):
    validate_in_subprocess(SEEDS / f"{name}.py", timeout=60.0)


@pytest.fixture
def corpus_env(tmp_path, monkeypatch):
    path = tmp_path / "corpus.jsonl"
    rows = [_item(1, "Find the angle in a triangle", "Angles sum to 180"),
            _item(2, "Show that 7 is prime", "Check divisors")]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setenv("META_HARNESS_CORPUS", str(path))


def test_router_seed_runs_through_corpus(tmp_path, corpus_env):
    harness = _load("router").Harness()
    prompts = []
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        out = harness.run({"problem": "What is the angle of a triangle?"},
                          lambda p, **_: prompts.append(p) or "\\boxed{60}", trace)
    assert out == "\\boxed{60}" and "Route: geometry" in prompts[0] and "Angles sum to 180" in prompts[0]


def test_random_few_shot_seed_includes_k_examples(tmp_path, corpus_env, monkeypatch):
    monkeypatch.setenv("META_HARNESS_RETRIEVAL_K", "2")
    prompts = []
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        _load("random_few_shot").Harness().run({"problem": "q"}, lambda p, **_: prompts.append(p) or "1", trace)
    assert prompts[0].count("Reference problem:") == 2


@pytest.mark.parametrize("k", [1, 5])
def test_dense_seed_uses_k_from_env_and_embeddings(tmp_path, corpus_env, monkeypatch, k):
    monkeypatch.setenv("META_HARNESS_RETRIEVAL_K", str(k))
    module = _load("dense")
    monkeypatch.setattr(module, "embed", lambda texts, *a, **kw: [[1.0, 0.0] if "angle" in t or "triangle" in t else [0.0, 1.0] for t in texts])
    prompts = []
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        module.Harness().run({"problem": "triangle angle?"}, lambda p, **_: prompts.append(p) or "1", trace)
    assert prompts[0].count("Reference problem:") == min(k, 2)
    assert prompts[0].index("Angles sum") < prompts[0].index("Check divisors") if k > 1 else "Check divisors" not in prompts[0]


def test_ace_seed_learns_bullet_from_wrong_answer(tmp_path):
    replies = iter(["\\boxed{5}", "Check small cases first.", "\\boxed{6}"])
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return next(replies)

    harness = _load("ace").Harness()
    with TraceRecorder(tmp_path / "t.jsonl") as trace:
        harness.run({"problem": "p1", "answer": "6"}, model, trace)
        harness.run({"problem": "p2", "answer": "6"}, model, trace)
    assert harness.playbook == ["Check small cases first."]
    assert "Check small cases first." in prompts[2]
