import json

import pytest

from meta_harness.core import FilesystemExperience
from meta_harness.views import VIEW_FILES, VIEW_MODES, build_view


@pytest.fixture
def store(tmp_path):
    exp = FilesystemExperience(tmp_path / "exp")
    (exp.root / "run.json").write_text("{}", encoding="utf-8")
    (exp.root / "frontier.json").write_text("[]", encoding="utf-8")
    for cid in ("initial-0000", "iteration-0001-00"):
        d = exp.candidate_dir(cid)
        d.mkdir(parents=True)
        (d / "harness.py").write_text("# " + cid, encoding="utf-8")
        (d / "scores.json").write_text(json.dumps({"candidate_id": cid}), encoding="utf-8")
        (d / "traces.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
        (d / "proposer_reasoning.md").write_text("because", encoding="utf-8")
        (d / "proposer_log.jsonl").write_text("{}\n", encoding="utf-8")
    return exp


def test_scores_view_hides_traces_and_reasoning(store):
    view = build_view(store, "scores", store.root / "views" / "i1")
    names = {p.name for p in (view / "candidates" / "iteration-0001-00").iterdir()}
    assert names == {"harness.py", "scores.json"}
    assert (view / "run.json").is_file() and (view / "frontier.json").is_file()


def test_summary_view_adds_cached_summary(store):
    calls = []
    summarize = lambda prompt: calls.append(prompt) or "it fails on vans"
    build_view(store, "summary", store.root / "views" / "i1", summarize)
    build_view(store, "summary", store.root / "views" / "i2", summarize)
    assert len(calls) == 2, "one summary per candidate, cached across iterations"
    assert (store.root / "views" / "i2" / "candidates" / "initial-0000" / "summary.md").read_text(encoding="utf-8") == "it fails on vans"
    assert "traces.jsonl" not in {p.name for p in (store.root / "views" / "i2" / "candidates" / "initial-0000").iterdir()}
    assert (store.candidate_dir("initial-0000") / "summary.md").is_file()


def test_summary_prompt_is_truncated(store):
    (store.candidate_dir("initial-0000") / "harness.py").write_text("Q" * 7000, encoding="utf-8")
    (store.candidate_dir("initial-0000") / "traces.jsonl").write_text("Z" * 20000, encoding="utf-8")
    prompts = []
    build_view(store, "summary", store.root / "views" / "i1", lambda p: prompts.append(p) or "s")
    big = max(prompts, key=len)
    assert big.count("Q") == 6000 and big.count("Z") == 12000


def test_summary_requires_summarizer(store):
    with pytest.raises(ValueError):
        build_view(store, "summary", store.root / "views" / "i1")


def test_unknown_mode(store):
    with pytest.raises(ValueError):
        build_view(store, "bogus", store.root / "views" / "i1")


def test_full_view_is_the_root(store):
    assert build_view(store, "full", store.root / "views" / "x") == store.root


def test_seed_view_only_has_initial_candidates(store):
    view = build_view(store, "seed", store.root / "views" / "s")
    assert [p.name for p in (view / "candidates").iterdir()] == ["initial-0000"]


def test_constants():
    assert VIEW_MODES == ("scores", "summary", "full")
    assert VIEW_FILES["scores"] == ("harness.py", "scores.json")
