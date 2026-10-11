import importlib.util
import json
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness.commands import sweep as sweep_cmd
from meta_harness.core import HarnessValidationError, TraceRecorder
from meta_harness.sandbox import validate_in_subprocess

ROOT = Path(__file__).resolve().parent.parent / "baselines" / "classification"
LABELS = ["fruit", "vehicle"]


def _load(name: str, *args, **kwargs):
    spec = importlib.util.spec_from_file_location("cls_" + name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Harness(*args, **kwargs)


def _task(index: int, label: str = "fruit") -> dict:
    return {"input": f"item {index}", "label": label, "labels": LABELS}


def _answer(prompt, **_):
    return "fruit"


@pytest.fixture
def trace(tmp_path):
    with TraceRecorder(tmp_path / "t.jsonl") as recorder:
        yield recorder


STATEFUL = ["zero_shot", "few_shot", "random_few_shot", "ace", "mce"]


def _call_learn(harness, task, result, model):
    try:
        harness.learn(task, result, model)
    except TypeError:
        harness.learn(task, result)


@pytest.mark.parametrize("name", STATEFUL)
def test_run_never_mutates_memory_and_learn_adds(name, trace):
    harness = _load(name)
    for index in range(2):
        harness.run(_task(index), _answer, trace)
        harness.run(_task(index), _answer, trace)
    assert harness.memory == []
    assert not getattr(harness, "playbook", []) and not getattr(harness, "skills", [])
    _call_learn(harness, _task(0), {"output": "fruit"}, _answer)
    assert harness.memory == [("item 0", "fruit")]


def test_few_shot_all_uses_every_example_and_n_limits(trace, monkeypatch):
    monkeypatch.setenv("META_HARNESS_FEW_SHOT_N", "all")
    everything = _load("few_shot")
    monkeypatch.setenv("META_HARNESS_FEW_SHOT_N", "4")
    four = _load("few_shot")
    prompts = {}
    for key, harness in (("all", everything), ("4", four)):
        for index in range(40):
            harness.learn(_task(index), {"output": "fruit"})
        harness.run(_task(99), lambda p, **_: prompts.setdefault(key, p) and "fruit", trace)
    assert prompts["all"].count("Example:") == 40
    assert prompts["4"].count("Example:") == 4


@pytest.mark.parametrize("name", ["few_shot", "random_few_shot", "dense_few_shot"])
def test_n_must_be_in_the_paper_set(name, monkeypatch):
    for value in ("4", "8", "16", "32", "all"):
        monkeypatch.setenv("META_HARNESS_FEW_SHOT_N", value)
        _load(name)
    monkeypatch.setenv("META_HARNESS_FEW_SHOT_N", "5")
    with pytest.raises(ValueError):
        _load(name)


def test_random_few_shot_is_deterministic_per_input(trace):
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return "fruit"

    first, second = _load("random_few_shot", 4), _load("random_few_shot", 4)
    for harness in (first, second):
        for index in range(30):
            harness.learn(_task(index), {"output": "fruit"})
    first.run(_task(100), model, trace)
    second.run(_task(100), model, trace)
    first.run(_task(101), model, trace)
    assert prompts[0] == prompts[1]
    assert prompts[0].count("Example:") == 4
    assert prompts[0].split("Input:")[0] != prompts[2].split("Input:")[0]


def _fake_embed(calls):
    vectors = {"the cat sat": [1.0, 0.0, 0.0], "a kitten": [0.9, 0.1, 0.0],
               "stock market": [0.0, 1.0, 0.0], "tax law": [0.0, 0.9, 0.1]}

    def fake(texts, provider="openai", model="text-embedding-3-small", **_):
        calls.append((list(texts), provider, model))
        return [vectors[t] for t in texts]
    return fake


def test_dense_few_shot_ranks_the_planted_neighbour_first_and_caches(trace, monkeypatch):
    monkeypatch.delenv("META_HARNESS_EMBED_PROVIDER", raising=False)
    monkeypatch.delenv("META_HARNESS_EMBED_MODEL", raising=False)
    calls = []
    harness = _load("dense_few_shot", 2, embed_fn=_fake_embed(calls))
    for text, label in [("stock market", "vehicle"), ("a kitten", "fruit"), ("tax law", "vehicle")]:
        harness.learn({"input": text, "label": label, "labels": LABELS}, {"output": label})
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return "fruit"

    harness.run({"input": "the cat sat", "labels": LABELS}, model, trace)
    assert prompts[0].index("Example: a kitten") < prompts[0].index("Example:", prompts[0].index("Example:") + 1)
    assert prompts[0].count("Example:") == 2
    assert calls[0][1:] == ("openai", "text-embedding-3-small")
    harness.run({"input": "the cat sat", "labels": LABELS}, model, trace)
    assert len(calls) == 1   # every text is embedded once per harness


def test_dense_few_shot_env_selects_provider_and_model(trace, monkeypatch):
    monkeypatch.setenv("META_HARNESS_EMBED_PROVIDER", "groq")
    monkeypatch.setenv("META_HARNESS_EMBED_MODEL", "m-1")
    calls = []
    harness = _load("dense_few_shot", embed_fn=_fake_embed(calls))
    harness.run({"input": "the cat sat", "labels": LABELS}, _answer, trace)
    assert calls[0][1:] == ("groq", "m-1")


def test_dense_few_shot_raises_without_a_provider(trace, monkeypatch):
    for var in ("OPENAI_API_KEY", "META_HARNESS_EMBED_PROVIDER", "META_HARNESS_EMBED_MODEL"):
        monkeypatch.delenv(var, raising=False)
    harness = _load("dense_few_shot")
    with pytest.raises(RuntimeError, match="embedding provider"):
        harness.run({"input": "anything", "labels": LABELS}, _answer, trace)
    with pytest.raises(HarnessValidationError):
        validate_in_subprocess(ROOT / "dense_few_shot.py", timeout=60.0)


# -- ACE ----------------------------------------------------------------------------------------

def _playbook(texts):
    harness = _load("ace")
    harness.apply_delta("\n".join(f"ADD: {t}" for t in texts))
    return harness


def test_ace_curator_update_touches_only_the_named_bullet():
    harness = _playbook(["alpha", "beta", "gamma"])
    harness.playbook[1]["helpful"] = 3
    applied = harness.apply_delta("UPDATE 2: beta revised")
    assert applied == [("UPDATE", 2)]
    assert [b["text"] for b in harness.playbook] == ["alpha", "beta revised", "gamma"]
    assert harness.playbook[1]["helpful"] == 3
    harness.apply_delta("REMOVE 1\nUPDATE 9: ghost\nnot an op")
    assert [b["id"] for b in harness.playbook] == [2, 3]
    harness.apply_delta("ADD: delta")
    assert [b["id"] for b in harness.playbook] == [2, 3, 4]   # ids are never reused


def test_ace_prunes_a_bullet_whose_harmful_count_exceeds_helpful():
    harness = _playbook(["keep", "drop"])
    harness.playbook[0].update(helpful=2, harmful=2)
    harness.playbook[1].update(helpful=1, harmful=2)
    harness.prune()
    assert [b["text"] for b in harness.playbook] == ["keep"]


def test_ace_learn_scores_shown_bullets_and_applies_the_model_delta(trace):
    harness = _playbook(["rule one", "rule two"])
    for bullet in harness.playbook:
        bullet["helpful"] = 1   # one past success each, so a single blame does not prune
    replies = iter(["shape: look at the shape", "UPDATE 2: rule two sharpened\nADD: rule three"])
    model = lambda prompt, **_: next(replies) if "mistake" in prompt or "curate" in prompt else "vehicle"
    prediction = harness.run(_task(0, "fruit"), model, trace)
    assert prediction == "vehicle"
    harness.learn(_task(0, "fruit"), {"output": prediction}, model)
    assert [b["text"] for b in harness.playbook] == ["rule one", "rule two sharpened", "rule three"]
    assert [b["harmful"] for b in harness.playbook] == [1, 1, 0]   # shown bullets were blamed
    assert harness.insights == ["shape: look at the shape"]


def test_ace_learn_drops_a_bullet_that_keeps_misleading(trace):
    harness = _playbook(["bad rule"])
    harness.playbook[0].update(helpful=0, harmful=0)
    model = lambda prompt, **_: "" if "mistake" in prompt else "vehicle"
    harness.run(_task(0, "fruit"), model, trace)
    harness.learn(_task(0, "fruit"), {"output": "vehicle"}, model)
    assert harness.playbook == []


# -- MCE ----------------------------------------------------------------------------------------

def test_mce_evolves_skills_from_errors_and_retains_by_usefulness(trace):
    harness = _load("mce", 2)
    model = lambda prompt, **_: ("SHAPE: check the shape" if "library of classification" in prompt
                                 else "Skill: 1\nLabel: vehicle")
    harness.run(_task(0, "fruit"), model, trace)
    harness.learn(_task(0, "fruit"), {"output": "vehicle"}, model)
    assert [s["text"] for s in harness.skills] == ["SHAPE: check the shape"]
    harness.skills.append({"id": 7, "text": "WINNER: good", "used": 4, "helpful": 4})
    harness.skills.append({"id": 8, "text": "LOSER: bad", "used": 4, "helpful": 0})
    harness._prune()
    assert sorted(s["id"] for s in harness.skills) == [1, 7]
    model_skill = lambda prompt, **_: "Skill: 7\nLabel: fruit"
    assert harness.run(_task(1), model_skill, trace) == "fruit"
    harness.learn(_task(1), {"output": "fruit"}, model_skill)
    skill = next(s for s in harness.skills if s["id"] == 7)
    assert (skill["used"], skill["helpful"]) == (5, 5)


# -- sweep --------------------------------------------------------------------------------------

def test_sweep_writes_one_row_per_setting(tmp_path, monkeypatch, capsys):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(
        json.dumps({"input": f"item {i}", "label": "fruit" if i % 2 else "vehicle"}) for i in range(12)),
        encoding="utf-8")
    monkeypatch.setattr(sweep_cmd, "model_from_environment", lambda *a, **k: (lambda prompt, **_: "fruit"))
    monkeypatch.delenv("META_HARNESS_FEW_SHOT_N", raising=False)
    root = tmp_path / "out"
    code = cli.main(["sweep", "--harness", str(ROOT / "few_shot.py"),
                     "--env", "META_HARNESS_FEW_SHOT_N=4,8,16,32,all",
                     "--tasks", str(tasks), "--provider", "unikey", "--model", "m",
                     "--root", str(root), "--no-cache"])
    assert code == 0
    rows = json.loads((root / "sweep.json").read_text(encoding="utf-8"))["rows"]
    assert [r["setting"] for r in rows] == ["4", "8", "16", "32", "all"]
    assert all(r["valid"] and r["examples"] > 0 for r in rows)
    import os
    assert "META_HARNESS_FEW_SHOT_N" not in os.environ
    capsys.readouterr()


def test_sweep_rejects_a_malformed_env_spec():
    with pytest.raises(ValueError):
        sweep_cmd.parse_env("NOVALUES")
