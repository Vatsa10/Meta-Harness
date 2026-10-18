import json

from meta_harness.cache import proposer_cache_enabled, proposer_call_key
from meta_harness.core import FilesystemExperience
from meta_harness.proposer_agent import AgentProposer
from meta_harness.providers import model_from_environment

NEW = "class Harness:\n    def run(self, task, model, trace):\n        return 'fruit'\n"
WRITE = {"name": "write_candidate", "arguments": {"source": NEW, "rationale": "always fruit"}}


class Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return self.inner.chat(*args, **kwargs)


def setup(tmp_path, cache=None, rows=2, view="scores"):
    root = tmp_path / "store"
    cand = root / "candidates" / "initial-0000"
    cand.mkdir(parents=True)
    (cand / "harness.py").write_text("class Harness: pass\n", encoding="utf-8")
    (cand / "scores.json").write_text(json.dumps({"score": 0.5}), encoding="utf-8")
    (root / "run.json").write_text("{}", encoding="utf-8")
    script = tmp_path / "p.jsonl"
    script.write_text("\n".join(json.dumps({"tool_calls": [WRITE]}) for _ in range(rows)),
                      encoding="utf-8")
    model = Counting(model_from_environment("scripted", str(script)))
    proposer = AgentProposer(model, view=view, max_tool_calls=5, cache=cache)
    return FilesystemExperience(root), proposer, model, cand


def log(experience, iteration):
    path = experience.root / "proposals" / f"iteration-{iteration:04d}" / "candidate-00.proposer_log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_identical_context_returns_the_cached_proposal(tmp_path):
    experience, proposer, model, _ = setup(tmp_path, cache=tmp_path / "cache")
    first = proposer(experience, 1, 1)
    second = proposer(experience, 1, 1)
    assert model.calls == 1
    assert second[0].read_text(encoding="utf-8") == first[0].read_text(encoding="utf-8") == NEW
    entries = log(experience, 1)
    assert entries[0]["cache_hit"] is False
    assert entries[-1]["event"] == "cache_hit" and entries[-1]["cache_hit"] is True
    proposal = json.loads((second[0].parent / "candidate-00.proposal.json").read_text(encoding="utf-8"))
    assert proposal["cache_hit"] is True


def test_any_visible_file_change_is_a_miss(tmp_path):
    experience, proposer, model, cand = setup(tmp_path, cache=tmp_path / "cache")
    proposer(experience, 1, 1)
    (cand / "scores.json").write_text(json.dumps({"score": 0.6}), encoding="utf-8")
    proposer(experience, 1, 1)
    assert model.calls == 2
    assert all(e.get("event") != "cache_hit" for e in log(experience, 1))


def test_cache_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("META_HARNESS_PROPOSER_CACHE", raising=False)
    experience, proposer, model, _ = setup(tmp_path)
    proposer(experience, 1, 1)
    proposer(experience, 1, 1)
    assert proposer.cache is None and model.calls == 2


def test_env_flag_turns_the_cache_on(tmp_path, monkeypatch):
    monkeypatch.setenv("META_HARNESS_PROPOSER_CACHE", "1")
    monkeypatch.setenv("META_HARNESS_PROPOSER_CACHE_DIR", str(tmp_path / "envcache"))
    experience, proposer, model, _ = setup(tmp_path)
    proposer(experience, 1, 1)
    proposer(experience, 1, 1)
    assert model.calls == 1 and (tmp_path / "envcache").is_dir()
    assert proposer_cache_enabled({"META_HARNESS_PROPOSER_CACHE": "1"})
    assert not proposer_cache_enabled({})


def test_key_covers_files_workspace_model_and_prompt(tmp_path):
    view = tmp_path / "v"
    view.mkdir()
    (view / "a.txt").write_text("x", encoding="utf-8")
    base = proposer_call_key(view, tmp_path / "w1", "m", "p")
    assert base == proposer_call_key(view, tmp_path / "w1", "m", "p")
    assert base != proposer_call_key(view, tmp_path / "w2", "m", "p")
    assert base != proposer_call_key(view, tmp_path / "w1", "m2", "p")
    assert base != proposer_call_key(view, tmp_path / "w1", "m", "p2")
    (view / "a.txt").rename(view / "b.txt")
    assert base != proposer_call_key(view, tmp_path / "w1", "m", "p")
