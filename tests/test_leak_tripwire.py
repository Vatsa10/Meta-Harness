import json

import pytest

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner
from meta_harness.proposer_agent import (LEAK_REASON, AgentProposer, ProposerLeak,
                                         heldout_digests, scan_for_leaks)
from meta_harness.providers import model_from_environment

SEED = "class Harness:\n    def run(self, task, model, trace):\n        return model(task['input'])\n"
NEW = "class Harness:\n    def run(self, task, model, trace):\n        return 'fruit'\n"
WRITE = {"name": "write_candidate", "arguments": {"source": NEW, "rationale": "always fruit"}}
SEARCH = [{"input": "apple", "label": "fruit"}, {"input": "car", "label": "vehicle"}]
HELDOUT = [{"input": "a pear from the orchard", "label": "fruit",
            "answer": "a pear is a pome fruit of the genus Pyrus"}]


class Counting:
    def __init__(self, inner):
        self.inner, self.calls = inner, 0

    def chat(self, *args, **kwargs):
        self.calls += 1
        return self.inner.chat(*args, **kwargs)


def make_view(tmp_path):
    view = tmp_path / "view"
    (view / "candidates" / "c1").mkdir(parents=True)
    (view / "run.json").write_text(json.dumps({"tasks": SEARCH}), encoding="utf-8")
    (view / "candidates" / "c1" / "traces.jsonl").write_text(
        json.dumps({"event": "task_end", "input": "apple", "reward": 1}) + "\n", encoding="utf-8")
    return view


def test_clean_view_passes(tmp_path):
    scan_for_leaks(make_view(tmp_path), tmp_path / "store-test", heldout_digests(HELDOUT))


@pytest.mark.parametrize("plant", [
    lambda v, t: (v / "candidates" / "c1" / "notes.md").write_text(f"see {t}", encoding="utf-8"),
    lambda v, t: (v / "candidates" / "c1" / "notes.md").write_text("store-test/x", encoding="utf-8"),
    lambda v, t: (v / "candidates" / "c1" / "test_results.json").write_text("{}", encoding="utf-8"),
    lambda v, t: (v / "candidates" / "c1" / "judge.txt").write_text("ok\nVERDICT: PASS", encoding="utf-8"),
    lambda v, t: (v / "candidates" / "c1" / "traces.jsonl").write_text(
        json.dumps({"event": "model_call", "prompt": "a pear from the orchard"}) + "\n", encoding="utf-8"),
    lambda v, t: (v / "candidates" / "c1" / "harness.py").write_text(
        "EXAMPLES = '''\na pear from the orchard\n'''\n", encoding="utf-8"),
    lambda v, t: (v / "candidates" / "c1" / "summary.md").write_text(
        json.dumps({"expected": "a pear is a pome fruit of the genus Pyrus"}), encoding="utf-8"),
])
def test_planted_heldout_marker_is_caught(tmp_path, plant):
    view = make_view(tmp_path)
    plant(view, tmp_path / "store-test")
    with pytest.raises(ProposerLeak, match=LEAK_REASON):
        scan_for_leaks(view, tmp_path / "store-test", heldout_digests(HELDOUT))


def test_digest_set_carries_no_raw_heldout_text():
    digests = heldout_digests(HELDOUT)
    assert all(len(d) == 64 and "pear" not in d for d in digests)


def run(tmp_path, plant=None):
    seed = tmp_path / "seed.py"
    seed.write_text(SEED, encoding="utf-8")
    rows = tmp_path / "p.jsonl"
    rows.write_text(json.dumps({"tool_calls": [WRITE]}), encoding="utf-8")
    model = Counting(model_from_environment("scripted", str(rows)))
    proposer = AgentProposer(model, max_tool_calls=5, heldout_hashes=heldout_digests(HELDOUT))
    if plant:
        original = proposer._tripwire

        def planted(view_root, output, splits):
            plant(view_root)
            return original(view_root, output, splits)
        proposer._tripwire = planted
    config = SearchConfig(root=tmp_path / "store", iterations=1)
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "vehicle"), proposer).run(
        [seed], SEARCH, HELDOUT)
    return config.root, model


def test_session_aborts_on_a_planted_leak_and_core_records_it(tmp_path):
    def plant(view):
        (view / "candidates" / "initial-0000" / "leak.txt").write_text(
            "a pear from the orchard", encoding="utf-8")
    root, model = run(tmp_path, plant)
    out = root / "proposals" / "iteration-0001"
    assert model.calls == 0
    assert LEAK_REASON in (out / "proposer.error").read_text(encoding="utf-8")
    assert json.loads((out / "proposer.leak.json").read_text(encoding="utf-8"))["reason"] == LEAK_REASON
    assert not (root / "candidates" / "iteration-0001-00").exists()


def test_session_proceeds_on_a_clean_view(tmp_path):
    root, model = run(tmp_path)
    assert model.calls == 1
    assert (root / "candidates" / "iteration-0001-00" / "harness.py").is_file()
    assert not (root / "proposals" / "iteration-0001" / "proposer.error").exists()
