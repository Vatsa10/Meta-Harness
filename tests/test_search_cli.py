import json
import os
from pathlib import Path

import pytest

from meta_harness import __main__ as cli

FIX = Path(__file__).parent / "fixtures" / "smoke"


def _smoke(root, *extra):
    return cli.main(["search", "--harness", str(FIX / "seed.py"), "--data", str(FIX / "data.jsonl"),
                     "--metric", "exact", "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
                     "--proposer-provider", "scripted", "--proposer-model", str(FIX / "proposer.jsonl"),
                     "--root", str(root), "--no-cache", *extra])


def test_search_end_to_end_offline(tmp_path, capsys):
    root = tmp_path / "store"
    code = cli.main(["search", "--harness", str(FIX / "seed.py"), "--data", str(FIX / "data.jsonl"),
                     "--metric", "exact", "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
                     "--proposer-provider", "scripted", "--proposer-model", str(FIX / "proposer.jsonl"),
                     "--iterations", "1", "--root", str(root), "--no-cache"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["search_tasks"] == 7 and out["test_tasks"] == 3
    proposed = root / "candidates" / "iteration-0001-00"
    assert (proposed / "proposer_log.jsonl").is_file() and (proposed / "traces.jsonl").is_file()
    test = json.loads(Path(out["test_results"]).read_text(encoding="utf-8"))
    assert {r["split"] for r in test["results"]} == {"test"}
    manifest = json.loads((root / "run.json").read_text(encoding="utf-8"))
    test_inputs = {json.loads(l)["input"] for l in (FIX / "data.jsonl").read_text(encoding="utf-8").splitlines()} \
        - {t["input"] for t in manifest["tasks"]}
    assert len(test_inputs) == 3, "run.json holds the search split only"
    assert "API_KEY" not in (root / "run.json").read_text(encoding="utf-8")
    assert out["verdicts"] == {"accepted": 2}
    assert out["budget"]["proposer_calls"] > 0 and out["budget"]["harness_evaluations"] == 2
    assert out["context_tokens"]["model"] > 0 and out["context_tokens"]["proposer"] > 0
    assert out["cost_usd"] is None, "no price given: billed cost is unknown, never guessed"


def test_module_metric_and_explicit_split(tmp_path, capsys):
    (tmp_path / "score.py").write_text("def always(task, result):\n    return 1.0\n", encoding="utf-8")
    code = cli.main(["search", "--harness", str(FIX / "seed.py"), "--search", str(FIX / "data.jsonl"),
                     "--test", str(FIX / "data.jsonl"), "--metric", f"{tmp_path / 'score.py'}:always",
                     "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
                     "--iterations", "0", "--root", str(tmp_path / "s"), "--no-cache"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["frontier"][0]["score"] == 1.0


def test_missing_key_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    code = cli.main(["search", "--harness", str(FIX / "seed.py"), "--data", str(FIX / "data.jsonl"),
                     "--provider", "anthropic", "--model", "m", "--root", str(tmp_path / "s")])
    assert code == 2 and "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_data_and_search_are_exclusive():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["search", "--harness", "h.py", "--data", "a", "--search", "b",
                                       "--provider", "scripted", "--model", "m"])


def _base_args(tmp_path, data):
    return ["search", "--harness", str(FIX / "seed.py"), "--data", str(data),
            "--provider", "scripted", "--model", str(FIX / "base.jsonl"), "--root", str(tmp_path / "s"),
            "--iterations", "0", "--no-cache"]


def _one_line_error(capsys) -> str:
    err = capsys.readouterr().err
    assert err.startswith("error: ") and len(err.strip().splitlines()) == 1
    return err


def test_bad_metric_is_exit_2(tmp_path, capsys):
    assert cli.main(_base_args(tmp_path, FIX / "data.jsonl") + ["--metric", "nope"]) == 2
    assert "metric" in _one_line_error(capsys)


def test_missing_data_is_exit_2(tmp_path, capsys):
    assert cli.main(_base_args(tmp_path, tmp_path / "missing.jsonl")) == 2
    _one_line_error(capsys)


def test_empty_data_is_exit_2(tmp_path, capsys):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert cli.main(_base_args(tmp_path, empty)) == 2
    _one_line_error(capsys)


def test_resume_refuses_without_run_json(tmp_path, capsys):
    assert _smoke(tmp_path / "fresh", "--resume", "--iterations", "1") == 2
    assert "run.json" in capsys.readouterr().err


def test_resume_evaluates_only_the_new_iteration(tmp_path, capsys):
    root = tmp_path / "store"
    assert _smoke(root, "--iterations", "1") == 0
    capsys.readouterr()
    candidates = root / "candidates"
    before = {d.name: (d / "scores.json").stat().st_mtime_ns for d in candidates.iterdir()}
    assert set(before) == {"initial-0000", "iteration-0001-00"}
    assert _smoke(root, "--resume", "--iterations", "2") == 0
    out = json.loads(capsys.readouterr().out)
    after = {d.name: (d / "scores.json").stat().st_mtime_ns for d in candidates.iterdir()}
    assert set(after) == set(before) | {"iteration-0002-00"}
    assert all(after[name] == before[name] for name in before), "finished candidates re-evaluated"
    assert out["budget"]["harness_evaluations"] == 3
    assert out["verdicts"] == {"accepted": 3}


def test_force_resume_resets_the_attempt_counter(tmp_path, capsys):
    root = tmp_path / "store"
    assert _smoke(root, "--iterations", "0") == 0
    (root / ".resume_attempts").write_text("3", encoding="utf-8")
    assert _smoke(root, "--resume", "--iterations", "0") == 2
    assert "force-resume" in capsys.readouterr().err
    assert _smoke(root, "--force-resume", "--iterations", "0") == 0
    assert (root / ".resume_attempts").read_text(encoding="utf-8").strip() == "1"


def test_price_flag_feeds_billed_cost_separately_from_tokens(tmp_path, capsys):
    base, proposer = str(FIX / "base.jsonl"), str(FIX / "proposer.jsonl")
    assert _smoke(tmp_path / "store", "--iterations", "1",
                  "--price", f"scripted:{base}=1000000,0", "--price", f"scripted:{proposer}=0,0") == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cost_usd"] == pytest.approx(out["context_tokens"]["model"])
    run = json.loads((tmp_path / "store" / "run.json").read_text(encoding="utf-8"))
    assert run["budget"]["cost_usd"] == pytest.approx(out["cost_usd"])
    assert "META_HARNESS_PRICES" not in os.environ


def test_bad_price_is_exit_2(tmp_path, capsys):
    assert _smoke(tmp_path / "s", "--price", "nonsense") == 2
    assert "--price" in capsys.readouterr().err


# -- online protocol: train on a split disjoint from search and test -------------------------------

MEMORIZER = """class Harness:
    def __init__(self):
        self.seen = {}

    def run(self, task, model, trace):
        return {"output": self.seen.get(task["input"], "unknown")}

    def learn(self, task, result):
        self.seen[task["input"]] = task["label"]
"""


@pytest.fixture
def online_domain(monkeypatch):
    import types

    from meta_harness.commands import search as search_cmd

    domain = types.SimpleNamespace(NAME="online-test", PROMPT="", METRICS={}, SEED_DIR=None,
                                   PROTOCOL="online", adapt=lambda rows: [dict(r) for r in rows])
    monkeypatch.setattr(search_cmd, "get_domain", lambda name: domain)
    return domain


def _online(tmp_path, *extra):
    seed = tmp_path / "memorizer.py"
    seed.write_text(MEMORIZER, encoding="utf-8")
    return cli.main(["search", "--harness", str(seed), "--provider", "scripted",
                     "--model", str(FIX / "base.jsonl"), "--root", str(tmp_path / "s"),
                     "--iterations", "0", "--no-cache", *extra])


def test_online_memorizer_does_not_win_on_search(tmp_path, capsys, online_domain):
    assert _online(tmp_path, "--data", str(FIX / "data.jsonl")) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["test_tasks"] == 3 and out["train_tasks"] + out["search_tasks"] == 7
    assert out["train_tasks"] > 0 and out["search_tasks"] > 0
    score = json.loads((tmp_path / "s" / "candidates" / "initial-0000" / "scores.json")
                       .read_text(encoding="utf-8"))["score"]
    assert score < 1.0, "the memorizer was trained on the tasks it is scored on"


def test_online_splits_are_disjoint(tmp_path, capsys, online_domain):
    from meta_harness.commands import search as search_cmd

    captured = {}
    real = search_cmd.CandidateEvaluator

    def spy(*args, **kwargs):
        captured["train"] = kwargs.get("train_tasks")
        return real(*args, **kwargs)

    import _pytest.monkeypatch
    mp = _pytest.monkeypatch.MonkeyPatch()
    mp.setattr(search_cmd, "CandidateEvaluator", spy)
    try:
        assert _online(tmp_path, "--data", str(FIX / "data.jsonl")) == 0
    finally:
        mp.undo()
    manifest = json.loads((tmp_path / "s" / "run.json").read_text(encoding="utf-8"))
    test = json.loads((Path(str(tmp_path / "s") + "-test") / "test_results.json").read_text(encoding="utf-8"))
    train = {t["input"] for t in captured["train"]}
    search = {t["input"] for t in manifest["tasks"]}
    every = {json.loads(l)["input"] for l in (FIX / "data.jsonl").read_text(encoding="utf-8").splitlines()}
    assert train and search and not (train & search)
    assert len(every - train - search) == 3 and test["results"]


def test_online_overlapping_train_is_refused(tmp_path, capsys, online_domain):
    assert _online(tmp_path, "--search", str(FIX / "data.jsonl"), "--train", str(FIX / "data.jsonl")) == 2
    assert "overlap" in capsys.readouterr().err


def test_online_without_a_train_split_is_refused(tmp_path, capsys, online_domain):
    assert _online(tmp_path, "--search", str(FIX / "data.jsonl")) == 2
    assert "train split" in capsys.readouterr().err


def test_train_flag_is_rejected_for_plain_domains(tmp_path, capsys):
    assert _smoke(tmp_path / "s", "--train", str(FIX / "data.jsonl"), "--iterations", "0") == 2
    assert "online" in capsys.readouterr().err


def test_engine_errors_keep_their_traceback(tmp_path, monkeypatch):
    from meta_harness.commands import search as search_cmd

    def boom(self, *args, **kwargs):
        raise RuntimeError("engine failure")

    monkeypatch.setattr(search_cmd.SearchRunner, "run", boom)
    with pytest.raises(RuntimeError, match="engine failure"):
        _smoke(tmp_path / "s", "--iterations", "0")


def test_price_is_keyed_by_provider_and_model_with_bare_fallback():
    from meta_harness.commands.search import _price, parse_prices

    prices = parse_prices(["openai:m=1,2", "anthropic:m=3,4"])
    assert _price("openai", "m", prices).input == 1.0
    assert _price("anthropic", "m", prices).input == 3.0
    assert _price("unikey", "m", prices).input == 1.0, "bare model id is the fallback"
    assert _price("openai", "other", prices) is None
