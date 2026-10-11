import json
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness import experience_query as q

SECRET = "sk-abcdefgh12345678"


def _cand(root: Path, cid: str, score: float, cost: float, source: str, valid: bool = True, receipt=None,
          reasoning: str = "") -> Path:
    d = root / "candidates" / cid
    d.mkdir(parents=True)
    (d / "harness.py").write_text(source, encoding="utf-8")
    (d / "scores.json").write_text(json.dumps({"candidate_id": cid, "score": score, "context_cost": cost,
                                                "valid": valid}), encoding="utf-8")
    if receipt is not None:
        (d / "receipt.json").write_text(json.dumps({"candidate_id": cid, **receipt}), encoding="utf-8")
    if reasoning:
        (d / "proposer_reasoning.md").write_text(reasoning, encoding="utf-8")
    return d


def _log(path: Path, rows: list[dict], torn: bool = False) -> None:
    text = "".join(json.dumps(r) + "\n" for r in rows)
    if torn:
        text += '{"role": "tool", "tool": "re'
    path.write_text(text, encoding="utf-8")


def _read(path: str, ok: bool = True) -> dict:
    return {"role": "tool", "tool": "read", "ok": ok, "path": path, "arguments": {"path": path}}


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _cand(root, "seed", 0.5, 10, "a = 1\nb = 2\n")
    _cand(root, "iteration-0001-00", 0.9, 30, "a = 1\nb = 3\n")
    _cand(root, "iteration-0002-00", 0.9, 20, "a = 1\nb = 4\n")
    _cand(root, "bad", 1.0, 1, "x\n", valid=False)
    _log(root / "candidates" / "iteration-0001-00" / "proposer_log.jsonl", [
        {"role": "assistant", "tool": None},
        _read("candidates/seed/harness.py"), _read("candidates/seed/harness.py"),
        _read("candidates/seed/scores.json"), _read("candidates/seed/traces.jsonl"),
        _read("../secret/test.json", ok=False),
        {"role": "tool", "tool": "grep", "ok": True}, {"role": "tool", "tool": "list", "ok": True},
        {"role": "tool", "tool": "list", "ok": True}], torn=True)
    _log(root / "candidates" / "iteration-0002-00" / "proposer_log.jsonl", [
        _read("candidates/seed/proposer_reasoning.md"), _read("run.json"), _read("notes.txt")])
    return root


def test_top_k_order_valid_only_ties_by_cost(run):
    ids = [r["candidate_id"] for r in q.top_k(run, 5)]
    assert ids == ["iteration-0002-00", "iteration-0001-00", "seed"]
    assert [r["candidate_id"] for r in q.top_k(run, 1)] == ["iteration-0002-00"]
    assert q.top_k(run, 2, by="cost")[0]["candidate_id"] == "seed"


def test_diff_candidates_has_changes_and_deltas(run):
    text = q.diff_candidates(run, "seed", "iteration-0001-00")
    assert "-b = 2" in text and "+b = 3" in text
    assert "delta=+0.4" in text and "delta=+20" in text
    with pytest.raises(ValueError):
        q.diff_candidates(run, "../x", "seed")


def test_read_stats_counts_only_ok_reads(run):
    stats = q.read_stats(run)
    first, second = stats["iterations"]
    assert (first["iteration"], first["files_read"], first["distinct_files"]) == (1, 4, 3)
    assert first["grep_calls"] == 1 and first["list_calls"] == 2 and first["refused_calls"] == 1
    assert first["categories"]["source"] == 2 and first["categories"]["scores"] == 1
    assert first["categories"]["traces"] == 1
    assert (second["files_read"], second["categories"]["reasoning"], second["categories"]["run_meta"],
            second["categories"]["other"]) == (3, 1, 1, 1)
    for entry in stats["iterations"]:
        assert sum(entry["categories"].values()) == entry["files_read"]
    assert stats["medians"]["files_read"] == 3.5 and stats["medians"]["grep_calls"] == 0.5


def test_categorise():
    assert q.categorise("candidates/a/harness.py") == "source"
    assert q.categorise("candidates\\a\\scores.json") == "scores"
    assert q.categorise("candidates/a/traces-1.jsonl") == "traces"
    assert q.categorise("candidates/a/proposer_reasoning.md") == "reasoning"
    assert q.categorise("frontier.json") == "run_meta"
    assert set(q.CATEGORIES) >= {q.categorise("whatever.bin")}


def test_import_candidates_copies_sources(run, tmp_path):
    seeds = tmp_path / "seeds"
    out = q.import_candidates(run, seeds, 2)
    assert len(out) == 2
    assert out[0].read_text(encoding="utf-8") == "a = 1\nb = 4\n"
    assert out[1].read_text(encoding="utf-8") == "a = 1\nb = 3\n"


def test_diff_runs(run, tmp_path):
    other = tmp_path / "other"
    _cand(other, "c", 0.7, 5, "z\n")
    (other / "run.json").write_text(json.dumps({"budget": {"model_calls": 7}}), encoding="utf-8")
    (run / "frontier.json").write_text(json.dumps([{"candidate_id": "seed", "score": 0.5}]), encoding="utf-8")
    result = q.diff_runs(run, other)
    assert result["a"]["best_score"] == 0.9 and result["b"]["best_score"] == 0.7
    assert result["budget"]["model_calls"] == {"a": None, "b": 7}
    assert result["a"]["frontier"][0]["candidate_id"] == "seed"


def _receipts_run(tmp_path: Path) -> Path:
    root = tmp_path / "rr"
    for i, verdict in enumerate(q.VERDICTS):
        _cand(root, f"iteration-{i // 4 + 1:04d}-{i % 4:02d}", 0.1 * i, 1, "x\n",
              receipt={"verdict": verdict, "parents_read": ["seed"], "fingerprint": f"v1:{i}",
                       "tool_outcomes": {"dispatched": 3, "denied": 1, "failed": i, "aborted": 0},
                       "evaluation": {} if verdict == "accepted" else None},
              reasoning=f"because {SECRET} works" if i == 0 else "")
    (root / "candidates" / "iteration-0002-02" / "receipt.json").write_text('{"verdict": "acc', encoding="utf-8")
    return root


def test_report_counts_one_of_each_verdict(tmp_path):
    root = _receipts_run(tmp_path)
    (root / "candidates" / "iteration-0002-02" / "receipt.json").write_text(
        json.dumps({"verdict": "incomplete"}), encoding="utf-8")
    rep = q.report(root)
    assert rep["receipts"] == 7 and all(rep["verdicts"][v] == 1 for v in q.VERDICTS)
    assert rep["by_iteration"]["0001"]["accepted"] == 1 and rep["by_iteration"]["0002"]["incomplete"] == 1
    assert rep["tool_outcomes_by_verdict"]["failed_eval"]["failed"] == 5
    assert rep["tool_outcomes_by_verdict"]["accepted"]["denied"] == 1
    assert rep["tool_outcomes_by_verdict"]["accepted"]["mean_score"] == 0.0
    assert rep["stages"]["evaluated"] == 1


def test_export_one_case_per_receipt_without_secrets(tmp_path):
    root = _receipts_run(tmp_path)  # one torn receipt.json is skipped
    out = tmp_path / "cases.jsonl"
    assert q.export_cases(root, out) == 6
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6 and "sk-" not in "".join(lines)
    case = json.loads(lines[0])
    assert {"id", "verdict", "parents", "fingerprint", "scores", "rationale"} <= set(case)
    assert case["parents"] == ["seed"] and "[redacted]" in case["rationale"]
    with pytest.raises(ValueError):
        q.export_cases(root, root / "cases.jsonl")


def test_cli(run, tmp_path, capsys):
    assert cli.main(["experience", "top", str(run), "-k", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["candidate_id"] == "iteration-0002-00"
    assert cli.main(["experience", "import", str(run), str(tmp_path / "s"), "-k", "1"]) == 0
    assert "--harness" in capsys.readouterr().out
    assert cli.main(["experience", "reads", str(run)]) == 0
    assert cli.main(["experience", "diff", str(run), "seed", "nope"]) == 2
