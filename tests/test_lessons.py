import hashlib
import json
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness import lessons

SECRET = "sk-abcdefgh12345678"
BASE = "def pick(x):\n    return x\n\ndef other():\n    return 1\n"
BETTER = "def pick(x):\n    return sorted(x)[0]\n\ndef other():\n    return 1\n"


def _cand(root: Path, cid: str, score: float, source: str, parent=None, repeats: int = 1) -> None:
    d = root / "candidates" / cid
    d.mkdir(parents=True)
    (d / "harness.py").write_text(source, encoding="utf-8")
    (d / "scores.json").write_text(json.dumps({"candidate_id": cid, "score": score, "context_cost": 1,
                                                "valid": True, "repeats": repeats}), encoding="utf-8")
    (d / "receipt.json").write_text(json.dumps({"candidate_id": cid, "verdict": "accepted",
                                                "parents_read": [parent] if parent else []}), encoding="utf-8")


@pytest.fixture
def run(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    _cand(root, "seed", 0.4, BASE)
    _cand(root, "iteration-0001-00", 0.8, BETTER, parent="seed", repeats=3)
    _cand(root, "iteration-0002-00", 0.3, BASE + "\nLIMIT = 3\n", parent="seed")
    (root / "frontier.json").write_text(json.dumps([{"candidate_id": "seed"}, {"candidate_id": "iteration-0001-00"}]),
                                        encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def lib(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "lib"
    monkeypatch.setenv(lessons.LIBRARY_ENV, str(path))
    return path


def test_distil_produces_record_for_frontier_candidate(run):
    records = lessons.distil(run, created="2026-10-18T00:00:00+00:00")
    assert len(records) == 1  # seed has no parent; iteration-0002 is off the frontier
    rec = records[0]
    assert set(rec) == set(lessons.FIELDS)
    assert rec["cue"] == "changed pick" and rec["evidence"] == ["iteration-0001-00", "seed"]
    assert "0.4 -> 0.8" in rec["lesson"] and "improved" in rec["lesson"]
    assert rec["confidence"] == pytest.approx(0.75) and rec["source_run"] == "run"
    assert lessons.changed_components(BASE, BASE + "\nLIMIT = 3\n") == ["LIMIT"]


def test_confidence_grows_with_delta_and_n():
    assert lessons._confidence(0.1, 1) < lessons._confidence(0.2, 1) < lessons._confidence(0.2, 5)
    assert 0 <= lessons._confidence(5, 100) <= 1


def test_no_frontier_falls_back_to_all_valid_candidates(run):
    (run / "frontier.json").unlink()
    cues = sorted(r["cue"] for r in lessons.distil(run))
    assert cues == ["changed LIMIT", "changed pick"]


def test_add_is_idempotent_and_query_filters(run, lib):
    assert lessons.distil_to_library(run) == (1, 1)
    assert lessons.distil_to_library(run) == (1, 0)
    assert len(lessons.load()) == 1
    assert [r["cue"] for r in lessons.query("PICK")] == ["changed pick"]
    assert lessons.query("nonexistentword") == []
    assert len(lessons.query(None)) == 1


def test_redaction_of_stored_text(run, lib):
    lessons.add([{"cue": f"changed {SECRET}", "lesson": f"api_key={SECRET} helped", "evidence": [SECRET],
                  "confidence": 0.5, "source_run": "r", "created": "t"}])
    blob = (lib / lessons.LIBRARY_FILE).read_text(encoding="utf-8")
    assert "sk-abcdefgh" not in blob and "[redacted]" in blob
    assert "sk-abcdefgh" not in json.dumps(lessons.query("helped"))


def test_cli_and_run_root_unchanged(run, lib, capsys):
    def tree():
        return {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(run.rglob("*")) if p.is_file()}
    before = tree()
    assert cli.main(["experience", "distil", str(run)]) == 0
    assert "derived 1 lessons, stored 1 new" in capsys.readouterr().out
    assert cli.main(["experience", "lessons", "--query", "pick"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["evidence"] == ["iteration-0001-00", "seed"]
    assert cli.main(["experience", "lessons", "--query", "zzz"]) == 0
    assert json.loads(capsys.readouterr().out) == []
    assert cli.main(["experience", "distil", str(run), "--library", str(run / "lib")]) == 2
    assert cli.main(["experience", "distil", str(run.parent / "missing")]) == 2
    assert tree() == before
