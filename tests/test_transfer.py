import json
from pathlib import Path

from meta_harness import core
from meta_harness.protocol import exact
from meta_harness.transfer import cell_name, parse_models, slug, transfer

TASKS = [{"input": "q1", "label": "yes"}, {"input": "q2", "label": "yes"}, {"input": "q3", "label": "no"}]

ECHO = "class H:\n    def run(self, task, model):\n        return {'output': model(task['input'])}\n\ndef build_harness():\n    return H()\n"
FLIP = ("class H:\n    def run(self, task, model):\n        return {'output': 'yes' if model(task['input']) == 'no' else 'no'}\n\n"
        "def build_harness():\n    return H()\n")
MEMO = ("class H:\n    def __init__(self):\n        self.memory = {}\n"
        "    def run(self, task, model):\n        return {'output': self.memory.get(task['input']) or model(task['input'])}\n"
        "    def learn(self, task, result):\n        self.memory[task['input']] = task['label']\n\n"
        "def build_harness():\n    return H()\n")


def script(path: Path, reply: str) -> str:
    path.write_text(json.dumps({"default": reply}) + "\n", encoding="utf-8")
    return str(path)


def harness(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def rows_by_cell(report):
    return {(r["harness"], Path(r["model"]).stem): r for r in report["rows"]}


def test_two_harnesses_by_two_models_give_four_rows(tmp_path):
    yes, no = script(tmp_path / "yes.jsonl", "yes"), script(tmp_path / "no.jsonl", "no")
    a, b = harness(tmp_path / "echo.py", ECHO), harness(tmp_path / "flip.py", FLIP)
    report = transfer([a, b], TASKS, [("scripted", yes), ("scripted", no)], exact, tmp_path / "out")
    cells = rows_by_cell(report)
    assert len(report["rows"]) == 4
    assert cells[("echo", "yes")]["score"] == 2 / 3
    assert cells[("echo", "no")]["score"] == 1 / 3
    assert cells[("flip", "yes")]["score"] == 1 / 3
    assert cells[("flip", "no")]["score"] == 2 / 3
    saved = json.loads((tmp_path / "out" / "transfer.json").read_text(encoding="utf-8"))
    assert saved["rows"] == report["rows"]
    row = cells[("echo", "yes")]
    assert {"harness", "provider", "model", "score", "score_std", "context_cost", "n"} <= set(row)
    assert row["n"] == 3 and row["context_cost"] > 0


def test_outcomes_are_per_task_binary_with_cost(tmp_path, monkeypatch):
    monkeypatch.setenv("META_HARNESS_PRICES", json.dumps({"x": {"input": 1.0, "output": 1.0}}))
    yes = script(tmp_path / "yes.jsonl", "yes")
    report = transfer([harness(tmp_path / "echo.py", ECHO)], TASKS, [("scripted", yes)], exact, tmp_path / "out")
    outcomes = report["rows"][0]["outcomes"]
    assert [o["pass"] for o in outcomes] == [True, True, False]
    assert [o["task"] for o in outcomes] == [0, 1, 2]
    assert all(o["prompt_tokens"] > 0 and "cost_usd" in o for o in outcomes)
    assert outcomes[0]["cost_usd"] is None  # unknown price is never guessed


def test_unbuildable_model_yields_error_row_and_run_continues(tmp_path):
    yes = script(tmp_path / "yes.jsonl", "yes")
    report = transfer([harness(tmp_path / "echo.py", ECHO)], TASKS,
                      [("nonesuch", "m"), ("scripted", str(tmp_path / "missing.jsonl")), ("scripted", yes)],
                      exact, tmp_path / "out")
    bad1, bad2, good = report["rows"]
    assert "error" in bad1 and bad1["score"] is None
    assert "error" in bad2
    assert good["score"] == 2 / 3 and "error" not in good


def test_slugs_are_windows_safe(tmp_path):
    for provider, model in [("unikey", "gpt-5-mini"), ("groq", "meta/llama:3?"), ("scripted", 'C:\a\b "x".jsonl')]:
        name = cell_name(provider, model)
        assert not set(name) & set(':/\\*?"<>|'), name
    assert slug("gpt-5-mini") == "gpt-5-mini"
    assert slug("a:b") != slug("a/b")
    yes = script(tmp_path / "yes.jsonl", "yes")
    transfer([harness(tmp_path / "echo.py", ECHO)], TASKS[:1], [("scripted", yes)], exact, tmp_path / "out")
    assert (tmp_path / "out" / "echo" / cell_name("scripted", yes) / "trace-0.jsonl").is_file()


def test_memory_does_not_leak_between_models(tmp_path):
    a, b = script(tmp_path / "a.jsonl", "wrong"), script(tmp_path / "b.jsonl", "wrong")
    tasks = [{"input": "q", "label": "x"}, {"input": "q", "label": "x"}]
    report = transfer([harness(tmp_path / "memo.py", MEMO)], tasks, [("scripted", a), ("scripted", b)],
                      exact, tmp_path / "out")
    first, second = report["rows"]
    assert first["score"] == second["score"] == 0.5
    assert [o["pass"] for o in second["outcomes"]] == [False, True]


def test_parse_models():
    assert parse_models("unikey:gpt-5-mini, groq:llama-3.3-70b-versatile") == [
        ("unikey", "gpt-5-mini"), ("groq", "llama-3.3-70b-versatile")]


def test_cli_prints_table(tmp_path, capsys):
    from meta_harness.__main__ import main
    yes = script(tmp_path / "yes.jsonl", "yes")
    data = tmp_path / "t.jsonl"
    data.write_text("\n".join(json.dumps(t) for t in TASKS) + "\n", encoding="utf-8")
    h = harness(tmp_path / "echo.py", ECHO)
    code = main(["transfer", "--harness", str(h), "--data", str(data), "--metric", "exact",
                 "--models", f"scripted:{yes},nonesuch:m", "--root", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert code == 0 and "echo" in out and "0.667" in out and "ERR" in out
