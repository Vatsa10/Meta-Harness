import json
import sys
from pathlib import Path

from meta_harness import __main__ as cli


def test_run_end_to_end_with_a_stub_proposer(tmp_path: Path, monkeypatch, capsys):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"input": text, "label": label}) for text, label in [
        ("an apple", "fruit"), ("a pear", "fruit"), ("a fast car", "vehicle"),
        ("a red truck", "vehicle"), ("a banana", "fruit"), ("a blue van", "vehicle"),
    ]), encoding="utf-8")

    agent = tmp_path / "agent.py"
    agent.write_text('''
import os
from pathlib import Path

out = Path(os.environ["META_HARNESS_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
(out / "candidate-00.py").write_text(
    "class Harness:\\n    def run(self, task, model, trace):\\n        return model(task['input'])\\n")
(out / "candidate-00.reasoning.md").write_text("echo the input through the model")
''', encoding="utf-8")

    fruit = ("apple", "pear", "banana")
    monkeypatch.setattr(cli, "model_from_environment", lambda *a, **k: (
        lambda prompt, **_: "fruit" if any(w in prompt for w in fruit) else "vehicle"))

    code = cli.main([
        "run", "--tasks", str(tasks), "--provider", "unikey", "--model", "gpt-5.2",
        "--root", str(tmp_path / "store"), "--iterations", "1",
        "--proposer-command", f"{sys.executable} {agent}",
        "--search-fraction", "0.5",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frontier"]
    assert Path(payload["test_results"]).is_file()


def test_run_defaults_to_bundled_baselines():
    parsed = cli.build_parser().parse_args([
        "run", "--tasks", "t.jsonl", "--provider", "unikey", "--model", "m"])
    assert parsed.baseline is None
    assert [p.name for p in cli.default_baselines("classification")] == [
        "zero_shot.py", "few_shot.py", "ace.py", "mce.py"]
    assert [p.name for p in cli.default_baselines("math")] == [
        "math_zero_shot.py", "math_bm25.py"]
    assert [p.name for p in cli.default_baselines("terminal")] == ["terminal_basic.py"]


def test_models_subcommand_lists_ids(monkeypatch, capsys):
    monkeypatch.setattr(cli, "list_unikey_models", lambda **kw: ["claude-sonnet-4-6", "gpt-5.2"])
    assert cli.main(["models", "--provider", "unikey"]) == 0
    assert "gpt-5.2" in capsys.readouterr().out


def test_explicit_test_tasks_file_is_used(tmp_path: Path, monkeypatch):
    search = tmp_path / "s.jsonl"
    search.write_text(json.dumps({"input": "a", "label": "a"}), encoding="utf-8")
    held = tmp_path / "t.jsonl"
    held.write_text(json.dumps({"input": "b", "label": "b"}), encoding="utf-8")
    monkeypatch.setattr(cli, "model_from_environment", lambda *a, **k: (lambda prompt, **_: "a"))
    code = cli.main([
        "run", "--tasks", str(search), "--test-tasks", str(held),
        "--provider", "unikey", "--model", "m", "--root", str(tmp_path / "store"),
        "--iterations", "0", "--baseline", str(cli.BASELINE_ROOT / "zero_shot.py")])
    assert code == 0
    assert (Path(str(tmp_path / "store") + "-test") / "test_results.json").is_file()


def test_demo_still_runs(tmp_path: Path, capsys):
    assert cli.main(["demo", "--iterations", "1", "--root", str(tmp_path / "demo")]) == 0
    assert json.loads(capsys.readouterr().out)
