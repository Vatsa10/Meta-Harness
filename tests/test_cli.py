import json
import sys
from pathlib import Path

from meta_harness import __main__ as cli
from meta_harness.commands import models as models_cmd

FIX = Path(__file__).parent / "fixtures" / "smoke"
STUB_AGENT = '''
import os
from pathlib import Path

out = Path(os.environ["META_HARNESS_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
(out / "candidate-00.py").write_text(
    "class Harness:\\n    def run(self, task, model, trace):\\n        return model(task['input'])\\n")
(out / "candidate-00.reasoning.md").write_text("echo the input through the model")
'''


def _tasks(tmp_path: Path) -> Path:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps({"input": text, "label": label}) for text, label in [
        ("an apple", "fruit"), ("a pear", "fruit"), ("a fast car", "vehicle"),
        ("a red truck", "vehicle"), ("a banana", "fruit"), ("a blue van", "vehicle"),
    ]), encoding="utf-8")
    return tasks


def test_search_end_to_end_with_a_stub_proposer_command(tmp_path: Path, capsys):
    agent = tmp_path / "agent.py"
    agent.write_text(STUB_AGENT, encoding="utf-8")
    code = cli.main([
        "search", "--harness", str(FIX / "seed.py"), "--data", str(_tasks(tmp_path)),
        "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
        "--root", str(tmp_path / "store"), "--iterations", "1", "--no-cache",
        "--proposer-command", f"{sys.executable} {agent}", "--test-fraction", "0.5",
    ])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frontier"]
    assert payload["search_tasks"] == 3 and payload["test_tasks"] == 3
    assert Path(payload["test_results"]).is_file()
    assert (tmp_path / "store" / "candidates" / "iteration-0001-00" / "proposer_reasoning.md").is_file()


def test_help_lists_search_not_run(capsys):
    import pytest
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    import re
    choices = re.search(r"\{([a-z_,-]+)\}", out.replace(" ", "")).group(1).split(",")
    assert "search" in choices and "run" not in choices


def test_models_subcommand_uses_list_models(monkeypatch, capsys):
    seen = {}

    def fake(provider, base_url=None, **_):
        seen.update(provider=provider, base_url=base_url)
        return ["claude-sonnet-4-6", "gpt-5.2"]

    monkeypatch.setattr(models_cmd, "list_models", fake)
    assert cli.main(["models", "--provider", "openai", "--base-url", "http://local/v1"]) == 0
    assert "gpt-5.2" in capsys.readouterr().out
    assert seen == {"provider": "openai", "base_url": "http://local/v1"}


def test_models_without_key_is_a_clean_error(monkeypatch, capsys):
    monkeypatch.delenv("UNIKEY_API_KEY", raising=False)
    assert cli.main(["models", "--provider", "unikey"]) == 2
    assert "UNIKEY_API_KEY" in capsys.readouterr().err


def test_explicit_test_file_is_used(tmp_path: Path):
    search = tmp_path / "s.jsonl"
    search.write_text(json.dumps({"input": "an apple", "label": "fruit"}), encoding="utf-8")
    held = tmp_path / "t.jsonl"
    held.write_text(json.dumps({"input": "a car", "label": "vehicle"}), encoding="utf-8")
    code = cli.main([
        "search", "--harness", str(FIX / "seed.py"), "--search", str(search), "--test", str(held),
        "--provider", "scripted", "--model", str(FIX / "base.jsonl"), "--root", str(tmp_path / "store"),
        "--iterations", "0", "--no-cache"])
    assert code == 0
    results = json.loads((Path(str(tmp_path / "store") + "-test") / "test_results.json").read_text(encoding="utf-8"))
    assert results["results"][0]["score"] == 1.0


def test_demo_still_runs(tmp_path: Path, capsys):
    assert cli.main(["demo", "--iterations", "1", "--root", str(tmp_path / "demo")]) == 0
    assert json.loads(capsys.readouterr().out)
    assert (tmp_path / "demo" / "candidates" / "iteration-0001-00" / "proposer_log.jsonl").is_file()
