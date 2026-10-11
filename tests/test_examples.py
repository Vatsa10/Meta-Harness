import re
from pathlib import Path

from meta_harness import __main__ as cli

ROOT = Path(__file__).resolve().parent.parent


def test_example_runs_offline(tmp_path, capsys):
    ex = ROOT / "examples" / "sentiment"
    root = tmp_path / "s"
    assert cli.main(["search", "--harness", str(ex / "harness.py"), "--data", str(ex / "data.jsonl"),
                     "--metric", f"{ex / 'metric.py'}:score", "--provider", "scripted",
                     "--model", str(ex / "base_script.jsonl"), "--proposer-provider", "scripted",
                     "--proposer-model", str(ex / "proposer_script.jsonl"), "--iterations", "1",
                     "--root", str(root), "--no-cache"]) == 0
    import json
    seed = json.loads((root / "candidates" / "initial-0000" / "scores.json").read_text(encoding="utf-8"))
    best = json.loads((root / "candidates" / "iteration-0001-00" / "scores.json").read_text(encoding="utf-8"))
    assert seed["score"] == 0.0 and best["score"] == 1.0, "the example must show the proposer's fix"
    assert (tmp_path / "s-test" / "test_results.json").is_file()


def test_example_data_is_thirty_rows_three_labels():
    import json
    rows = [json.loads(l) for l in (ROOT / "examples" / "sentiment" / "data.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(rows) == 30 and {r["label"] for r in rows} == {"positive", "negative", "neutral"}


def test_readme_commands_match_the_cli():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    parser = cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    commands = re.findall(r"^\s*meta-harness ([\w-]+)", readme, re.M)
    assert commands
    for name in commands:
        assert name in choices, f"README documents unknown command {name!r}"
    assert "claude-cli" not in readme and "/plugin" not in readme
    assert "Claude Code" not in readme


def test_readme_results_cite_only_produced_runs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = readme.split("## Results", 1)[1].split("\n## ", 1)[0]
    for number in re.findall(r"\b0\.\d{2,}\b|\b\d{2}\.\d%?", results):
        assert "run.json" in results, f"result {number} has no run.json provenance"
