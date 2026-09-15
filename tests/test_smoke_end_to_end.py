import json
import sys
from pathlib import Path

from meta_harness import __main__ as cli

AGENT = '''
import json
import os
from pathlib import Path

root = Path(os.environ["META_HARNESS_ROOT"])
out = Path(os.environ["META_HARNESS_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)

# The proposer must actually be able to read prior experience.
scores = sorted((root / "candidates").glob("*/scores.json"))
assert scores, "no prior candidates visible to the proposer"
best = max(json.loads(p.read_text())["score"] for p in scores)
traces = sorted((root / "candidates").glob("*/traces.jsonl"))
assert traces, "no execution traces visible to the proposer"

(out / "candidate-00.py").write_text(
    "class Harness:\\n"
    "    def run(self, task, model, trace):\\n"
    "        labels = task.get('labels') or ['fruit', 'vehicle']\\n"
    "        reply = model('labels: ' + ', '.join(labels) + ' input: ' + task['input'])\\n"
    "        for label in labels:\\n"
    "            if label in reply:\\n"
    "                return label\\n"
    "        return labels[0]\\n")
(out / "candidate-00.reasoning.md").write_text(f"best prior score was {best}; forcing a declared label")
'''


def test_full_loop_produces_frontier_traces_reasoning_and_test_results(tmp_path: Path, monkeypatch, capsys):
    tasks = tmp_path / "tasks.jsonl"
    rows = [("an apple", "fruit"), ("a pear", "fruit"), ("a plum", "fruit"),
            ("a fast car", "vehicle"), ("a red truck", "vehicle"), ("a blue van", "vehicle")]
    tasks.write_text("\n".join(json.dumps({"input": t, "label": l, "labels": ["fruit", "vehicle"]})
                               for t, l in rows), encoding="utf-8")

    agent = tmp_path / "agent.py"
    agent.write_text(AGENT, encoding="utf-8")

    fruit_words = ("apple", "pear", "plum", "banana")
    monkeypatch.setattr(cli, "model_from_environment", lambda *a, **k: (
        lambda prompt, **_: "fruit" if any(w in prompt for w in fruit_words) else "vehicle"))

    root = tmp_path / "store"
    assert cli.main([
        "run", "--tasks", str(tasks), "--provider", "unikey", "--model", "gpt-5.2",
        "--root", str(root), "--iterations", "1", "--candidates", "1", "--repeats", "2",
        "--max-workers", "2", "--search-fraction", "0.5",
        "--proposer-command", f"{sys.executable} {agent}",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["frontier"], "search produced no frontier"

    candidates = sorted((root / "candidates").iterdir())
    assert len(candidates) >= 5, "four baselines plus one proposal expected"
    proposed = root / "candidates" / "iteration-0001-00"
    assert (proposed / "proposer_reasoning.md").is_file()
    assert (proposed / "traces.jsonl").is_file()
    assert (proposed / "traces-1.jsonl").is_file()

    scores = json.loads((proposed / "scores.json").read_text())
    assert scores["repeats"] == 2
    assert scores["context_cost"] > 0, "context cost must be measured in tokens"

    test_results = json.loads(Path(payload["test_results"]).read_text())
    assert all(item["split"] == "test" for item in test_results["results"])
