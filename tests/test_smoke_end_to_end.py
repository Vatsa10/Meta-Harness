import json
import sys
from pathlib import Path

from meta_harness import __main__ as cli

FIX = Path(__file__).parent / "fixtures" / "smoke"

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


def test_full_loop_produces_frontier_traces_reasoning_and_test_results(tmp_path: Path, capsys):
    tasks = tmp_path / "tasks.jsonl"
    rows = [("an apple", "fruit"), ("a pear", "fruit"), ("a plum", "fruit"),
            ("a fast car", "vehicle"), ("a red truck", "vehicle"), ("a blue van", "vehicle")]
    tasks.write_text("\n".join(json.dumps({"input": t, "label": l, "labels": ["fruit", "vehicle"]})
                               for t, l in rows), encoding="utf-8")
    agent = tmp_path / "agent.py"
    agent.write_text(AGENT, encoding="utf-8")
    second_seed = tmp_path / "seed2.py"
    second_seed.write_text("class Harness:\n    def run(self, task, model, trace):\n"
                           "        return model('label primed: ' + task['input'])\n", encoding="utf-8")

    root = tmp_path / "store"
    assert cli.main([
        "search", "--harness", str(FIX / "seed.py"), "--harness", str(second_seed),
        "--data", str(tasks), "--provider", "scripted", "--model", str(FIX / "base.jsonl"),
        "--root", str(root), "--iterations", "1", "--candidates", "1", "--repeats", "2",
        "--max-workers", "2", "--test-fraction", "0.5", "--no-cache",
        "--proposer-command", f"{sys.executable} {agent}",
    ]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["frontier"], "search produced no frontier"
    assert payload["frontier"][0]["candidate_id"] != "initial-0001", "the primed seed scores 0"

    candidates = sorted((root / "candidates").iterdir())
    assert len(candidates) == 3, "two seeds plus one proposal expected"
    proposed = root / "candidates" / "iteration-0001-00"
    assert (proposed / "proposer_reasoning.md").is_file()
    assert (proposed / "traces.jsonl").is_file()
    assert (proposed / "traces-1.jsonl").is_file()

    scores = json.loads((proposed / "scores.json").read_text(encoding="utf-8"))
    assert scores["repeats"] == 2
    assert scores["context_cost"] > 0, "context cost must be measured in tokens"

    test_results = json.loads(Path(payload["test_results"]).read_text(encoding="utf-8"))
    assert all(item["split"] == "test" for item in test_results["results"])
    assert payload["verdicts"] == {"accepted": 3}
