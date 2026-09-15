import json
from pathlib import Path

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner

HARNESS = '''
class Harness:
    def run(self, task, model, trace):
        return task["input"]
'''


def _runner(tmp_path: Path) -> SearchRunner:
    config = SearchConfig(tmp_path / "store", iterations=0)
    return SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])


def test_test_results_land_outside_the_experience_root(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    runner = _runner(tmp_path)
    frontier = runner.run([source], [{"input": "a", "label": "a"}], test_tasks=[{"input": "b", "label": "b"}])
    assert frontier

    test_root = Path(runner.config.test_root)
    payload = json.loads((test_root / "test_results.json").read_text())
    assert payload["results"][0]["split"] == "test"
    assert payload["results"][0]["score"] == 1.0

    # Nothing under the experience root may mention the test split.
    for path in runner.experience.root.rglob("*"):
        if path.is_file():
            assert '"split": "test"' not in path.read_text(encoding="utf-8", errors="ignore")


def test_no_test_tasks_means_no_test_file(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    runner = _runner(tmp_path)
    runner.run([source], [{"input": "a", "label": "a"}])
    assert not (Path(runner.config.test_root) / "test_results.json").exists()


def test_run_manifest_records_only_search_tasks(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    runner = _runner(tmp_path)
    runner.run([source], [{"input": "a", "label": "a"}], test_tasks=[{"input": "b", "label": "b"}])
    manifest = json.loads((runner.experience.root / "run.json").read_text())
    assert manifest["task_count"] == 1
    assert manifest["tasks"] == [{"input": "a", "label": "a"}]
    assert manifest["test_task_count"] == 1
    assert '"input": "b"' not in json.dumps(manifest["tasks"])
