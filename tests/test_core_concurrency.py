import time
from pathlib import Path

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner

# Each repeat re-imports the candidate module, so per-run state must live on disk
# for the harness to behave differently across repeats.
FLAKY = '''
import os
from pathlib import Path

COUNTER = Path(os.environ["META_HARNESS_TEST_COUNTER"])


class Harness:
    def __init__(self):
        previous = int(COUNTER.read_text()) if COUNTER.is_file() else 0
        COUNTER.write_text(str(previous + 1))
        self.run_index = previous

    def run(self, task, model, trace):
        return task["label"] if self.run_index % 2 == 0 else "wrong"
'''

SLEEPER = '''
import time


class Harness:
    def run(self, task, model, trace):
        time.sleep(0.6)
        return task["label"]
'''


def test_repeats_average_and_record_spread(tmp_path: Path, monkeypatch):
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("META_HARNESS_TEST_COUNTER", str(counter))
    source = tmp_path / "flaky.py"
    source.write_text(FLAKY)
    config = SearchConfig(tmp_path / "store", iterations=0, repeats=2)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])
    runner.run([source], [{"input": "a", "label": "a"}])
    result = runner.experience.results()[0]
    assert result.repeats == 2
    assert result.score == 0.5
    assert result.score_std > 0
    directory = runner.experience.candidate_dir(result.candidate_id)
    assert (directory / "traces.jsonl").exists()
    assert (directory / "traces-1.jsonl").exists()


def test_candidates_evaluate_concurrently(tmp_path: Path):
    sources = []
    for index in range(4):
        path = tmp_path / f"s{index}.py"
        path.write_text(SLEEPER)
        sources.append(path)
    config = SearchConfig(tmp_path / "store", iterations=0, max_workers=4)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])

    started = time.perf_counter()
    runner.run(sources, [{"input": "a", "label": "a"}])
    elapsed = time.perf_counter() - started

    # 4 candidates x 0.6s of sleeping is 2.4s serial; concurrent must beat that clearly
    # even with subprocess validation overhead counted in.
    assert elapsed < 2.4
    assert len(runner.experience.results()) == 4


def test_results_are_tagged_with_split(tmp_path: Path):
    source = tmp_path / "ok.py"
    source.write_text('''
class Harness:
    def run(self, task, model, trace):
        return task["label"]
''')
    config = SearchConfig(tmp_path / "store", iterations=0)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), lambda *_: [])
    runner.run([source], [{"input": "a", "label": "a"}])
    assert runner.experience.results()[0].split == "search"
