from pathlib import Path

from meta_harness.core import CandidateEvaluator, FilesystemExperience, SearchConfig, SearchRunner

HARNESS = '''
class Harness:
    def run(self, task, model, trace):
        return task["label"]
'''


def test_add_source_writes_notes(tmp_path: Path):
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    experience = FilesystemExperience(tmp_path / "store")
    candidate_id = experience.add_source(source, "c0", notes={"proposer_reasoning.md": "because X"})
    assert (experience.candidate_dir(candidate_id) / "proposer_reasoning.md").read_text() == "because X"


def test_runner_copies_reasoning_sidecar(tmp_path: Path):
    proposal = tmp_path / "prop.py"
    proposal.write_text(HARNESS)
    (tmp_path / "prop.reasoning.md").write_text("isolated the prompt change from the retrieval change")

    def proposer(experience, iteration, count):
        return [proposal]

    config = SearchConfig(tmp_path / "store", iterations=1)
    runner = SearchRunner(config, CandidateEvaluator(lambda p, **_: p), proposer)
    runner.run([], [{"input": "a", "label": "a"}])
    directory = runner.experience.candidate_dir("iteration-0001-00")
    assert "isolated the prompt change" in (directory / "proposer_reasoning.md").read_text()
