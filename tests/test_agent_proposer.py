import json
import sys
from pathlib import Path

from meta_harness.agent_proposer import SKILL_PATH, ClaudeCodeProposer, build_view
from meta_harness.core import EvaluationResult, FilesystemExperience

HARNESS = "class Harness:\n    def run(self, task, model, trace):\n        return 'x'\n"


def _experience(tmp_path: Path) -> FilesystemExperience:
    experience = FilesystemExperience(tmp_path / "store")
    source = tmp_path / "h.py"
    source.write_text(HARNESS)
    candidate_id = experience.add_source(source, "initial-0000")
    experience.write_result(EvaluationResult(candidate_id, 0.5, context_cost=100.0))
    directory = experience.candidate_dir(candidate_id)
    (directory / "traces.jsonl").write_text(json.dumps({"event": "model_call"}) + "\n", encoding="utf-8")
    return experience


def test_skill_file_is_bundled_and_describes_the_contract():
    text = SKILL_PATH.read_text(encoding="utf-8")
    assert "META_HARNESS_OUTPUT" in text
    assert "run(self, task, model, trace)" in text
    assert ".reasoning.md" in text


def test_full_view_returns_the_root_itself(tmp_path: Path):
    experience = _experience(tmp_path)
    view = build_view(experience, "full", tmp_path / "view")
    assert view == experience.root


def test_scores_view_hides_traces(tmp_path: Path):
    experience = _experience(tmp_path)
    view = build_view(experience, "scores", tmp_path / "view")
    candidate = next((view / "candidates").iterdir())
    assert (candidate / "harness.py").exists()
    assert (candidate / "scores.json").exists()
    assert not (candidate / "traces.jsonl").exists()


def test_summary_view_adds_summaries_but_still_hides_traces(tmp_path: Path):
    experience = _experience(tmp_path)
    view = build_view(experience, "summary", tmp_path / "view",
                      summarizer=lambda prompt, **_: "it mispredicted rare labels")
    candidate = next((view / "candidates").iterdir())
    assert not (candidate / "traces.jsonl").exists()
    assert "mispredicted" in (candidate / "summary.md").read_text()


def test_proposer_invokes_the_binary_and_collects_candidates(tmp_path: Path):
    experience = _experience(tmp_path)
    fake = tmp_path / "fake_agent.py"
    fake.write_text('''
import os
from pathlib import Path

out = Path(os.environ["META_HARNESS_OUTPUT"])
out.mkdir(parents=True, exist_ok=True)
(out / "candidate-00.py").write_text("class Harness:\\n    def run(self, task, model, trace):\\n        return 'y'\\n")
(out / "candidate-00.reasoning.md").write_text("tried a wider label primer")
print("done")
''', encoding="utf-8")

    proposer = ClaudeCodeProposer(binary=sys.executable, extra_args=(str(fake),), prompt_as_argument=False)
    paths = proposer(experience, iteration=1, count=1)
    assert [p.name for p in paths] == ["candidate-00.py"]
    assert paths[0].with_name("candidate-00.reasoning.md").is_file()
    stdout = (experience.root / "proposals" / "iteration-0001" / "proposer.stdout").read_text()
    assert "done" in stdout


def test_proposer_records_failure_without_swallowing_it(tmp_path: Path):
    experience = _experience(tmp_path)
    proposer = ClaudeCodeProposer(binary=sys.executable, extra_args=("-c", "raise SystemExit(3)"),
                                  prompt_as_argument=False)
    try:
        proposer(experience, iteration=2, count=1)
    except RuntimeError as exc:
        assert "3" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
    assert (experience.root / "proposals" / "iteration-0002" / "proposer.stderr").exists()
