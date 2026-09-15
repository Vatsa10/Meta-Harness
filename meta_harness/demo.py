from __future__ import annotations

import shutil
from pathlib import Path

from .core import CandidateEvaluator, SearchConfig, SearchRunner


class DemoModel:
    def __call__(self, prompt: str, **_: object) -> str:
        text = prompt.lower()
        if "fruit" in text or "apple" in text:
            return "fruit"
        if "vehicle" in text or "car" in text:
            return "vehicle"
        return "other"


def _write(path: Path, mode: str) -> None:
    path.write_text(f'''class Harness:\n    def run(self, task, model, trace):\n        text = task["input"]\n        prompt = text + (" fruit" if {mode!r} == "keyword" else " vehicle")\n        trace.event("prompt", {{"prompt": prompt}})\n        return model(prompt)\n''', encoding="utf-8")


def run(iterations: int = 3, root: str = ".meta-harness-demo") -> Path:
    root_path = Path(root)
    if root_path.exists():
        shutil.rmtree(root_path)
    proposal_dir = root_path / "demo-proposals"
    proposal_dir.mkdir(parents=True)
    initial = proposal_dir / "initial.py"
    _write(initial, "keyword")

    def proposer(experience, iteration, count):
        path = experience.root / "demo-proposals" / f"proposal-{iteration}.py"
        _write(path, "keyword" if iteration >= 2 else "wrong")
        return [path]

    tasks = [{"input": "an apple", "label": "fruit"}, {"input": "a car", "label": "vehicle"}]
    runner = SearchRunner(SearchConfig(root_path, iterations, 1), CandidateEvaluator(DemoModel()), proposer)
    runner.run([initial], tasks)
    return root_path
