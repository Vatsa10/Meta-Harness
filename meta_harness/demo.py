"""A deterministic, offline end-to-end demo: scripted providers + the Meta-Harness tool-using proposer."""

from __future__ import annotations

import shutil
from pathlib import Path

from .core import CandidateEvaluator, SearchConfig, SearchRunner
from .datasets import read_records, split_tasks
from .domains import get_domain
from .optimizers import OptimizerContext, build
from .scripted import ScriptedModel

DATA = Path(__file__).with_name("demo_data")


def run(iterations: int = 3, root: str = ".meta-harness-demo") -> Path:
    root_path = Path(root)
    if root_path.exists():
        shutil.rmtree(root_path)
    config = SearchConfig(root_path, iterations, 1,
                          run_meta={"provider": "scripted", "model": "demo_data/base.jsonl",
                                    "proposer_provider": "scripted",
                                    "proposer_model": "demo_data/proposer.jsonl",
                                    "optimizer": "metaharness", "view": "full"})
    test_root = Path(config.test_root)
    if test_root.exists():
        shutil.rmtree(test_root)
    search, test = split_tasks(read_records(DATA / "data.jsonl"), seed=0, test_fraction=0.3)
    ctx = OptimizerContext.for_config(config, ScriptedModel(DATA / "proposer.jsonl"), get_domain("generic"))
    runner = SearchRunner(config, CandidateEvaluator(ScriptedModel(DATA / "base.jsonl")),
                          build("metaharness", ctx))
    runner.run([DATA / "seed.py"], search, test)
    return root_path
