import importlib.util
import json
from pathlib import Path

import pytest

from meta_harness.core import TraceRecorder
from meta_harness.sandbox import validate_in_subprocess

ROOT = Path(__file__).resolve().parent.parent / "baselines"
BASELINES = ["zero_shot", "few_shot", "ace", "mce", "math_zero_shot", "math_bm25", "terminal_basic"]
CLASSIFICATION = ["zero_shot", "few_shot", "ace", "mce"]


def _load(name: str):
    path = ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"baseline_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_harness() if hasattr(module, "build_harness") else module.Harness()


@pytest.mark.parametrize("name", BASELINES)
def test_baseline_passes_interface_validation(name):
    validate_in_subprocess(ROOT / f"{name}.py", timeout=60.0)


@pytest.mark.parametrize("name", CLASSIFICATION)
def test_classification_baseline_predicts_a_declared_label(tmp_path: Path, name):
    harness = _load(name)
    tasks = [
        {"input": "an apple", "label": "fruit", "labels": ["fruit", "vehicle"]},
        {"input": "a fast car", "label": "vehicle", "labels": ["fruit", "vehicle"]},
    ]

    def model(prompt, **_):
        return "vehicle" if "car" in prompt.rsplit("Input:", 1)[-1] else "fruit"

    with TraceRecorder(tmp_path / f"{name}.jsonl") as trace:
        predictions = [harness.run(task, model, trace) for task in tasks]
    assert predictions == ["fruit", "vehicle"]


def test_few_shot_respects_n(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("META_HARNESS_FEW_SHOT_N", "2")
    harness = _load("few_shot")
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return "a"

    with TraceRecorder(tmp_path / "fs.jsonl") as trace:
        for index in range(5):
            harness.run({"input": f"text {index}", "label": "a", "labels": ["a", "b"]}, model, trace)
    assert prompts[-1].count("Example:") <= 2


def test_math_bm25_retrieves_from_corpus(tmp_path: Path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(x) for x in [
        {"problem": "Find the angle in a triangle", "solution": "Angles sum to 180"},
        {"problem": "Show that 7 is prime", "solution": "Check divisors"},
    ]), encoding="utf-8")
    monkeypatch.setenv("META_HARNESS_CORPUS", str(corpus))
    harness = _load("math_bm25")
    prompts = []

    def model(prompt, **_):
        prompts.append(prompt)
        return r"\boxed{60}"

    with TraceRecorder(tmp_path / "m.jsonl") as trace:
        answer = harness.run({"problem": "What is the angle of an equilateral triangle?", "answer": "60"},
                             model, trace)
    assert answer == r"\boxed{60}"
    assert "Angles sum to 180" in prompts[0]
