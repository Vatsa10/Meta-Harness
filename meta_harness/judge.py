"""Blind LLM-judge metric: `--metric meta_harness.judge:verdict`."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .protocol import expected_value

JUDGE_PROMPT = ("You are a strict grader. Criteria:\n{criteria}\n\nAnswer:\n{answer}\n\n"
                "Reason briefly, then end with exactly one line: VERDICT: PASS or VERDICT: FAIL")

PROVIDER_ENV = "META_HARNESS_JUDGE_PROVIDER"
MODEL_ENV = "META_HARNESS_JUDGE_MODEL"


def judge_model() -> Any:
    provider, model = os.environ.get(PROVIDER_ENV, ""), os.environ.get(MODEL_ENV, "")
    if not provider or not model:
        raise ValueError(f"set {PROVIDER_ENV} and {MODEL_ENV} to choose the judge model")
    from .providers import model_from_environment
    return model_from_environment(provider, model)


def parse_verdict(text: str) -> float:
    """Score the LAST line containing 'VERDICT:'; only 'VERDICT: PASS' is 1.0."""
    last = None
    for line in str(text).splitlines():
        if "VERDICT:" in line:
            last = line
    if last is None:
        return 0.0
    return 1.0 if last.strip() == "VERDICT: PASS" else 0.0


def build_prompt(task: Mapping, result: Mapping) -> str:
    criteria = task.get("criteria")
    if criteria is None:
        criteria = expected_value(task)
    return JUDGE_PROMPT.format(criteria="" if criteria is None else criteria,
                               answer=result.get("output", ""))


def verdict(task: Mapping, result: Mapping) -> float:
    """Blind: the judge sees the criteria (or expected value) and result['output'] only."""
    return parse_verdict(judge_model()(build_prompt(task, result)))
