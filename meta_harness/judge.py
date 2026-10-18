"""Blind LLM-judge metric: `--metric meta_harness.judge:verdict`."""

from __future__ import annotations

import hashlib
import os
import random
import re
import time
from typing import Any, Mapping

from .protocol import expected_value

JUDGE_PROMPT = ("You are a strict grader. Criteria:\n{criteria}\n\nAnswer:\n{answer}\n\n"
                "Reason briefly, then end with exactly one line: VERDICT: PASS or VERDICT: FAIL")

PROVIDER_ENV = "META_HARNESS_JUDGE_PROVIDER"
MODEL_ENV = "META_HARNESS_JUDGE_MODEL"
JUDGE_RETRIES = 2
RETRY_DELAY = 1.0


def http_status(exc: BaseException) -> int | None:
    """HTTP status of a provider failure: `.status` if set, else parsed from 'HTTP 429: ...'."""
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status
    if type(exc).__name__ == "ProviderError":
        found = re.search(r"\bHTTP (\d{3})\b", str(exc))
        if found:
            return int(found.group(1))
    return None


def retryable(status: int | None) -> bool:
    return status is not None and (status == 429 or status >= 500)


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
    model, prompt = judge_model(), build_prompt(task, result)
    for attempt in range(JUDGE_RETRIES + 1):
        try:
            return parse_verdict(model(prompt))
        except Exception as exc:
            if not retryable(http_status(exc)) or attempt == JUDGE_RETRIES:
                raise
            time.sleep(RETRY_DELAY)
    return 0.0


# -- multi-candidate judging with anonymous ids --------------------------------------------------
# Idea from google-research/harness2 (Apache-2.0), core/selection.py assign_ids: candidates are
# shown under seeded, shuffled, opaque ids so neither order nor names reach the judge.

RANK_PROMPT = ("You are a strict grader. Criteria:\n{criteria}\n\nCandidate answers:\n{candidates}\n\n"
               "Reason briefly, then end with exactly one line: BEST: <candidate id>")


def assign_ids(names: list[str], seed: int | str = 0) -> dict[str, str]:
    """Opaque id per name ("cand-7f3a"), deterministic given seed and the SET of names; the
    id carries no information about the name or its position."""
    ids: dict[str, str] = {}
    taken: set[str] = set()
    for name in sorted(set(names)):
        salt = 0
        while True:
            digest = hashlib.sha256(f"{seed}\x00{name}\x00{salt}".encode("utf-8")).hexdigest()[:4]
            if f"cand-{digest}" not in taken:
                break
            salt += 1
        taken.add(f"cand-{digest}")
        ids[name] = f"cand-{digest}"
    return ids


def shuffled_order(ids: Mapping[str, str], seed: int | str = 0) -> list[str]:
    """Names in a seeded shuffled order (stable for a given seed and name set)."""
    names = sorted(ids)
    random.Random(f"{seed}:order").shuffle(names)
    return names


def build_rank_prompt(task: Mapping, outputs: Mapping[str, Any], seed: int | str = 0) -> tuple[str, dict[str, str]]:
    """(prompt, id -> name). The prompt shows only ids and outputs, in shuffled order."""
    criteria = task.get("criteria")
    if criteria is None:
        criteria = expected_value(task)
    ids = assign_ids(list(outputs), seed)
    blocks = [f"[{ids[name]}]\n{outputs[name]}" for name in shuffled_order(ids, seed)]
    prompt = RANK_PROMPT.format(criteria="" if criteria is None else criteria, candidates="\n\n".join(blocks))
    return prompt, {cid: name for name, cid in ids.items()}


def parse_best(text: str, mapping: Mapping[str, str]) -> str | None:
    """Map the LAST 'BEST: <id>' line back to the candidate name; None if absent or unknown."""
    last = None
    for line in str(text).splitlines():
        found = re.match(r"\s*BEST:\s*\[?([\w-]+)\]?\s*$", line)
        if found:
            last = found.group(1)
    return mapping.get(last) if last is not None else None


def pick_best(task: Mapping, outputs: Mapping[str, Any], seed: int | str = 0) -> str | None:
    """Ask the judge to choose among several outputs; returns the winning NAME (or None)."""
    if len(outputs) < 2:
        return next(iter(outputs), None)
    model = judge_model()
    prompt, mapping = build_rank_prompt(task, outputs, seed)
    for attempt in range(JUDGE_RETRIES + 1):
        try:
            return parse_best(model(prompt), mapping)
        except Exception as exc:
            if not retryable(http_status(exc)) or attempt == JUDGE_RETRIES:
                raise
            time.sleep(RETRY_DELAY)
    return None
