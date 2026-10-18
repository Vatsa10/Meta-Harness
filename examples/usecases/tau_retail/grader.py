"""Deterministic grader: final DB state equals the gold state AND every required output was said.

Mirrors tau-bench's reward (state hash match x outputs-in-messages); no LLM judging.
"""
from __future__ import annotations


def _norm(text: str) -> str:
    return str(text).lower().replace(",", "").replace("$", "")


def grade(task: dict, result: dict) -> dict:
    state_ok = result.get("state_digest") == task["gold_state_digest"]
    said = _norm(" ".join(result.get("agent_messages") or [result.get("output", "")]))
    missing = [o for o in task.get("required_outputs") or [] if _norm(o) not in said]
    score = 1.0 if (state_ok and not missing) else 0.0
    return {"score": score, "state_ok": state_ok, "missing_outputs": missing}


def metric(task: dict, result: dict) -> float:
    return grade(task, result)["score"]
