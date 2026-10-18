"""Deterministic grader: the ANSWER line matches an accepted spelling AND the ticket store holds
exactly the gold ticket (queue, doc_id, priority) - or no ticket when none is due."""
from __future__ import annotations

import re


def _norm(text: str) -> str:
    text = str(text).lower().replace("-", " ").replace(",", "").replace("$", "")
    return " " + re.sub(r"\s+", " ", re.sub(r"[^\w/ ]", " ", text)).strip() + " "


def answer_ok(task: dict, answer: str) -> bool:
    got = _norm(answer)
    return any(f" {_norm(a).strip()} " in got for a in task["accept"])


def ticket_ok(task: dict, tickets: list[dict]) -> bool:
    gold = task.get("ticket")
    if not gold:
        return not tickets
    return len(tickets) == 1 and all(tickets[0].get(k) == v for k, v in gold.items())


def grade(task: dict, result: dict) -> dict:
    a = answer_ok(task, result.get("answer", ""))
    t = ticket_ok(task, result.get("tickets") or [])
    return {"score": 1.0 if (a and t) else 0.0, "answer_ok": a, "ticket_ok": t}


def metric(task: dict, result: dict) -> float:
    return grade(task, result)["score"]
