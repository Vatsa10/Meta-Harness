"""Deterministic grader: the saved scratch result equals the gold result set.

Rows are compared as a multiset; within a row, values are compared as a multiset too (column order
and names do not matter, column count does). Numbers are rounded to 2 decimals; text is case-folded.
"""
from __future__ import annotations

from collections import Counter


def _cell(v) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, (int, float)):
        return f"{round(float(v), 2):.2f}"
    if v is None:
        return "NULL"
    s = str(v).strip()
    try:
        return f"{round(float(s), 2):.2f}"
    except ValueError:
        return s.lower()


def canon(rows) -> Counter:
    return Counter(tuple(sorted(_cell(v) for v in row)) for row in rows or [])


def grade(task: dict, result: dict) -> dict:
    rows = result.get("result_rows")
    ok = rows is not None and canon(rows) == canon(task["gold_rows"])
    return {"score": 1.0 if ok else 0.0, "saved": rows is not None, "got_rows": len(rows or []),
            "gold_rows": len(task["gold_rows"])}


def metric(task: dict, result: dict) -> float:
    return grade(task, result)["score"]
