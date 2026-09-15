"""Dataset adapters with no mandatory third-party dependencies."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def read_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("JSON dataset must contain a list")
        return [dict(item) for item in value]
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"unsupported dataset extension: {path.suffix}")


def classification_tasks(records: Iterable[Mapping[str, Any]], text_field: str = "input", label_field: str = "label") -> list[dict[str, Any]]:
    result = []
    for record in records:
        if text_field not in record or label_field not in record:
            raise ValueError(f"classification records need {text_field!r} and {label_field!r}")
        task = {"input": str(record[text_field]), "label": str(record[label_field])}
        if record.get("labels"):
            task["labels"] = [str(x) for x in record["labels"]]
        result.append(task)
    return result


def math_tasks(records: Iterable[Mapping[str, Any]], problem_field: str = "problem", answer_field: str | None = "answer") -> list[dict[str, Any]]:
    result = []
    for record in records:
        if problem_field not in record:
            raise ValueError(f"math records need {problem_field!r}")
        task = {"problem": str(record[problem_field])}
        if answer_field and answer_field in record:
            task["answer"] = record[answer_field]
        result.append(task)
    return result


def terminal_tasks(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for record in records:
        instruction = record.get("instruction") or record.get("input")
        if not instruction:
            raise ValueError("terminal records need 'instruction'")
        result.append({
            "instruction": str(instruction),
            "test_command": str(record.get("test_command", "")),
            "image": str(record["image"]) if record.get("image") else None,
            "workdir": str(record["workdir"]) if record.get("workdir") else None,
            "timeout": float(record.get("timeout", 300.0)),
        })
    return result


def split_tasks(tasks: Sequence[Mapping[str, Any]], search_fraction: float = 0.7,
                seed: int = 0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Shuffle deterministically, then cut into a search split and a held-out test split."""
    if not 0.0 < search_fraction < 1.0:
        raise ValueError("search_fraction must be strictly between 0 and 1")
    items = [dict(task) for task in tasks]
    random.Random(seed).shuffle(items)
    cut = max(1, min(len(items) - 1, round(len(items) * search_fraction)))
    return items[:cut], items[cut:]
