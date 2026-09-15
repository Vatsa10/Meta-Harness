"""Dataset adapters with no mandatory third-party dependencies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


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
        result.append({"input": str(record[text_field]), "label": str(record[label_field])})
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
