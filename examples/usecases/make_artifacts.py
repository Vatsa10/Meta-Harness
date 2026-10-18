"""Produce recordings-ready traces and labels for each use case, offline.

    python examples/usecases/make_artifacts.py [tau_retail sql_analyst research_ops]

For every task the oracle policy (scripted fake model) runs the real graph against a fresh sandbox;
the train split is also run with the broken policy so the labels hold both good and bad runs.
traces.jsonl rows carry `tool_calls: [{name, args, output, write}]`, which
`meta_harness.integrations.langgraph.tools.Recordings.from_traces` replays. labels.jsonl rows use the
`meta-harness label` format, graded by the deterministic grader (no human or LLM judgement).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from usecases._common import read_jsonl, write_jsonl  # noqa: E402

USECASES = ("tau_retail", "sql_analyst", "research_ops")
DROP = ("sandbox",)


def artifacts(name: str) -> tuple[list[dict], list[dict]]:
    app = importlib.import_module(f"usecases.{name}.app")
    grader = importlib.import_module(f"usecases.{name}.grader")
    oracle = importlib.import_module(f"usecases.{name}.oracle")
    traces, labels = [], []
    for task in read_jsonl(HERE / name / "tasks.jsonl"):
        runs = [("oracle", None)] + ([("broken", oracle.broken_model(task))] if task["split"] == "train" else [])
        for policy, llm in runs:
            result = app.run(task, llm)
            g = grader.grade(task, result)
            row = {k: v for k, v in result.items() if k not in DROP}
            row.update(id=task["id"] if policy == "oracle" else f"{task['id']}#broken", task_id=task["id"],
                       split=task["split"], policy=policy, score=g["score"], grade=g)
            traces.append(row)
            labels.append({"id": row["id"], "label": "good" if g["score"] == 1.0 else "bad",
                           "reason": "grader: " + json.dumps(g, sort_keys=True), "input": task["input"],
                           "output": row.get("output", ""), "criteria": f"deterministic grader for {name}",
                           "tool_calls": row["tool_calls"]})
    return traces, labels


def main(names) -> None:
    for name in names or USECASES:
        traces, labels = artifacts(name)
        write_jsonl(HERE / name / "traces.jsonl", traces)
        write_jsonl(HERE / name / "labels.jsonl", labels)
        good = sum(1 for x in labels if x["label"] == "good")
        print(f"{name}: {len(traces)} traces, labels good={good} bad={len(labels) - good}")


if __name__ == "__main__":
    main(sys.argv[1:])
