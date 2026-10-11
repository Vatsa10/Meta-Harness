"""GEPA-style reflective prompt/harness optimizer (Agrawal et al.), under the equal evaluation budget.

Per-task rewards come from each candidate's `traces.jsonl` `task_end` events (this run's
`candidates/*/traces.jsonl` only; never the held-out split). The candidates that are best on at
least one task form the Pareto set. The parent is sampled with probability proportional to the
number of tasks it wins. A minibatch of the 3 tasks where the parent scored lowest is shown with
input, output and reward; the model writes a reflection and then a revised harness (reflective
mutation). Every 4th proposal instead merges two Pareto parents whose task wins are disjoint.
`candidate-KK.parent.json` records `{"parents": [...], "operator": "reflect" | "merge"}`.

Not reproduced: GEPA's minibatch accept/reject loop. Acceptance is the SearchRunner's job, and every
proposal here is evaluated on the same budget as the other optimizers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

from ..metrics import estimate_tokens
from ..redact import bounded, redact, safe_id
from . import extract_python

NAME = "gepa"
MINIBATCH = 3
MERGE_EVERY = 4
FIELD_CHARS = 600

REFLECT_PROMPT = """You are improving a {domain} harness (a Python module with class Harness).

Current harness:
```python
{source}
```

It scored lowest on these tasks (input, its output, reward):
{minibatch}

Write a short reflection: what went wrong on these tasks and what change would fix it.
Do not write code yet."""

REVISE_PROMPT = """Reflection on the current harness:
{reflection}

Current harness:
```python
{source}
```

Write the complete revised harness as one fenced Python block (class Harness with run(self, task, model, trace))."""

MERGE_PROMPT = """You are combining two {domain} harnesses. Each is best on different tasks.

Harness A (best on tasks {wins_a}):
```python
{source_a}
```

Harness B (best on tasks {wins_b}):
```python
{source_b}
```

Write one complete harness (class Harness with run(self, task, model, trace)) that keeps what each
does well, as one fenced Python block."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def task_records(directory: Path) -> dict[int, dict[str, Any]]:
    """{task index: {input, output, reward}} from one candidate's traces.jsonl."""
    out: dict[int, dict[str, Any]] = {}
    inputs: dict[int, Any] = {}
    for row in _read_jsonl(Path(directory) / "traces.jsonl"):
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        index = payload.get("index")
        if not isinstance(index, int):
            continue
        if row.get("event") == "task_start":
            task = payload.get("task")
            inputs[index] = task.get("input") if isinstance(task, dict) else task
        elif row.get("event") == "task_end":
            try:
                reward = float(payload.get("reward") or 0.0)
            except (TypeError, ValueError):
                reward = 0.0
            out[index] = {"input": inputs.get(index), "output": payload.get("output"), "reward": reward}
    return out


def reward_matrix(candidates_dir: Path, valid: set[str] | None = None) -> dict[str, dict[int, float]]:
    """{candidate id: {task index: reward}} over candidates/*/traces.jsonl of this run."""
    matrix: dict[str, dict[int, float]] = {}
    for path in sorted(Path(candidates_dir).glob("*/traces.jsonl")):
        cid = path.parent.name
        if valid is not None and cid not in valid:
            continue
        records = task_records(path.parent)
        if records:
            matrix[cid] = {i: r["reward"] for i, r in records.items()}
    return matrix


def task_wins(matrix: Mapping[str, Mapping[int, float]]) -> dict[str, set[int]]:
    """Tasks each non-dominated candidate is best on (ties: every tied candidate wins that task).

    Only tasks every candidate was scored on count, so a partial (killed) candidate cannot win
    tasks it has no score for. A candidate strictly dominated by another (no better anywhere, worse
    somewhere) is pruned first and wins nothing.
    """
    wins: dict[str, set[int]] = {cid: set() for cid in matrix}
    if not matrix:
        return wins
    tasks = set.intersection(*(set(row) for row in matrix.values()))
    alive = [cid for cid in matrix if not any(
        other != cid
        and all(matrix[other][t] >= matrix[cid][t] for t in tasks)
        and any(matrix[other][t] > matrix[cid][t] for t in tasks)
        for other in matrix)]
    for task in tasks:
        best = max(matrix[cid][task] for cid in alive)
        for cid in alive:
            if matrix[cid][task] == best:
                wins[cid].add(task)
    return wins


def pareto_set(matrix: Mapping[str, Mapping[int, float]]) -> list[str]:
    """Candidates that are best on at least one task."""
    return sorted(cid for cid, won in task_wins(matrix).items() if won)


def sample_parent(rng: random.Random, wins: Mapping[str, set[int]]) -> str:
    """A Pareto candidate, with probability proportional to the number of tasks it wins."""
    pool = sorted(cid for cid, won in wins.items() if won)
    if not pool:
        raise ValueError("no candidate wins any task")
    return rng.choices(pool, weights=[len(wins[cid]) for cid in pool], k=1)[0]


def lowest_tasks(rewards: Mapping[int, float], k: int = MINIBATCH) -> list[int]:
    """The k task indices with the lowest reward (ties broken by index)."""
    return [i for i, _ in sorted(rewards.items(), key=lambda item: (item[1], item[0]))[:k]]


def merge_pair(rng: random.Random, wins: Mapping[str, set[int]]) -> tuple[str, str] | None:
    """Two Pareto parents with disjoint task wins, or None when no such pair exists."""
    pool = sorted(cid for cid, won in wins.items() if won)
    pairs = [(a, b) for n, a in enumerate(pool) for b in pool[n + 1:] if not (wins[a] & wins[b])]
    return rng.choice(pairs) if pairs else None


def _show(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return bounded(redact(text), FIELD_CHARS)


class GepaProposer:
    def __init__(self, model: Any, *, domain: str = "classification", seed: int = 0):
        self.model, self.domain, self.seed = model, domain, int(seed)
        self.usage: dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0}

    def _ask(self, prompt: str) -> str:
        reply = self.model(prompt)
        text = reply if isinstance(reply, str) else str(reply[0] if isinstance(reply, tuple) else reply)
        self.usage["calls"] += 1
        self.usage["input_tokens"] += estimate_tokens(prompt)
        self.usage["output_tokens"] += estimate_tokens(text)
        return text

    @staticmethod
    def _source(experience: Any, cid: str) -> str:
        return (Path(experience.root) / "candidates" / cid / "harness.py").read_text(
            encoding="utf-8", errors="replace")

    def __call__(self, experience: Any, iteration: int, count: int) -> list[Path]:
        root = Path(experience.root)
        valid = {r.candidate_id for r in experience.results() if r.valid}
        matrix = reward_matrix(root / "candidates", valid)
        wins = task_wins(matrix)
        if not any(wins.values()):
            raise RuntimeError("gepa needs at least one evaluated candidate with a traces.jsonl")
        output = root / "proposals" / f"iteration-{iteration:04d}"
        output.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for index in range(count):
            ordinal = (iteration - 1) * count + index + 1   # 1-based proposal number across the run
            rng = random.Random(f"{self.seed}:{iteration}:{index}")
            name = safe_id(f"candidate-{index:02d}")
            pair = merge_pair(rng, wins) if ordinal % MERGE_EVERY == 0 else None
            if pair:
                a, b = pair
                reasoning = f"merge of {a} and {b}"
                text = self._ask(MERGE_PROMPT.format(
                    domain=self.domain, source_a=self._source(experience, a),
                    source_b=self._source(experience, b),
                    wins_a=sorted(wins[a]), wins_b=sorted(wins[b])))
                parents, operator = [a, b], "merge"
            else:
                parent = sample_parent(rng, wins)
                records = task_records(root / "candidates" / parent)
                batch = lowest_tasks({i: r["reward"] for i, r in records.items()})
                shown = "\n\n".join(
                    f"Task {i}\n  input: {_show(records[i]['input'])}\n  output: {_show(records[i]['output'])}"
                    f"\n  reward: {records[i]['reward']}" for i in batch)
                source = self._source(experience, parent)
                reasoning = self._ask(REFLECT_PROMPT.format(domain=self.domain, source=source,
                                                            minibatch=shown))
                text = self._ask(REVISE_PROMPT.format(reflection=reasoning, source=source))
                parents, operator = [parent], "reflect"
            target = output / f"{name}.py"
            target.write_text(extract_python(text), encoding="utf-8")
            target.with_name(f"{name}.reasoning.md").write_text(reasoning, encoding="utf-8")
            target.with_name(f"{name}.parent.json").write_text(
                json.dumps({"parents": parents, "operator": operator}, indent=2), encoding="utf-8")
            paths.append(target)
        return paths


def build(ctx):
    return GepaProposer(ctx.proposer_model, domain=getattr(ctx.domain, "NAME", "classification"),
                        seed=ctx.seed)
