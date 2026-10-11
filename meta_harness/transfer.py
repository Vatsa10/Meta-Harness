"""Multi-model transfer: score fixed harnesses across several models (paper Table 6). No proposer."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .chat_types import Usage
from .core import CandidateEvaluator, atomic_write
from .protocol import Metric
from .providers import Price, cost_usd, model_from_environment, price_for

PASS_THRESHOLD = 0.5


def slug(text: str, limit: int = 48) -> str:
    """A directory-safe name: no colon, slash, backslash or other character Windows rejects."""
    text = str(text)
    name = re.split(r"[\\/]", text)[-1] if re.search(r"[\\/]", text) else text
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.") or "model"
    if clean != text or len(clean) > limit:
        clean = clean[:limit].rstrip("-.") + "-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:6]
    return clean


def cell_name(provider: str, model: str) -> str:
    return f"{slug(provider)}-{slug(model)}"


def parse_models(text: str) -> list[tuple[str, str]]:
    """'unikey:gpt-5-mini,groq:llama' -> [('unikey', 'gpt-5-mini'), ('groq', 'llama')]."""
    out = []
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        provider, sep, model = item.partition(":")
        if not sep or not provider or not model:
            raise ValueError(f"model must be provider:model, got {item!r}")
        out.append((provider, model))
    return out


def task_outcomes(trace_path: Path, price: Price | None) -> list[dict[str, Any]]:
    """Per-task reward and cost from one evaluation trace (model calls between task_start/task_end)."""
    outcomes: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in Path(trace_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        event, payload = record.get("event"), record.get("payload") or {}
        if event == "task_start":
            current = {"task": payload.get("index"), "prompt_tokens": 0, "completion_tokens": 0,
                       "cache_read_tokens": 0, "cache_write_tokens": 0}
        elif event == "model_call" and current is not None:
            usage = payload.get("usage") or {}
            current["prompt_tokens"] += int(payload.get("tokens") or 0)
            current["completion_tokens"] += int(usage.get("completion_tokens") or usage.get("output") or 0)
            current["cache_read_tokens"] += int(usage.get("cache_read") or 0)
            current["cache_write_tokens"] += int(usage.get("cache_write") or 0)
        elif event == "task_end" and current is not None:
            current["reward"] = float(payload.get("reward") or 0.0)
            current["stop_reason"] = payload.get("stop_reason")
            current["cost_usd"] = cost_usd(
                Usage(input=current["prompt_tokens"], output=current["completion_tokens"],
                      cache_read=current["cache_read_tokens"], cache_write=current["cache_write_tokens"]),
                price)
            outcomes.append(current)
            current = None
    return outcomes


def _merge_repeats(runs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Average each task over repeats. `pass` is the binary outcome a paired test consumes."""
    merged = []
    for index in range(max((len(r) for r in runs), default=0)):
        cells = [r[index] for r in runs if index < len(r)]
        reward = sum(c["reward"] for c in cells) / len(cells)
        costs = [c["cost_usd"] for c in cells]
        merged.append({
            "task": cells[0]["task"], "reward": reward, "pass": reward > PASS_THRESHOLD,
            "prompt_tokens": sum(c["prompt_tokens"] for c in cells) / len(cells),
            "completion_tokens": sum(c["completion_tokens"] for c in cells) / len(cells),
            "cache_read_tokens": sum(c["cache_read_tokens"] for c in cells) / len(cells),
            "cache_write_tokens": sum(c["cache_write_tokens"] for c in cells) / len(cells),
            "cost_usd": None if any(c is None for c in costs) else sum(costs) / len(costs),
        })
    return merged


def transfer(harnesses: Sequence[Path], tasks: Sequence[Mapping], models: Sequence[tuple[str, str]],
             metric: Metric, root: Path, repeats: int = 1, protocol: str = "plain",
             train_tasks: Sequence[Mapping] = ()) -> dict:
    """Evaluate each fixed harness with each (provider, model); no proposer. Writes
    root/transfer.json {"rows": [...]} and per-cell traces under root/<harness-stem>/<cell>/.

    Each row carries "outcomes": one {task, reward, pass, prompt_tokens, completion_tokens,
    cost_usd} per task, aligned by task index across rows, which is the paired shape a
    significance test consumes. Row "cost_usd" is the TOTAL billed cost of the n tasks (mean over
    repeats); "cost_usd_per_task" is that total divided by n. Both are None when the model's price is
    unknown (META_HARNESS_PRICES keyed by model id). Cost covers evaluation only: tokens spent
    streaming train_tasks under the online protocol are excluded (see context_cost, also eval-only). Every (harness, model, repeat) evaluation loads the harness
    afresh, so nothing a harness memorises under one model reaches another. A model that cannot
    be built, or a harness that cannot be loaded, yields a row with "error" and the run goes on.
    """
    stems = [Path(h).stem for h in harnesses]
    duplicates = sorted({x for x in stems if stems.count(x) > 1})
    if duplicates:
        raise ValueError(f"harness file names must be unique (same stem would share a trace directory): {duplicates}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    built: dict[tuple[str, str], Any] = {}
    build_errors: dict[tuple[str, str], str] = {}
    for provider, model in models:
        try:
            built[(provider, model)] = model_from_environment(provider, model)
        except Exception as exc:  # noqa: BLE001 - a missing key must not stop the other models
            build_errors[(provider, model)] = f"{type(exc).__name__}: {exc}"
    rows: list[dict[str, Any]] = []
    for harness in map(Path, harnesses):
        for provider, model in models:
            row: dict[str, Any] = {"harness": harness.stem, "provider": provider, "model": model,
                                   "score": None, "score_std": None, "context_cost": None, "n": 0}
            key = (provider, model)
            if key in build_errors:
                rows.append({**row, "error": build_errors[key]})
                continue
            cell = root / harness.stem / cell_name(provider, model)
            price = price_for(model)
            scores: list[float] = []
            costs: list[float] = []
            runs: list[list[dict[str, Any]]] = []
            error = None
            try:
                evaluator = CandidateEvaluator(built[key], metric, protocol=protocol, train_tasks=train_tasks)
                for repeat in range(max(1, repeats)):
                    trace = cell / f"trace-{repeat}.jsonl"
                    result = evaluator.evaluate(harness, tasks, f"{harness.stem}@{cell.name}", trace, repeat)
                    if not result.valid:
                        error = result.error or "invalid harness"
                        break
                    scores.append(result.score)
                    costs.append(result.context_cost)
                    runs.append(task_outcomes(trace, price))
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            if error:
                rows.append({**row, "error": error})
                continue
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
            outcomes = _merge_repeats(runs)
            usd = [o["cost_usd"] for o in outcomes]
            total = None if not usd or any(u is None for u in usd) else sum(usd)
            rows.append({**row, "score": mean, "score_std": std,
                         "context_cost": sum(costs) / len(costs), "n": len(tasks),
                         "cost_usd": total, "cost_usd_per_task": None if total is None else total / len(usd),
                         "outcomes": outcomes})
    report = {"rows": rows, "repeats": max(1, repeats), "protocol": protocol}
    atomic_write(root / "transfer.json", json.dumps(report, indent=2))
    return report


def format_table(rows: Sequence[Mapping[str, Any]]) -> str:
    """Harness x model grid of scores; failed cells show ERR."""
    models = list(dict.fromkeys(f"{r['provider']}:{r['model']}" for r in rows))
    harnesses = list(dict.fromkeys(r["harness"] for r in rows))
    cells = {(r["harness"], f"{r['provider']}:{r['model']}"):
             "ERR" if r.get("error") else f"{r['score']:.3f}" for r in rows}
    width = max([len(h) for h in harnesses] + [7])
    lines = ["harness".ljust(width) + "  " + "  ".join(models)]
    for h in harnesses:
        lines.append(h.ljust(width) + "  " + "  ".join(cells.get((h, m), "-").ljust(len(m)) for m in models))
    return "\n".join(lines)
