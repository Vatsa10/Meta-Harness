"""Ablation over experience views (paper Table 3 shape): N searches per view, median and best."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core import atomic_write
from .providers import Price, price_for
from .stats import compare
from .transfer import task_outcomes

DEFAULT_RUNS = 3
DEFAULT_ITERATIONS = 3
RATE = "ended_without_writing_code_rate"


def _read(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _flag(argv: Sequence[str], name: str) -> str | None:
    for i, token in enumerate(argv):
        if token == name and i + 1 < len(argv):
            return argv[i + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def has_flag(argv: Sequence[str], name: str) -> bool:
    return any(t == name or t.startswith(name + "=") for t in argv)


def _price(argv: Sequence[str]) -> Price | None:
    model = _flag(argv, "--model")
    if not model:
        return None
    for i, spec in enumerate(argv):
        if i and argv[i - 1] == "--price":
            target, _, numbers = spec.rpartition("=")
            if target.partition(":")[2] == model:
                try:
                    return Price(*[float(v) for v in numbers.split(",")])
                except (TypeError, ValueError):
                    break
    return price_for(model)


def _run_record(run_root: Path, price: Price | None) -> dict[str, Any]:
    """Best search score, best seed score, best test score, per-task test rewards, cost, phase rate."""
    record: dict[str, Any] = {"search_best": None, "seed_best": None, "test_best": None,
                              "rewards": {}, "cost_per_task": None, RATE: None}
    frontier = _read(run_root / "frontier.json")
    scores = [float(x["score"]) for x in frontier or [] if isinstance(x, Mapping) and "score" in x]
    record["search_best"] = max(scores) if scores else None
    seeds = [_read(p) for p in sorted((run_root / "candidates").glob("initial-*/scores.json"))]
    seed_scores = [float(s["score"]) for s in seeds if isinstance(s, Mapping) and "score" in s]
    record["seed_best"] = max(seed_scores) if seed_scores else None
    test_root = run_root.with_name(run_root.name + "-test")
    payload = _read(test_root / "test_results.json")
    results = payload.get("results") if isinstance(payload, Mapping) else None
    results = [r for r in results or [] if isinstance(r, Mapping) and "score" in r]
    if results:
        best = max(results, key=lambda r: float(r["score"]))
        record["test_best"] = float(best["score"])
        rate = (best.get("metrics") or {}).get(RATE)
        record[RATE] = None if rate is None else float(rate)
        trace = test_root / "traces" / f"{best.get('candidate_id')}.jsonl"
        if trace.is_file():
            try:
                outcomes = task_outcomes(trace, price)
            except (OSError, ValueError):
                outcomes = []
            record["rewards"] = {o["task"]: o["reward"] for o in outcomes}
            costs = [o["cost_usd"] for o in outcomes]
            if costs and all(c is not None for c in costs):
                record["cost_per_task"] = sum(costs) / len(costs)
    return record


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def run_ablation(search_argv: list[str], views: Sequence[str], runs: int, root: Path,
                 search: Callable[[list[str]], int], baseline: str | None = None) -> dict:
    """For each view and run i, calls search(search_argv + ["--view", v, "--root", root/v/f"run-{i}"]).

    Reads each run's frontier.json (best search score) and test_results.json (best test score).
    Writes root/ablation.json. Per view: median/best of search and test scores, `above_seed` (runs
    whose best search score beats the best initial-* candidate), mean cost per test task, the
    ended-without-writing-code rate when the harness reports phases, and (for every view but the
    baseline, default the first) paired win/loss/tie, exact McNemar p, BH q and `significant` on the
    held-out per-task rewards of the best test candidate, pooled over runs (run i against run i).
    """
    for flag in ("--view", "--root"):
        if has_flag(search_argv, flag):
            raise ValueError(f"search arguments must not include {flag}; ablate sets it per view and run")
    views = list(dict.fromkeys(views))
    if not views:
        raise ValueError("at least one view is required")
    if runs < 1:
        raise ValueError("--runs must be at least 1")
    baseline = baseline or views[0]
    if baseline not in views:
        raise ValueError(f"baseline {baseline!r} is not one of the views {views}")
    root = Path(root)
    price = _price(search_argv)
    report: dict[str, Any] = {"views": {}, "baseline": baseline}
    rewards: dict[str, dict[tuple[int, Any], float]] = {}
    for view in views:
        records, exit_codes = [], []
        for i in range(runs):
            run_root = root / view / f"run-{i}"
            exit_codes.append(int(search(list(search_argv) + ["--view", view, "--root", str(run_root)])))
            records.append(_run_record(run_root, price))
        search_best = [r["search_best"] for r in records if r["search_best"] is not None]
        test_best = [r["test_best"] for r in records if r["test_best"] is not None]
        costs = [r["cost_per_task"] for r in records if r["cost_per_task"] is not None]
        rates = [r[RATE] for r in records if r[RATE] is not None]
        report["views"][view] = {
            "runs": runs, "search_best": search_best, "test_best": test_best,
            "median_search": _median(search_best), "best_search": max(search_best, default=None),
            "median_test": _median(test_best), "best_test": max(test_best, default=None),
            "above_seed": sum(1 for r in records if r["search_best"] is not None
                              and r["seed_best"] is not None and r["search_best"] > r["seed_best"]),
            "mean_cost_usd_per_task": sum(costs) / len(costs) if costs else None,
            RATE: sum(rates) / len(rates) if rates else None,
            "exit_codes": exit_codes,
        }
        rewards[view] = {(i, t): v for i, r in enumerate(records) for t, v in r["rewards"].items()}
    others = {v: rewards[v] for v in views if v != baseline}
    for view, result in compare(rewards[baseline], others).items():
        report["views"][view]["vs_baseline"] = result
    atomic_write(root / "ablation.json", json.dumps(report, indent=2))
    return report


def format_table(report: Mapping[str, Any]) -> str:
    def num(x, spec="{:.3f}"):
        return "-" if x is None else spec.format(x)

    lines = [f"{'view':<10}{'median':>8}{'best':>8}{'>seed':>7}{'test med':>10}{'test best':>11}"
             f"{'$/task':>10}{'w/l/t':>10}{'p':>8}{'q':>8}{'sig':>5}"]
    for view, v in report["views"].items():
        c = v.get("vs_baseline")
        wlt = f"{c['wins']}/{c['losses']}/{c['ties']}" if c else "-"
        sig = ("yes" if c["significant"] else "no") if c else "-"
        lines.append(f"{view:<10}{num(v['median_search']):>8}{num(v['best_search']):>8}"
                     f"{str(v['above_seed']) + '/' + str(v['runs']):>7}"
                     f"{num(v['median_test']):>10}{num(v['best_test']):>11}"
                     f"{num(v['mean_cost_usd_per_task'], '{:.4f}'):>10}"
                     f"{wlt:>10}{num(c and c['p']):>8}{num(c and c['q']):>8}{sig:>5}")
        if v.get(RATE) is not None:
            lines.append(f"{'':<10}ended without writing code: {v[RATE]:.1%}")
    return "\n".join(lines)
