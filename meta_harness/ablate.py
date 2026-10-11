"""Ablation over experience views (paper Table 3 shape): N searches per view, median and best."""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .core import atomic_write
from .datasets import read_records
from .providers import Price, price_for
from .stats import PASS_THRESHOLD, bh, mcnemar_exact
from .transfer import task_outcomes

DEFAULT_RUNS = 3
DEFAULT_ITERATIONS = 3
DEFAULT_MAX_TASKS = 20
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


def _task_key(task: Any, index: Any) -> str:
    if isinstance(task, Mapping):
        if task.get("id") is not None:
            return f"id:{task['id']}"
        if "input" in task:
            return "input:" + json.dumps(task["input"], sort_keys=True, default=str)
    return f"index:{index}"


def _trace_rewards(trace: Path, price: Price | None) -> tuple[dict[str, float], float | None]:
    """{task key: reward} and mean cost per task, read from one held-out evaluation trace."""
    keys: dict[Any, str] = {}
    for line in trace.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("event") == "task_start":
                payload = rec.get("payload") or {}
                keys[payload.get("index")] = _task_key(payload.get("task"), payload.get("index"))
    outcomes = task_outcomes(trace, price)
    rewards = {keys.get(o["task"], f"index:{o['task']}"): o["reward"] for o in outcomes}
    costs = [o["cost_usd"] for o in outcomes]
    cost = sum(costs) / len(costs) if costs and all(c is not None for c in costs) else None
    return rewards, cost


def _run_record(run_root: Path, price: Price | None) -> dict[str, Any]:
    """The run's search-selected candidate: its search score, held-out score, rewards, cost, phase rate.

    The candidate is the frontier member with the highest SEARCH score (tie: lower context cost);
    the held-out numbers are that candidate's, never the best over the held-out split.
    """
    record: dict[str, Any] = {"search_best": None, "seed_best": None, "test_best": None,
                              "rewards": {}, "cost_per_task": None, RATE: None}
    frontier = [x for x in _read(run_root / "frontier.json") or [] if isinstance(x, Mapping) and "score" in x]
    chosen = max(frontier, key=lambda x: (float(x["score"]), -float(x.get("context_cost") or 0.0)),
                 default=None)
    record["search_best"] = None if chosen is None else float(chosen["score"])
    seeds = [_read(p) for p in sorted((run_root / "candidates").glob("initial-*/scores.json"))]
    seed_scores = [float(s["score"]) for s in seeds if isinstance(s, Mapping) and "score" in s]
    record["seed_best"] = max(seed_scores) if seed_scores else None
    test_root = run_root.with_name(run_root.name + "-test")
    payload = _read(test_root / "test_results.json")
    results = payload.get("results") if isinstance(payload, Mapping) else None
    match = next((r for r in results or [] if chosen is not None and isinstance(r, Mapping)
                  and "score" in r and r.get("candidate_id") == chosen.get("candidate_id")), None)
    if match is not None:
        record["test_best"] = float(match["score"])
        rate = (match.get("metrics") or {}).get(RATE)
        record[RATE] = None if rate is None else float(rate)
        trace = test_root / "traces" / f"{match['candidate_id']}.jsonl"
        if trace.is_file():
            try:
                record["rewards"], record["cost_per_task"] = _trace_rewards(trace, price)
            except (OSError, ValueError):
                pass
    return record


def _collapse(runs: Sequence[Mapping[str, float]]) -> dict[str, bool | None]:
    """One outcome per task: the majority pass/fail over the runs, None on an exact tie."""
    votes: dict[str, list[bool]] = {}
    for rewards in runs:
        for key, reward in rewards.items():
            votes.setdefault(key, []).append(reward > PASS_THRESHOLD)
    return {k: (None if 2 * sum(v) == len(v) else 2 * sum(v) > len(v)) for k, v in votes.items()}


def _compare(baseline: Mapping[str, bool | None],
             arms: Mapping[str, Mapping[str, bool | None]]) -> dict[str, dict]:
    counts = []
    for arm in arms.values():
        wins = losses = ties = 0
        for key in baseline.keys() & arm.keys():
            b, a = baseline[key], arm[key]
            if a is None or b is None or a == b:
                ties += 1
            elif a:
                wins += 1
            else:
                losses += 1
        counts.append({"wins": wins, "losses": losses, "ties": ties})
    pvals = [mcnemar_exact(c["losses"], c["wins"]) for c in counts]
    qvals, flags = bh(pvals)
    return {n: {**c, "p": p, "q": q, "significant": f}
            for n, c, p, q, f in zip(arms, counts, pvals, qvals, flags)}


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def cap_tasks(search_argv: list[str], max_tasks: int, root: Path) -> list[str]:
    """Replace `--data FILE` by a deterministic sample of at most max_tasks records (seed 0)."""
    argv = list(search_argv)
    for i, token in enumerate(argv):
        if token == "--data" and i + 1 < len(argv):
            records = read_records(argv[i + 1])
            if len(records) > max_tasks:
                keep = sorted(random.Random(0).sample(range(len(records)), max_tasks))
                target = Path(root) / "data-sampled.jsonl"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(json.dumps(records[k]) for k in keep) + "\n", encoding="utf-8")
                argv[i + 1] = str(target)
    return argv


def run_ablation(search_argv: list[str], views: Sequence[str], runs: int, root: Path,
                 search: Callable[[list[str]], int], baseline: str | None = None,
                 resume: bool = False) -> dict:
    """For each view and run i, calls search(search_argv + ["--view", v, "--root", root/v/f"run-{i}"]).

    Per run the candidate with the highest SEARCH score on frontier.json is taken (tie: lower context
    cost) and its held-out score from test_results.json is reported, never the best over the test
    split. Writes root/ablation.json. Per view: median/best of search and test scores, `above_seed`
    (runs whose best search score beats the best initial-* candidate), mean cost per test task, the
    ended-without-writing-code rate when reported, and (for every view but the baseline, default the
    first) paired win/loss/tie, exact McNemar p, BH q and `significant`. Pairing is by task id, with
    the runs of a view collapsed to one majority outcome per task (exact tie -> tie). Views whose
    held-out task ids differ raise ValueError. Runs whose search exited non-zero are left out of every
    statistic and listed under report["failed"]. A non-empty root is refused unless resume=True.
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
    if not resume and root.is_dir() and any(root.iterdir()):
        raise ValueError(f"{root} is not empty; pass --resume to continue it or choose a new --root")
    price = _price(search_argv)
    report: dict[str, Any] = {"views": {}, "baseline": baseline, "failed": []}
    collapsed: dict[str, dict[str, bool | None]] = {}
    for view in views:
        records, exit_codes = [], []
        for i in range(runs):
            run_root = root / view / f"run-{i}"
            argv = list(search_argv) + ["--view", view, "--root", str(run_root)]
            if resume and (run_root / "run.json").is_file():
                argv.append("--resume")
            code = int(search(argv))
            exit_codes.append(code)
            if code != 0:
                report["failed"].append({"view": view, "run": i, "exit_code": code})
                continue
            records.append(_run_record(run_root, price))
        search_best = [r["search_best"] for r in records if r["search_best"] is not None]
        test_best = [r["test_best"] for r in records if r["test_best"] is not None]
        costs = [r["cost_per_task"] for r in records if r["cost_per_task"] is not None]
        rates = [r[RATE] for r in records if r[RATE] is not None]
        report["views"][view] = {
            "runs": len(records), "requested_runs": runs,
            "search_best": search_best, "test_best": test_best,
            "median_search": _median(search_best), "best_search": max(search_best, default=None),
            "median_test": _median(test_best), "best_test": max(test_best, default=None),
            "above_seed": sum(1 for r in records if r["search_best"] is not None
                              and r["seed_best"] is not None and r["search_best"] > r["seed_best"]),
            "mean_cost_usd_per_task": sum(costs) / len(costs) if costs else None,
            RATE: sum(rates) / len(rates) if rates else None,
            "exit_codes": exit_codes,
        }
        collapsed[view] = _collapse([r["rewards"] for r in records if r["rewards"]])
    base_ids = set(collapsed[baseline])
    for view in views:
        if collapsed[view] and base_ids and set(collapsed[view]) != base_ids:
            raise ValueError(f"views {baseline!r} and {view!r} were held out on different tasks "
                             "(check --seed and --test-fraction); paired comparison refused")
    others = {v: collapsed[v] for v in views if v != baseline}
    for view, result in _compare(collapsed[baseline], others).items():
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
