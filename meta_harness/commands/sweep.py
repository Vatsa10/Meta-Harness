"""`meta-harness sweep`: evaluate one harness under several environment settings (no proposer)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from ..cache import CachedModel
from ..core import CandidateEvaluator, atomic_write
from ..datasets import read_records, split_tasks
from ..protocol import BUILTIN_METRICS
from ..providers import model_from_environment
from .run import ADAPTERS, METRIC_FOR

NAME = "sweep"
HELP = "evaluate a harness at each value of an environment variable (no proposer)"


def add_arguments(parser) -> None:
    parser.add_argument("--harness", required=True, help="harness file to evaluate")
    parser.add_argument("--env", required=True, metavar="KEY=v1,v2,...",
                        help="environment variable and the comma-separated values to sweep")
    parser.add_argument("--tasks", required=True, help="JSON, JSONL, or CSV dataset")
    parser.add_argument("--test-tasks", help="held-out dataset; omit to split --tasks")
    parser.add_argument("--search-fraction", type=float, default=0.7)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--provider", choices=["unikey", "openai", "anthropic", "compatible"],
                        default="unikey")
    parser.add_argument("--model", required=True)
    parser.add_argument("--task-type", choices=["classification", "math", "terminal"],
                        default="classification")
    parser.add_argument("--protocol", choices=["plain", "online"], default="plain",
                        help="online: stream the search split, then score a frozen copy on the test split")
    parser.add_argument("--metric", choices=sorted(BUILTIN_METRICS), help="override the task-type metric")
    parser.add_argument("--root", default=".meta-harness-sweep")
    parser.add_argument("--cache-dir")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--task-timeout", type=float, default=600.0)


def parse_env(spec: str) -> tuple[str, list[str]]:
    key, sep, values = spec.partition("=")
    items = [v.strip() for v in values.split(",") if v.strip()]
    if not sep or not key.strip() or not items:
        raise ValueError(f"--env must look like KEY=v1,v2,...; got {spec!r}")
    return key.strip(), items


def _setting_dir(index: int, value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in value)
    return f"{index:02d}-{safe}"


def run(args) -> int:
    key, values = parse_env(args.env)
    adapt = ADAPTERS[args.task_type]
    search = adapt(read_records(args.tasks))
    if args.test_tasks:
        test = adapt(read_records(args.test_tasks))
    elif len(search) >= 2:
        search, test = split_tasks(search, args.search_fraction, args.split_seed)
    else:
        test = []
    if args.protocol == "online" and not test:
        raise ValueError("--protocol online needs a held-out split (--test-tasks or at least 2 tasks)")
    root = Path(args.root)
    cache_dir = Path(args.cache_dir) if args.cache_dir else root.parent / ".meta-harness-cache"
    model = CachedModel(model_from_environment(args.provider, args.model), cache_dir,
                        enabled=not args.no_cache)
    metric = BUILTIN_METRICS[args.metric] if args.metric else METRIC_FOR[args.task_type]
    online = args.protocol == "online"
    scored = test if online else search
    rows = []
    previous = os.environ.get(key)
    try:
        for index, value in enumerate(values):
            os.environ[key] = value
            evaluator = CandidateEvaluator(model, metric, task_timeout=args.task_timeout,
                                           protocol=args.protocol, train_tasks=search if online else ())
            directory = root / _setting_dir(index, value)
            result = evaluator.evaluate(Path(args.harness), scored, f"{key}={value}",
                                        directory / "trace.jsonl")
            rows.append({"setting": value, "env": {key: value}, "score": result.score,
                         "context_cost": result.context_cost, "examples": result.examples,
                         "valid": result.valid, "error": result.error,
                         "elapsed_seconds": result.elapsed_seconds})
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    payload = {"harness": str(args.harness), "variable": key, "protocol": args.protocol,
               "split": "test" if online else "search", "rows": rows}
    root.mkdir(parents=True, exist_ok=True)
    atomic_write(root / "sweep.json", json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0
