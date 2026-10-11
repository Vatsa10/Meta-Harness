"""`meta-harness run`: a provider-backed harness search (replaced by `search` later)."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Sequence

from ..cache import CachedModel
from ..core import CandidateEvaluator, CommandProposer, SearchConfig, SearchRunner
from ..datasets import classification_tasks, math_tasks, read_records, split_tasks, terminal_tasks
from ..metrics import METRICS
from ..providers import model_from_environment

NAME = "run"
HELP = "run a provider-backed harness search"

BASELINE_ROOT = Path(__file__).resolve().parent.parent.parent / "baselines"
DEFAULT_BASELINES = {
    "classification": ["classification/zero_shot.py", "classification/few_shot.py",
                       "classification/ace.py", "classification/mce.py"],
    "math": ["math/zero_shot.py", "math/bm25.py"],
    "terminal": ["terminal/basic.py"],
}
ADAPTERS = {"classification": classification_tasks, "math": math_tasks,
            "terminal": terminal_tasks}


def default_baselines(task_type: str) -> list[Path]:
    return [BASELINE_ROOT / name for name in DEFAULT_BASELINES[task_type]]


def add_arguments(run) -> None:
    run.add_argument("--tasks", required=True, help="JSON, JSONL, or CSV dataset")
    run.add_argument("--test-tasks", help="held-out dataset; omit to split --tasks")
    run.add_argument("--search-fraction", type=float, default=0.7)
    run.add_argument("--split-seed", type=int, default=0)
    run.add_argument("--baseline", nargs="+", help="seed harness files (default: bundled baselines)")
    run.add_argument("--provider", choices=["unikey", "openai", "anthropic", "compatible"],
                     default="unikey", help="where the harness model runs")
    run.add_argument("--model", required=True, help="harness model id")
    run.add_argument("--task-type", choices=["classification", "math", "terminal"],
                     default="classification")
    run.add_argument("--root", default=".meta-harness")
    run.add_argument("--test-root")
    run.add_argument("--iterations", type=int, default=10)
    run.add_argument("--candidates", type=int, default=1)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--max-workers", type=int, default=1)
    run.add_argument("--proposer-command", required=True,
                     help="shell command writing .py files to META_HARNESS_OUTPUT")
    run.add_argument("--proposer-timeout", type=float, default=1800.0)
    run.add_argument("--cache-dir")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--validation-timeout", type=float, default=20.0)
    run.add_argument("--task-timeout", type=float, default=600.0)


def split_command(command: str) -> list[str]:
    """Split a shell-ish command string. On Windows, backslashes are path separators,
    not escapes, so posix-mode shlex would mangle every absolute path."""
    if os.name != "nt":
        return shlex.split(command)
    return [token[1:-1] if len(token) > 1 and token[0] == token[-1] == '"' else token
            for token in shlex.split(command, posix=False)]


def resolve_command(tokens: Sequence[str]) -> list[str]:
    """Absolutize tokens that name a real file.

    The proposer runs with its working directory set to the experience root, so a relative
    script path in --proposer-command would resolve against the wrong directory.
    """
    return [str(Path(token).resolve()) if token and Path(token).exists() else token
            for token in tokens]


def _load_tasks(args) -> tuple[list[dict], list[dict]]:
    adapt = ADAPTERS[args.task_type]
    search = adapt(read_records(args.tasks))
    if args.test_tasks:
        return search, adapt(read_records(args.test_tasks))
    if len(search) < 2:
        return search, []
    return split_tasks(search, args.search_fraction, args.split_seed)


def run(args) -> int:
    search_tasks, test_tasks = _load_tasks(args)
    base_model = model_from_environment(args.provider, args.model)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.root).parent / ".meta-harness-cache"
    model = CachedModel(base_model, cache_dir, enabled=not args.no_cache)
    proposer = CommandProposer(resolve_command(split_command(args.proposer_command)),
                               timeout=args.proposer_timeout)
    config = SearchConfig(root=args.root, iterations=args.iterations,
                          candidates_per_iteration=args.candidates,
                          validation_timeout=args.validation_timeout,
                          repeats=args.repeats, max_workers=args.max_workers,
                          task_timeout=args.task_timeout, test_root=args.test_root)
    evaluator = CandidateEvaluator(model, METRICS[args.task_type], task_timeout=args.task_timeout)
    runner = SearchRunner(config, evaluator, proposer)
    baselines = [Path(p) for p in (args.baseline or default_baselines(args.task_type))]
    frontier = runner.run(baselines, search_tasks, test_tasks or None)
    test_path = Path(config.test_root) / "test_results.json"
    print(json.dumps({
        "root": str(config.root),
        "search_tasks": len(search_tasks),
        "test_tasks": len(test_tasks),
        "cache": {"hits": model.hits, "misses": model.misses},
        "frontier": [item.__dict__ for item in frontier],
        "test_results": str(test_path) if test_path.is_file() else None,
    }, indent=2))
    return 0
