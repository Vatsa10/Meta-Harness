from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Sequence

from .agent_proposer import ClaudeCodeProposer
from .cache import CachedModel
from .core import (CandidateEvaluator, CommandProposer, FilesystemExperience, ParetoFrontier,
                   SearchConfig, SearchRunner)
from .datasets import classification_tasks, math_tasks, read_records, split_tasks, terminal_tasks
from .demo import run as run_demo
from .metrics import METRICS
from .providers import list_unikey_models, model_from_environment

BASELINE_ROOT = Path(__file__).resolve().parent.parent / "baselines"
DEFAULT_BASELINES = {
    "classification": ["zero_shot.py", "few_shot.py", "ace.py", "mce.py"],
    "math": ["math_zero_shot.py", "math_bm25.py"],
    "terminal": ["terminal_basic.py"],
}
ADAPTERS = {"classification": classification_tasks, "math": math_tasks, "terminal": terminal_tasks}


def default_baselines(task_type: str) -> list[Path]:
    return [BASELINE_ROOT / name for name in DEFAULT_BASELINES[task_type]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meta-harness")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a deterministic end-to-end example")
    demo.add_argument("--iterations", type=int, default=3)
    demo.add_argument("--root", default=".meta-harness-demo")

    inspect = sub.add_parser("inspect", help="show stored results and Pareto frontier")
    inspect.add_argument("root", nargs="?", default=".meta-harness")

    models = sub.add_parser("models", help="list models offered by the gateway")
    models.add_argument("--provider", choices=["unikey"], default="unikey")

    run = sub.add_parser("run", help="run a provider-backed harness search")
    run.add_argument("--tasks", required=True, help="JSON, JSONL, or CSV dataset")
    run.add_argument("--test-tasks", help="held-out dataset; omit to split --tasks")
    run.add_argument("--search-fraction", type=float, default=0.7)
    run.add_argument("--split-seed", type=int, default=0)
    run.add_argument("--baseline", nargs="+", help="seed harness files (default: bundled baselines)")
    run.add_argument("--provider", choices=["unikey", "openai", "anthropic", "compatible"], default="unikey")
    run.add_argument("--model", required=True)
    run.add_argument("--task-type", choices=["classification", "math", "terminal"], default="classification")
    run.add_argument("--root", default=".meta-harness")
    run.add_argument("--test-root")
    run.add_argument("--iterations", type=int, default=10)
    run.add_argument("--candidates", type=int, default=1)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--max-workers", type=int, default=1)
    run.add_argument("--proposer-command", help="shell command writing .py files to META_HARNESS_OUTPUT")
    run.add_argument("--proposer-model", help="model id passed to the coding agent")
    run.add_argument("--proposer-binary", default="claude")
    run.add_argument("--proposer-view", choices=["scores", "summary", "full"], default="full")
    run.add_argument("--proposer-timeout", type=float, default=1800.0)
    run.add_argument("--cache-dir")
    run.add_argument("--no-cache", action="store_true")
    run.add_argument("--validation-timeout", type=float, default=20.0)
    run.add_argument("--task-timeout", type=float, default=600.0)
    run.add_argument("--allow-local-shell", action="store_true",
                     help="terminal tasks only: run model-authored commands on this machine")
    return parser


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


def _command_run(args) -> int:
    if args.allow_local_shell:
        os.environ["META_HARNESS_ALLOW_LOCAL_SHELL"] = "1"
    search_tasks, test_tasks = _load_tasks(args)
    base_model = model_from_environment(args.provider, args.model)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.root).parent / ".meta-harness-cache"
    model = CachedModel(base_model, cache_dir, enabled=not args.no_cache)
    if args.proposer_command:
        proposer = CommandProposer(resolve_command(split_command(args.proposer_command)),
                                   timeout=args.proposer_timeout)
    else:
        proposer = ClaudeCodeProposer(binary=args.proposer_binary, model=args.proposer_model,
                                      view=args.proposer_view, summarizer=model,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "demo":
        root = run_demo(args.iterations, args.root)
        print(json.dumps([r.__dict__ for r in ParetoFrontier.select(FilesystemExperience(root).results())], indent=2))
        return 0
    if args.command == "models":
        print("\n".join(list_unikey_models()))
        return 0
    if args.command == "run":
        return _command_run(args)
    experience = FilesystemExperience(args.root)
    frontier_file = experience.root / "frontier.json"
    print(json.dumps({
        "results": experience.manifest(),
        "frontier": json.loads(frontier_file.read_text(encoding="utf-8")) if frontier_file.exists() else [],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
