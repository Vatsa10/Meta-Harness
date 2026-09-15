from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from .demo import run
from .core import CandidateEvaluator, CommandProposer, FilesystemExperience, ParetoFrontier, SearchConfig, SearchRunner
from .datasets import classification_tasks, math_tasks, read_records
from .providers import model_from_environment


def main() -> None:
    parser = argparse.ArgumentParser(prog="meta-harness")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a deterministic end-to-end example")
    demo.add_argument("--iterations", type=int, default=3)
    demo.add_argument("--root", default=".meta-harness-demo")
    inspect = sub.add_parser("inspect", help="show stored results and Pareto frontier")
    inspect.add_argument("root", nargs="?", default=".meta-harness")
    real = sub.add_parser("run", help="run a provider-backed search")
    real.add_argument("--tasks", required=True, help="JSON, JSONL, or CSV search dataset")
    real.add_argument("--baseline", nargs="+", required=True, help="candidate Python files")
    real.add_argument("--provider", choices=["openai", "anthropic", "compatible"], required=True)
    real.add_argument("--model", required=True)
    real.add_argument("--task-type", choices=["classification", "math"], default="classification")
    real.add_argument("--root", default=".meta-harness")
    real.add_argument("--iterations", type=int, default=10)
    real.add_argument("--candidates", type=int, default=1)
    real.add_argument("--proposer-command", required=True, help="shell-like command that writes .py files to META_HARNESS_OUTPUT")
    args = parser.parse_args()
    if args.command == "demo":
        root = run(args.iterations, args.root)
        print(json.dumps([r.__dict__ for r in ParetoFrontier.select(FilesystemExperience(root).results())], indent=2))
    else:
        if args.command == "run":
            records = read_records(args.tasks)
            tasks = classification_tasks(records) if args.task_type == "classification" else math_tasks(records)
            model = model_from_environment(args.provider, args.model)
            proposer = CommandProposer(shlex.split(args.proposer_command))
            runner = SearchRunner(SearchConfig(args.root, args.iterations, args.candidates), CandidateEvaluator(model), proposer)
            frontier = runner.run(args.baseline, tasks)
            print(json.dumps([item.__dict__ for item in frontier], indent=2))
            return
        experience = FilesystemExperience(args.root)
        print(json.dumps({"results": experience.manifest(), "frontier": json.loads((experience.root / "frontier.json").read_text()) if (experience.root / "frontier.json").exists() else []}, indent=2))


if __name__ == "__main__":
    main()
