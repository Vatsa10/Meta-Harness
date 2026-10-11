"""`meta-harness ood`: evaluate frozen harnesses on the nine out-of-distribution sets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..core import CandidateEvaluator
from ..httpdata import CACHE_ROOT, read_jsonl
from ..loaders import all_loaders
from ..loaders.ood import OOD_SPECS, fetch_one
from ..protocol import exact
from ..providers import model_from_environment

NAME = "ood"
HELP = "evaluate harnesses frozen on the nine out-of-distribution classification sets"


def add_arguments(parser) -> None:
    parser.add_argument("--harness", action="append", required=True, metavar="FILE",
                        help="harness file; repeat for several")
    parser.add_argument("--datasets", default="all",
                        help="comma-separated names from: " + ", ".join(OOD_SPECS) + ", or 'all'")
    parser.add_argument("--provider", default="unikey",
                        help="unikey, openai, anthropic, compatible, groq, or scripted")
    parser.add_argument("--model", required=True, help="model id (script file for scripted)")
    parser.add_argument("--train-from", metavar="LOADER",
                        help="stream this loader's train split to each harness first (default: zero-shot)")
    parser.add_argument("--train-limit", type=int, default=50,
                        help="train tasks streamed per harness (owner-scale default 50)")
    parser.add_argument("--limit", type=int, help="tasks per dataset (default: the spec's 1000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(CACHE_ROOT))
    parser.add_argument("--out", default=".meta-harness/ood", help="directory receiving ood.json")
    parser.add_argument("--task-timeout", type=float, default=600.0)


def _datasets(text: str) -> list[str]:
    names = list(OOD_SPECS) if text.strip() == "all" else [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in OOD_SPECS]
    if unknown or not names:
        raise ValueError(f"unknown dataset(s) {unknown or text!r}; known: {', '.join(OOD_SPECS)}")
    return names


def _train_tasks(args, cache: Path) -> list[dict]:
    if not args.train_from:
        return []
    loaders = all_loaders()
    if args.train_from not in loaders:
        raise ValueError(f"unknown loader {args.train_from!r}; known: {', '.join(sorted(loaders))}")
    splits = loaders[args.train_from].fetch(cache / args.train_from)
    if "train" not in splits:
        raise ValueError(f"loader {args.train_from!r} has no train split")
    return read_jsonl(splits["train"])[:max(0, args.train_limit)]


def format_table(rows: list[dict]) -> str:
    header = ("harness", "dataset", "accuracy", "context_tokens", "n")
    body = [(r["harness"], r["dataset"], f"{r['accuracy']:.3f}", f"{r['context_tokens']:.0f}", str(r["n"]))
            for r in rows]
    widths = [max(len(line[i]) for line in [header, *body]) for i in range(len(header))]
    return "\n".join("  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip()
                     for line in [header, *body])


def run(args) -> int:
    try:
        names = _datasets(args.datasets)
        harnesses = [Path(h) for h in args.harness]
        missing = [str(h) for h in harnesses if not h.is_file()]
        if missing:
            raise ValueError(f"harness file not found: {', '.join(missing)}")
        cache = Path(args.cache_dir)
        train = _train_tasks(args, cache)
        data = {name: read_jsonl(fetch_one(name, cache / "ood", limit=args.limit, seed=args.seed))
                for name in names}
        model = model_from_environment(args.provider, args.model)
    except (ValueError, KeyError, OSError) as error:
        print(f"ood: {error}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    evaluator = CandidateEvaluator(model, exact, task_timeout=args.task_timeout,
                                   protocol="online", train_tasks=train)
    rows = []
    for harness in harnesses:
        for name in names:
            trace = out / "traces" / f"{harness.stem}-{name}.jsonl"
            result = evaluator.evaluate(harness, data[name], f"{harness.stem}-{name}", trace)
            rows.append({"harness": harness.stem, "dataset": name, "accuracy": result.score if result.valid else 0.0,
                         "context_tokens": result.context_cost if result.valid else 0.0, "n": result.examples,
                         "train_n": len(train), "train_prompt_tokens": result.metrics.get("train_prompt_tokens", 0),
                         "valid": result.valid, "error": result.error})
    payload = {"provider": args.provider, "train_from": args.train_from, "seed": args.seed,
               "protocol": "online" if train else "frozen zero-shot", "rows": rows}
    (out / "ood.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(format_table(rows))
    return 0
