"""`meta-harness bench`: resumable arm x task x trial benchmark matrix."""

from __future__ import annotations

import json
from pathlib import Path

from ..bench import Arm, run_matrix
from ..datasets import read_records
from ..protocol import resolve_metric

NAME = "bench"
HELP = "run a resumable arm x task x trial benchmark matrix"


def parse_arm(text: str) -> Arm:
    name, sep, rest = text.partition("=")
    harness, at, target = rest.rpartition("@")
    provider, colon, model = target.partition(":")
    if not (sep and at and colon and name and harness and provider and model):
        raise ValueError(f"--arm must look like name=harness.py@provider:model, got {text!r}")
    return Arm(name, Path(harness), provider, model)


def add_arguments(parser) -> None:
    parser.add_argument("--arm", action="append", required=True, help="name=harness.py@provider:model")
    parser.add_argument("--data", required=True, help="JSON, JSONL, or CSV tasks")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--metric", default="exact", help="exact|contains|numeric or module:function")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true", help="continue an existing matrix in --out")
    parser.add_argument("--cell-timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=2)


def run(args) -> int:
    out = Path(args.out)
    matrix = out / "matrix.jsonl"
    if not args.resume and matrix.is_file() and matrix.stat().st_size:
        print(f"{matrix} exists; pass --resume to continue it or choose a new --out")
        return 2
    try:
        arms = [parse_arm(a) for a in args.arm]
        metric = resolve_metric(args.metric)
    except ValueError as exc:
        print(str(exc))
        return 2
    path = run_matrix(arms, read_records(args.data), args.trials, metric, out,
                      cell_timeout=args.cell_timeout, retries=args.retries)
    print(json.dumps({"matrix": str(path), "summary": str(out / "summary.json")}, indent=2))
    return 0
