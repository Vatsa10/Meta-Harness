"""`meta-harness demo`: a deterministic end-to-end example."""

from __future__ import annotations

import json

from ..core import FilesystemExperience, ParetoFrontier
from ..demo import run as run_demo

NAME = "demo"
HELP = "run a deterministic end-to-end example"


def add_arguments(parser) -> None:
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--root", default=".meta-harness-demo")


def run(args) -> int:
    root = run_demo(args.iterations, args.root)
    print(json.dumps([r.__dict__ for r in ParetoFrontier.select(FilesystemExperience(root).results())],
                     indent=2))
    return 0
