"""`meta-harness inspect`: show stored results and the Pareto frontier."""

from __future__ import annotations

import json

from ..core import FilesystemExperience

NAME = "inspect"
HELP = "show stored results and Pareto frontier"


def add_arguments(parser) -> None:
    parser.add_argument("root", nargs="?", default=".meta-harness")


def run(args) -> int:
    experience = FilesystemExperience(args.root)
    frontier_file = experience.root / "frontier.json"
    print(json.dumps({
        "results": experience.manifest(),
        "frontier": json.loads(frontier_file.read_text(encoding="utf-8")) if frontier_file.exists() else [],
    }, indent=2))
    return 0
