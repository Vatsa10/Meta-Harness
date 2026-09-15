from __future__ import annotations

import argparse
import json
from pathlib import Path

from .demo import run
from .core import FilesystemExperience, ParetoFrontier


def main() -> None:
    parser = argparse.ArgumentParser(prog="meta-harness")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a deterministic end-to-end example")
    demo.add_argument("--iterations", type=int, default=3)
    demo.add_argument("--root", default=".meta-harness-demo")
    inspect = sub.add_parser("inspect", help="show stored results and Pareto frontier")
    inspect.add_argument("root", nargs="?", default=".meta-harness")
    args = parser.parse_args()
    if args.command == "demo":
        root = run(args.iterations, args.root)
        print(json.dumps([r.__dict__ for r in ParetoFrontier.select(FilesystemExperience(root).results())], indent=2))
    else:
        experience = FilesystemExperience(args.root)
        print(json.dumps({"results": experience.manifest(), "frontier": json.loads((experience.root / "frontier.json").read_text()) if (experience.root / "frontier.json").exists() else []}, indent=2))


if __name__ == "__main__":
    main()
