"""`meta-harness fetch <loader>`: download a paper dataset into data/cache/<name>."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..httpdata import CACHE_ROOT
from ..loaders import all_loaders

NAME = "fetch"
HELP = "download a paper dataset into data/cache/<name>"


def add_arguments(parser) -> None:
    parser.add_argument("loader", help="dataset loader name")
    parser.add_argument("--dir", help="cache root (default: data/cache); files go to <dir>/<name>")


def run(args) -> int:
    loaders = all_loaders()
    if args.loader not in loaders:
        print(f"unknown loader {args.loader!r}; known: {', '.join(sorted(loaders))}", file=sys.stderr)
        return 2
    root = Path(args.dir) if args.dir else CACHE_ROOT
    paths = loaders[args.loader].fetch(root / args.loader)
    print(json.dumps({split: str(path) for split, path in paths.items()}, ensure_ascii=False))
    return 0
