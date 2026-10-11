from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .commands import all_commands
from .envfile import load_env


def _pre_parser() -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=".env",
                     help="KEY=value file loaded before dispatch (never overrides the environment)")
    return pre


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meta-harness", parents=[_pre_parser()])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, module in sorted(all_commands().items()):
        command = sub.add_parser(name, help=getattr(module, "HELP", None))
        module.add_arguments(command)
        command.set_defaults(_module=module)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    known, _ = _pre_parser().parse_known_args(argv)
    load_env(known.env_file)
    args = build_parser().parse_args(argv)
    return int(args._module.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
