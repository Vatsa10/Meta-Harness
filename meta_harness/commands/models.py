"""`meta-harness models`: list available harness models."""

from __future__ import annotations

from ..providers import list_unikey_models

NAME = "models"
HELP = "list available harness models"


def add_arguments(parser) -> None:
    parser.add_argument("--provider", choices=["unikey"], default="unikey")


def run(args) -> int:
    print("\n".join(list_unikey_models()))
    return 0
