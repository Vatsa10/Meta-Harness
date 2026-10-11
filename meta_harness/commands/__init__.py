"""CLI subcommands. Each module exposes NAME, HELP, add_arguments(parser) and run(args) -> int."""

from __future__ import annotations

from types import ModuleType

from ..registry import discover


def all_commands() -> dict[str, ModuleType]:
    return discover("meta_harness.commands")
