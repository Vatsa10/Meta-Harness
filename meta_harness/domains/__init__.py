"""Task domains.

Domain module contract:
    NAME: str
    PROMPT: str                      domain notes appended to the proposer prompt
    METRICS: dict[str, Callable[[Mapping, Mapping], float]]
    SEED_DIR: Path | None
    PROTOCOL: "plain" | "online"
    def adapt(records: list[dict]) -> list[dict]
"""

from __future__ import annotations

from types import ModuleType

from ..registry import discover


def all_domains() -> dict[str, ModuleType]:
    return discover("meta_harness.domains")


def get_domain(name: str) -> ModuleType:
    domains = all_domains()
    if name not in domains:
        raise KeyError(f"unknown domain {name!r}; known: {', '.join(sorted(domains))}")
    return domains[name]
