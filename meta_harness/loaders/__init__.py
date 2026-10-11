"""Dataset loaders.

Loader contract:
    NAME: str; DOMAIN: str; SOURCE: str (URL, documented)
    def fetch(cache_dir: Path, http: Http = urllib_get) -> dict[str, Path]   split name -> JSONL
    Uses the cache when every split file exists. Rows are already domain tasks.
"""

from __future__ import annotations

from types import ModuleType

from ..registry import discover


def all_loaders() -> dict[str, ModuleType]:
    return discover("meta_harness.loaders")
