"""Plugin discovery: every non-underscore module of a package that declares NAME."""

from __future__ import annotations

import importlib
import pkgutil
from types import ModuleType


def _modules(package: str) -> list[ModuleType]:
    pkg = importlib.import_module(package)
    found = []
    for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        # A module that fails to import raises: a broken plugin must not silently vanish.
        found.append(importlib.import_module(f"{package}.{info.name}"))
    return found


def discover(package: str) -> dict[str, ModuleType]:
    """Import every non-underscore module of `package`; key by its NAME attribute.

    Duplicate NAMEs raise RuntimeError naming both modules.
    """
    registry: dict[str, ModuleType] = {}
    for module in _modules(package):
        name = getattr(module, "NAME", None)
        if name is None:
            continue
        if name in registry:
            raise RuntimeError(f"duplicate NAME {name!r} in {registry[name].__name__} "
                               f"and {module.__name__}")
        registry[name] = module
    return registry
