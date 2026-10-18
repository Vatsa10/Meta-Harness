"""Refuse to execute real tools when the active environment looks like production.

`check(env, production_markers)` raises `ProductionEnvironmentError` when any of these hold:

* ``ENV``, ``APP_ENV``, ``NODE_ENV`` (and a few common aliases) is ``prod`` or ``production``;
* an env-file path variable (``ENV_FILE``, ``DOTENV_PATH``, ``*_ENV_FILE`` ...) names a file whose
  name contains ``prod``;
* a URL, host, DSN or database-name variable contains one of the user-declared production markers.

Markers come from the ``[production]`` section of ``meta-harness.toml``::

    [production]
    markers = ["db.prod.internal", "api.example.com"]
    hosts = ["10.0.4.12"]          # every list in the section is a marker list
    databases = ["orders_main"]

Only stdlib is used. The error names the offending variable and marker, never the variable's value,
so secrets in DSNs do not leak into logs.

Replay mode never executes a tool, so it does not need this guard; it is still meant to run with no
network inside isolation (the docker/no-network wiring belongs to the isolation layer, not here).
"""
from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Iterable, Mapping

SETTINGS_FILE = "meta-harness.toml"
ENV_NAME_KEYS = ("ENV", "APP_ENV", "NODE_ENV", "ENVIRONMENT", "PYTHON_ENV", "RAILS_ENV", "DJANGO_ENV",
                 "FLASK_ENV", "DEPLOY_ENV", "STAGE")
PROD_VALUES = {"prod", "production"}
_ENV_FILE_KEY = re.compile(r"(^|_)(ENV_?FILE|DOTENV(_PATH|_FILE)?|ENV_PATH)$", re.I)
_TARGET_KEY = re.compile(r"(URL|URI|HOST|HOSTNAME|DSN|ENDPOINT|DB|DATABASE|DB_NAME|SERVER|ADDR|ADDRESS)", re.I)
_URLISH = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)


class ProductionEnvironmentError(RuntimeError):
    """Raised when tools would run for real against something that looks like production."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("refusing to run tools against production: " + "; ".join(reasons))


def load_markers(path: str | Path | None = None) -> list[str]:
    """Every string in the `[production]` section of meta-harness.toml (missing file -> [])."""
    settings = load_settings(path)
    section = settings.get("production") or {}
    markers: list[str] = []
    for value in section.values():
        items = value if isinstance(value, list) else [value]
        markers.extend(str(item) for item in items if isinstance(item, str) and item.strip())
    return markers


def load_settings(path: str | Path | None = None) -> dict:
    target = Path(path) if path is not None else Path.cwd() / SETTINGS_FILE
    if not target.is_file():
        return {}
    return tomllib.loads(target.read_text(encoding="utf-8"))


def problems(env: Mapping[str, str], production_markers: Iterable[str] = ()) -> list[str]:
    """Reasons the environment looks like production (empty list -> safe)."""
    markers = [m.strip().lower() for m in production_markers if m and m.strip()]
    reasons: list[str] = []
    for key, raw in env.items():
        value = "" if raw is None else str(raw)
        upper = str(key).upper()
        if upper in ENV_NAME_KEYS and value.strip().lower() in PROD_VALUES:
            reasons.append(f"{key} is set to production")
            continue
        if _ENV_FILE_KEY.search(upper) and "prod" in Path(value.strip()).name.lower():
            reasons.append(f"{key} points at a production env file")
            continue
        if markers and (_TARGET_KEY.search(upper) or _URLISH.match(value.strip())):
            lowered = value.lower()
            hit = next((m for m in markers if m in lowered), None)
            if hit is not None:
                reasons.append(f"{key} matches production marker {hit!r}")
    return reasons


def check(env: Mapping[str, str] | None = None, production_markers: Iterable[str] | None = None) -> None:
    """Raise ProductionEnvironmentError if `env` (default os.environ) looks like production.

    `production_markers` defaults to the markers in ./meta-harness.toml.
    """
    env = os.environ if env is None else env
    markers = load_markers() if production_markers is None else production_markers
    reasons = problems(env, markers)
    if reasons:
        raise ProductionEnvironmentError(reasons)
