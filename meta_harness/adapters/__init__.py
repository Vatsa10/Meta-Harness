"""Agent adapters: search the config of a production agent instead of harness code.

See `config_harness` (the runtime, scaffolding and config-only gate), `schema` (the config
schema subset) and docs/adapters.md (wiring for LangGraph, the OpenAI Agents SDK and the Claude
Agent SDK)."""

from __future__ import annotations

from .config_harness import (ConfigHarnessError, init_seed, parse_harness, render_harness,
                             sync_seed, validate_candidate)
from .schema import SchemaError, validate_config

__all__ = ["ConfigHarnessError", "SchemaError", "init_seed", "parse_harness", "render_harness",
           "sync_seed", "validate_candidate", "validate_config"]
