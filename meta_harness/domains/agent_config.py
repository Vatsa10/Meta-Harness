"""Config-only search over a production agent wrapped by `meta-harness adapt init`.

The seed is a generated config harness (meta_harness/adapters/config_harness.py). Candidates may
change only the JSON between `# BEGIN CONFIG` and `# END CONFIG` in harness.py (it is the
agent's config.json). The gate (VALIDATE) refuses any other byte of change and any config value
outside the schema embedded in the seed (types, enums, bounds, max lengths, required and frozen
keys).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..adapters.config_harness import ConfigHarnessError, parse_harness, validate_candidate

NAME = "agent_config"
PROTOCOL = "plain"
SEED_DIR = None
METRICS: dict = {}
PROMPT = """Agent config domain (config-only search). Every harness is a GENERATED config
harness that calls the user's production agent; you may not change its code. Copy the seed's
harness.py byte for byte and change ONLY the JSON between the lines `# BEGIN CONFIG` and
`# END CONFIG` - that block is the agent's config.json (prompts, tool descriptions, few-shot
examples, routing thresholds, retrieval k, memory policy). The gate rejects any other change
(new imports, edited functions, a changed ENTRYPOINT or SCHEMA_JSON) and any config value that
violates SCHEMA_JSON: wrong type, value outside an enum, number outside minimum/maximum, string
longer than maxLength, missing required key, unknown key, or a changed `frozen` key. Read the
traces to see which config knob caused each failure, and keep the JSON valid."""


def adapt(records: list[dict]) -> list[dict]:
    return [dict(r) for r in records]


def _seed_dirs(candidate_dir: Path) -> list[Path]:
    """The search's seeds: the `initial-*` candidates beside this one in the experience store."""
    return sorted(p.parent for p in candidate_dir.parent.glob("initial-*/harness.py"))


def validate(source: str | Path, tasks: Sequence[Mapping[str, Any]] = ()) -> None:
    """Gate hook (`source_validator(source, tasks)`): raise ConfigHarnessError unless the
    candidate at `source` (candidates/<id>/harness.py) is one of the search's seeds with only
    its config changed, and that config satisfies the seed's schema."""
    source = Path(source)
    candidate_dir = source.parent
    candidate = parse_harness(source.read_text(encoding="utf-8"))
    seeds = [d for d in _seed_dirs(candidate_dir) if d != candidate_dir]
    if candidate_dir.name.startswith("initial-") or not seeds:
        # A seed (or a harness outside an experience store) is checked against itself: canonical
        # template, config valid for its own schema.
        validate_candidate(candidate_dir, candidate_dir)
        return
    match = seeds[0]  # no seed has the same code: the gate names the first difference against it
    for seed_dir in seeds:
        try:
            seed = parse_harness((seed_dir / "harness.py").read_text(encoding="utf-8"))
        except ConfigHarnessError:
            continue  # a non-adapter seed in a mixed search never matches
        if seed["fixed"] == candidate["fixed"]:
            match = seed_dir
            break
    validate_candidate(candidate_dir, match)


VALIDATE = validate
