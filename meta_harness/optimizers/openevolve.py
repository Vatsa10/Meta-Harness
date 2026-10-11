"""OpenEvolve-style optimizer: a MAP-Elites archive over islands, rewritten by a text-only model.

The archive is keyed by (score decile, context-cost decile) and keeps one elite per cell. Three
islands hold subsets of it (candidate id -> island by a stable hash); every MIGRATION_EVERY
iterations the best member of each island is copied to the next island. A parent is drawn with
p=EXPLOIT from the current island's elites (weighted by score), otherwise uniformly from the whole
archive. The model sees the parent, its scores and two inspiration programs and rewrites the whole
harness. It reads only experience.results() and each candidate's harness.py.
"""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path
from typing import Any

from . import extract_python

NAME = "openevolve"
ISLANDS = 3
MIGRATION_EVERY = 5
EXPLOIT = 0.7
INSPIRATIONS = 2
STATE_FILE = "openevolve_state.json"

PROMPT = """You are evolving a Python harness for the domain: {domain}.
{notes}
Parent program (id {parent_id}): score {parent_score:.4f}, context cost {parent_cost:.1f}.
```python
{parent_source}
```

Inspiration programs (strong earlier candidates):
{inspirations}

Write a complete improved rewrite of the parent harness as one fenced ```python block. Keep the
same interface. Briefly explain the change before the code block.
"""


def island_of(candidate_id: str) -> int:
    return zlib.crc32(candidate_id.encode("utf-8")) % ISLANDS


def _decile(value: float, top: float) -> int:
    if top <= 0:
        return 0
    return max(0, min(9, int(10 * value / top)))


def build_archive(results: list[Any], members: set[str] | None = None,
                  bounds: tuple[float, float, float] | None = None) -> dict[tuple[int, int], Any]:
    """One elite per (score decile, cost decile) cell. Deciles use the score normalised over all
    valid results and are clamped to [0, 9]; `members` restricts the archive to one island."""
    valid = [r for r in results if r.valid]
    lo, hi, max_cost = bounds or (min((r.score for r in valid), default=0.0),
                                  max((r.score for r in valid), default=0.0),
                                  max((r.context_cost for r in valid), default=0.0))
    archive: dict[tuple[int, int], Any] = {}
    for r in sorted(valid, key=lambda r: r.candidate_id):
        if members is not None and r.candidate_id not in members:
            continue
        norm = 0.0 if hi <= lo else (r.score - lo) / (hi - lo)
        cell = (max(0, min(9, int(norm * 10))), _decile(r.context_cost, max_cost))
        if cell not in archive or r.score > archive[cell].score:
            archive[cell] = r
    return archive


def sample_parent(rng: random.Random, archives: dict[int, dict[tuple[int, int], Any]],
                  island: int) -> tuple[Any, str]:
    """(parent, operator). Exploit draws from the island's own archive weighted by score; explore
    draws uniformly from every archive. An exploit draw on an empty island is recorded as
    'exploit_fallback_explore'."""
    local = sorted(archives.get(island, {}).values(), key=lambda r: r.candidate_id)
    everyone = sorted({r.candidate_id: r for a in archives.values() for r in a.values()}.values(),
                      key=lambda r: r.candidate_id)
    if rng.random() < EXPLOIT:
        if local:
            return rng.choices(local, weights=[r.score + 1e-9 for r in local], k=1)[0], "exploit"
        return rng.choice(everyone), "exploit_fallback_explore"
    return rng.choice(everyone), "explore"


class OpenEvolve:
    def __init__(self, model: Any, *, domain: str = "", notes: str = "", seed: int = 0):
        self.model, self.domain, self.notes, self.seed = model, domain, notes, seed

    def _state(self, root: Path) -> dict[str, Any]:
        path = root / "proposals" / STATE_FILE
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        data.setdefault("migrations", {})
        return data

    @staticmethod
    def members(state: dict[str, Any], ids: list[str], island: int) -> set[str]:
        found = {i for i in ids if island_of(i) == island}
        for moves in state["migrations"].values():
            found.update(i for i, target in moves.items() if target == island and i in ids)
        return found

    def _migrate(self, state: dict[str, Any], results: list[Any], iteration: int) -> None:
        key = str(iteration)
        if iteration % MIGRATION_EVERY or key in state["migrations"]:
            return
        valid = [r for r in results if r.valid]
        ids = [r.candidate_id for r in valid]
        moves: dict[str, int] = {}
        for island in range(ISLANDS):
            found = self.members(state, ids, island)
            best = max((r for r in valid if r.candidate_id in found),
                       key=lambda r: (r.score, r.candidate_id), default=None)
            if best is not None:
                moves[best.candidate_id] = (island + 1) % ISLANDS
        state["migrations"][key] = moves

    @staticmethod
    def _source(experience: Any, candidate_id: str) -> str:
        return (Path(experience.root) / "candidates" / candidate_id / "harness.py").read_text(encoding="utf-8")

    def __call__(self, experience: Any, iteration: int, count: int) -> list[Path]:
        root = Path(experience.root)
        output = root / "proposals" / f"iteration-{iteration:04d}"
        output.mkdir(parents=True, exist_ok=True)
        results = experience.results()
        valid = [r for r in results if r.valid]
        if not valid:
            return []
        state = self._state(root)
        self._migrate(state, results, iteration)
        ids = [r.candidate_id for r in valid]
        bounds = (min(r.score for r in valid), max(r.score for r in valid),
                  max(r.context_cost for r in valid))
        archives = {i: build_archive(valid, self.members(state, ids, i), bounds) for i in range(ISLANDS)}
        rng = random.Random(self.seed + iteration)
        paths: list[Path] = []
        for index in range(count):
            island = (iteration + index) % ISLANDS
            parent, operator = sample_parent(rng, archives, island)
            others = sorted((r for r in results if r.valid and r.candidate_id != parent.candidate_id),
                            key=lambda r: (-r.score, r.candidate_id))[:INSPIRATIONS]
            inspirations = "\n\n".join(
                f"Inspiration {n} (id {r.candidate_id}, score {r.score:.4f}, cost {r.context_cost:.1f}):\n"
                f"```python\n{self._source(experience, r.candidate_id)}\n```"
                for n, r in enumerate(others, start=1)) or "(none yet)"
            prompt = PROMPT.format(domain=self.domain, notes=self.notes, parent_id=parent.candidate_id,
                                   parent_score=parent.score, parent_cost=parent.context_cost,
                                   parent_source=self._source(experience, parent.candidate_id),
                                   inspirations=inspirations)
            text = str(self.model(prompt))
            try:
                source = extract_python(text)
            except ValueError:
                continue
            stem = f"candidate-{index:02d}"
            (output / f"{stem}.py").write_text(source, encoding="utf-8")
            (output / f"{stem}.reasoning.md").write_text(text, encoding="utf-8")
            (output / f"{stem}.parent.json").write_text(json.dumps(
                {"parents": [parent.candidate_id], "operator": operator, "island": island}, indent=2),
                encoding="utf-8")
            paths.append(output / f"{stem}.py")
        (root / "proposals" / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
        return paths


def build(ctx):
    return OpenEvolve(ctx.proposer_model, domain=ctx.domain.NAME,
                      notes=getattr(ctx.domain, "PROMPT", ""), seed=ctx.seed)
