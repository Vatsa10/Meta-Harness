"""TTT-Discover-style optimizer: the tree-search half only.

The test-time weight-training half of TTT-Discover (updating model weights on the search signal) is
not reproducible via hosted APIs, so only the PUCT tree search over candidates is implemented here.

Nodes are candidates; parent.json lineage defines the tree and candidates without a parent (the
seeds) are roots under a virtual root. Selection descends from the virtual root, at each level taking the sibling that maximises
    Q(n) + c * P(n) * sqrt(N(parent)) / (1 + N(n))
(stopping at a leaf or an unvisited node), with Q the best score in n's subtree normalised to [0, 1] across the tree, P uniform
over siblings and c = 1.0. Visit counts N (a selection counts for the node and its ancestors) persist
in proposals/ttt_state.json. The selected node is expanded by one text-model mutation conditioned on
its source, its score and its best child's source. Reads only experience.results(), parent.json
lineage and harness.py.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from . import extract_python

NAME = "ttt_discover"
C_PUCT = 1.0
ROOT = ""
STATE_FILE = "ttt_state.json"

PROMPT = """You are improving a Python harness for the domain: {domain}.
{notes}
Selected program (id {node_id}), score {score:.4f}:
```python
{source}
```
{child}
Write one improved version of the selected program as a single fenced ```python block, keeping the
same interface. Briefly explain the mutation before the code block.
"""


def load_state(root: Path) -> dict[str, Any]:
    try:
        data = json.loads((Path(root) / "proposals" / STATE_FILE).read_text(encoding="utf-8"))
        return {"visits": {str(k): int(v) for k, v in dict(data.get("visits", {})).items()}}
    except (OSError, ValueError, TypeError, AttributeError):
        return {"visits": {}}


def save_state(root: Path, state: dict[str, Any]) -> None:
    path = Path(root) / "proposals" / STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def parent_map(root: Path, ids: list[str]) -> dict[str, str]:
    """node id -> parent id (ROOT for seeds and for nodes whose parent is not a scored candidate)."""
    parents: dict[str, str] = {}
    for node in ids:
        parent = ROOT
        try:
            data = json.loads((Path(root) / "candidates" / node / "parent.json").read_text(encoding="utf-8"))
            listed = [str(p) for p in data.get("parents", [])]
            if listed and listed[0] in ids and listed[0] != node:
                parent = listed[0]
        except (OSError, ValueError, AttributeError):
            pass
        parents[node] = parent
    return parents


def puct_scores(results: list[Any], parents: dict[str, str], visits: dict[str, int]) -> dict[str, float]:
    score = {r.candidate_id: r.score for r in results}
    lo, hi = min(score.values()), max(score.values())

    def norm(value: float) -> float:
        return 0.5 if hi == lo else (value - lo) / (hi - lo)

    children: dict[str, list[str]] = {}
    for node, parent in parents.items():
        children.setdefault(parent, []).append(node)
    best: dict[str, float] = {}

    def subtree_best(node: str, seen: frozenset = frozenset()) -> float:
        if node not in best:
            best[node] = max([score[node], *(subtree_best(c, seen | {node})
                                             for c in children.get(node, []) if c not in seen)])
        return best[node]

    total_roots = sum(visits.get(n, 0) for n in children.get(ROOT, []))
    values: dict[str, float] = {}
    for node, parent in parents.items():
        n_parent = total_roots if parent == ROOT else visits.get(parent, 0)
        prior = 1.0 / len(children[parent])
        values[node] = (norm(subtree_best(node))
                        + C_PUCT * prior * math.sqrt(n_parent) / (1 + visits.get(node, 0)))
    return values


def select(results: list[Any], parents: dict[str, str], visits: dict[str, int]) -> str:
    """Descend from the virtual root by PUCT among siblings; stop at a leaf or an unvisited node."""
    values = puct_scores(results, parents, visits)
    children: dict[str, list[str]] = {}
    for node, parent in parents.items():
        children.setdefault(parent, []).append(node)
    node = ROOT
    while True:
        options = children.get(node)
        if not options:
            return node
        node = max(sorted(options), key=lambda n: values[n])  # ties: first by id
        if not visits.get(node, 0) or node not in children:
            return node


def _ancestors(node: str, parents: dict[str, str]) -> list[str]:
    path: list[str] = []
    while node != ROOT and node not in path:
        path.append(node)
        node = parents.get(node, ROOT)
    return path


class TTTDiscover:
    def __init__(self, model: Any, *, domain: str = "", notes: str = ""):
        self.model, self.domain, self.notes = model, domain, notes

    @staticmethod
    def _source(root: Path, node: str) -> str:
        return (root / "candidates" / node / "harness.py").read_text(encoding="utf-8")

    def __call__(self, experience: Any, iteration: int, count: int) -> list[Path]:
        root = Path(experience.root)
        output = root / "proposals" / f"iteration-{iteration:04d}"
        output.mkdir(parents=True, exist_ok=True)
        results = [r for r in experience.results() if r.valid]
        if not results:
            return []
        by_id = {r.candidate_id: r for r in results}
        parents = parent_map(root, list(by_id))
        state = load_state(root)
        visits = state["visits"]
        paths: list[Path] = []
        for index in range(count):
            node = select(results, parents, visits)
            for member in _ancestors(node, parents):
                visits[member] = visits.get(member, 0) + 1
            kids = [c for c, p in parents.items() if p == node]
            top = max(kids, key=lambda c: (by_id[c].score, c), default=None)
            child = (f"Its best child (id {top}, score {by_id[top].score:.4f}):\n"
                     f"```python\n{self._source(root, top)}\n```\n") if top else ""
            prompt = PROMPT.format(domain=self.domain, notes=self.notes, node_id=node,
                                   score=by_id[node].score, source=self._source(root, node), child=child)
            text = str(self.model(prompt))
            try:
                source = extract_python(text)
            except ValueError:
                continue
            stem = f"candidate-{index:02d}"
            (output / f"{stem}.py").write_text(source, encoding="utf-8")
            (output / f"{stem}.reasoning.md").write_text(text, encoding="utf-8")
            (output / f"{stem}.parent.json").write_text(
                json.dumps({"parents": [node], "operator": "puct-mutation"}, indent=2), encoding="utf-8")
            paths.append(output / f"{stem}.py")
        save_state(root, state)
        return paths


def build(ctx):
    return TTTDiscover(ctx.proposer_model, domain=ctx.domain.NAME, notes=getattr(ctx.domain, "PROMPT", ""))
