"""Math domain: problems with a final answer, scored by answer equivalence."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .. import metrics

NAME = "math"
PROTOCOL = "plain"
SEED_DIR = Path(__file__).resolve().parents[2] / "baselines" / "math"

PROMPT = """\
Domain: competition mathematics. Each task has a `problem` and an `answer`; the model should end
with the final answer in \\boxed{}. The strongest known design is a retrieval harness over a
solved-problem corpus (path in META_HARNESS_CORPUS, JSONL with problem / solution / difficulty /
source / id), using BM25 with a math-aware tokenizer that keeps LaTeX tokens atomic.
Route notes: a lexical router sends each problem to one of combinatorics (BM25@20, dedup to 8,
rerank by score and difficulty, keep 3), geometry (1 hard reference from a SEPARATE index of
NuminaMath items with difficulty > 6, plus 2 raw BM25 neighbours, no rerank), number theory
(BM25@12, rerank with a bonus when the solution states its technique early) and default
(BM25@10, rerank, adaptive K by how concentrated the top scores are).
Retrieval notes: skip corpus entries over 4000 characters, truncate inserted solutions to 3000,
and deduplicate by id.
"""


def math_equivalence(task: Mapping, result: Mapping) -> float:
    return metrics.math_equivalence(result.get("output", ""), task)


METRICS = {"math_equivalence": math_equivalence}


def adapt(records: list[dict]) -> list[dict]:
    out = []
    for index, record in enumerate(records):
        if not str(record.get("problem") or "").strip():
            raise ValueError(f"math record {index} has no 'problem' field")
        out.append(dict(record))
    return out
