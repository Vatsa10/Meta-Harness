"""Full lexical math router (paper App. B.2): the package implementation as a seed.

Corpus path comes from META_HARNESS_CORPUS (JSONL: problem, solution, difficulty, source, id).
Routes: combinatorics / geometry (separate hard index) / number theory / default (adaptive K).
To change the routing, copy the logic from meta_harness/math_harnesses.py into this file.
"""

import json
import os

from meta_harness.math_harnesses import MathRetrievalHarness


def _load_corpus():
    path = os.environ.get("META_HARNESS_CORPUS", "")
    if not path or not os.path.isfile(path):
        return []
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                items.append(record)
    return items


class Harness(MathRetrievalHarness):
    def __init__(self):
        super().__init__(_load_corpus(), default_k=int(os.environ.get("META_HARNESS_RETRIEVAL_K", "3")))
