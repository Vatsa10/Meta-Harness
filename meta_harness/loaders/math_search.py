"""Math search set: 250 hard problems from OlympiadBench + Omni-MATH (paper section 4.2).

Sources (field names pinned from the dataset cards, not re-verified offline):
    Hothan/OlympiadBench  config OE_TO_maths_en_COMP, split train: question, final_answer (list)
    KbsdJames/Omni-MATH   split test: problem, answer, difficulty  (hard = difficulty >= 7)
Rows are {"problem", "answer"}. Half the set comes from each source (the other source fills any
shortfall); the draw is seeded. Fetch scans each source in full, so a real run is a documented
command, not part of the test suite.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..httpdata import hf_rows, urllib_get, write_jsonl

NAME, DOMAIN = "math_search", "math"
OLYMPIAD = ("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", "train")
OMNI = ("KbsdJames/Omni-MATH", "default", "test")
HARD_DIFFICULTY = 7.0


def _answer(value) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(item) for item in value)
    return str(value or "").strip().strip("$").strip()


def _olympiad(row: dict) -> dict | None:
    problem, answer = str(row.get("question") or "").strip(), _answer(row.get("final_answer"))
    return {"problem": problem, "answer": answer} if problem and answer else None


def _omni(row: dict) -> dict | None:
    try:
        hard = float(row.get("difficulty")) >= HARD_DIFFICULTY
    except (TypeError, ValueError):
        return None
    problem, answer = str(row.get("problem") or "").strip(), _answer(row.get("answer"))
    return {"problem": problem, "answer": answer} if hard and problem and answer else None


def fetch(cache_dir, http=urllib_get, size: int = 250, seed: int = 0) -> dict[str, Path]:
    cache = Path(cache_dir)
    path, meta = cache / "search.jsonl", cache / "search.meta.json"
    want = {"size": size, "seed": seed}
    if path.exists() and meta.exists():
        try:
            if json.loads(meta.read_text(encoding="utf-8")) == want:
                return {"search": path}
        except ValueError:
            pass
    pools = []
    for (dataset, config, split), adapt in ((OLYMPIAD, _olympiad), (OMNI, _omni)):
        pool = [item for row in hf_rows(dataset, config, split, http) if (item := adapt(row))]
        random.Random(seed).shuffle(pool)
        pools.append(pool)
    first = min(size // 2, len(pools[0]))
    second = min(size - first, len(pools[1]))
    first = min(size - second, len(pools[0]))
    chosen = pools[0][:first] + pools[1][:second]
    write_jsonl(path, chosen)
    meta.write_text(json.dumps(want), encoding="utf-8")
    return {"search": path}
