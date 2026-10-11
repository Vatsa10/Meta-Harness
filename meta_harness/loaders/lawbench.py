"""LawBench charge prediction (paper: 215 classes), task 3-3 of open-compass/LawBench.

Source verified 2026-10-11: `data/zero_shot/3-3.json` on raw.githubusercontent.com is a JSON list of
500 records with fields `instruction` (a fixed Chinese prompt: act as judge, give the charge),
`question` (the case facts, prefixed with the Chinese word for "facts" and a colon) and `answer`
(the Chinese word for "charge", a colon, then the charge; several charges are joined by ";").
Task 3-3 is the charge-prediction task; the other 3-x files are statute, term and amount tasks.

The file holds 500 records with 275 distinct normalised answers (202 distinct single charges once
multi-charge answers are split), not the paper's 215: this public copy is the closest available.
Normalisation: the charge prefix is stripped, so the label is the bare charge name; a multi-charge
answer stays one label (two charge names joined by ";"). `input` is `question` stripped (the fixed
`instruction` is dropped: it demands an answer format the harness protocol does not use).
labels = sorted set of normalised answers in train and val only. A test row whose gold label is
absent from `labels` is allowed (the harness cannot have been told it); it can never be predicted
from the label list, so the `exact` metric simply scores it wrong.

`limit` (default 200) keeps only the first `limit` records; None keeps all 500.

Splits: test = seeded `test_fraction` of all records, val = seeded `val_fraction` of the rest,
train = remainder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..httpdata import Http, carve, urllib_get
from ._cache import DEFAULT_LIMIT, cached, store

NAME, DOMAIN = "lawbench", "classification"
SOURCE = "https://raw.githubusercontent.com/open-compass/LawBench/main/data/zero_shot/3-3.json"
_PREFIX = re.compile("^\\s*罪名\\s*[:：]\\s*")


def normalise_answer(answer: str) -> str:
    """Strip the charge-name prefix so the answer is the bare charge name."""
    return _PREFIX.sub("", str(answer)).strip()


def fetch(cache_dir: Path, http: Http = urllib_get, val_fraction: float = 0.2,
          test_fraction: float = 0.3, seed: int = 0,
          limit: int | None = DEFAULT_LIMIT) -> dict[str, Path]:
    key = {"source": SOURCE, "seed": seed, "val_fraction": val_fraction,
           "test_fraction": test_fraction, "limit": limit}
    hit = cached(cache_dir, key)
    if hit:
        return hit
    raw = json.loads(http(SOURCE).decode("utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LawBench file must contain a JSON list")
    if limit is not None:
        raw = raw[:limit]
    rows = [{"input": str(r["question"]).strip(), "label": normalise_answer(r["answer"])} for r in raw]
    rest, test = carve(rows, test_fraction, seed)
    train, val = carve(rest, val_fraction, seed)
    labels = sorted({r["label"] for r in train + val})
    for split_rows in (train, val, test):
        for row in split_rows:
            row["labels"] = labels
    return store(cache_dir, key, {"train": train, "val": val, "test": test})
