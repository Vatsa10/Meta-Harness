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
labels = sorted set of normalised answers over the whole file.

Splits: test = seeded `test_fraction` of all records, val = seeded `val_fraction` of the rest,
train = remainder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..httpdata import Http, carve, urllib_get, write_jsonl

NAME, DOMAIN = "lawbench", "classification"
SOURCE = "https://raw.githubusercontent.com/open-compass/LawBench/main/data/zero_shot/3-3.json"
SPLITS = ("train", "val", "test")
_PREFIX = re.compile("^\\s*罪名\\s*[:：]\\s*")


def normalise_answer(answer: str) -> str:
    """Strip the charge-name prefix so the answer is the bare charge name."""
    return _PREFIX.sub("", str(answer)).strip()


def _cached(cache_dir: Path) -> dict[str, Path] | None:
    paths = {split: Path(cache_dir) / f"{split}.jsonl" for split in SPLITS}
    return paths if all(p.is_file() for p in paths.values()) else None


def fetch(cache_dir: Path, http: Http = urllib_get, val_fraction: float = 0.2,
          test_fraction: float = 0.3, seed: int = 0) -> dict[str, Path]:
    cached = _cached(cache_dir)
    if cached:
        return cached
    raw = json.loads(http(SOURCE).decode("utf-8"))
    if not isinstance(raw, list):
        raise ValueError("LawBench file must contain a JSON list")
    labels = sorted({normalise_answer(r["answer"]) for r in raw})
    rows = [{"input": str(r["question"]).strip(), "label": normalise_answer(r["answer"]),
             "labels": labels} for r in raw]
    rest, test = carve(rows, test_fraction, seed)
    train, val = carve(rest, val_fraction, seed)
    out = {"train": train, "val": val, "test": test}
    return {split: write_jsonl(Path(cache_dir) / f"{split}.jsonl", out[split]) for split in SPLITS}
