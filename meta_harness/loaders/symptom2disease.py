"""Symptom2Disease (paper: 22 classes), the Gretel AI symptom-to-diagnosis set.

Source verified 2026-10-11 against the HF datasets-server: dataset `gretelai/symptom_to_diagnosis`,
config `default`, splits `train` (853 rows) and `test` (212 rows). Fields: `input_text`
(patient description) and `output_text` (disease name).

`limit` (default 200) caps the rows read per HF split (the first `limit`); None reads everything.
The cache is keyed on source, seed, val_fraction and limit (meta.json beside the splits).

Splits: HF `train` is carved (seeded) into train and val; `test` is the HF test split.
Each JSONL row is {"input", "label", "labels"} where `labels` is the sorted label set of the HF train
split (22 diseases on the full data).
"""

from __future__ import annotations

from pathlib import Path

from ..httpdata import Http, carve, hf_rows, urllib_get
from ._cache import DEFAULT_LIMIT, cached, store

NAME, DOMAIN = "symptom2disease", "classification"
SOURCE = "hf:gretelai/symptom_to_diagnosis (train 853, test 212; input_text, output_text)"
DATASET, CONFIG = "gretelai/symptom_to_diagnosis", "default"
PAGE = 100


def fetch(cache_dir: Path, http: Http = urllib_get, val_fraction: float = 0.2,
          seed: int = 0, limit: int | None = DEFAULT_LIMIT) -> dict[str, Path]:
    """train/val carved from HF train (seeded), test = HF test. Rows: {"input", "label", "labels"},
    labels = sorted label set of the HF train split (22)."""
    key = {"source": SOURCE, "seed": seed, "val_fraction": val_fraction, "test_fraction": None,
           "limit": limit}
    hit = cached(cache_dir, key)
    if hit:
        return hit
    train_raw = hf_rows(DATASET, CONFIG, "train", http, page=PAGE, limit=limit)
    test_raw = hf_rows(DATASET, CONFIG, "test", http, page=PAGE, limit=limit)
    labels = sorted({str(r["output_text"]) for r in train_raw})

    def rows(raw):
        return [{"input": str(r["input_text"]), "label": str(r["output_text"]), "labels": labels}
                for r in raw]

    train, val = carve(rows(train_raw), val_fraction, seed)
    return store(cache_dir, key, {"train": train, "val": val, "test": rows(test_raw)})
