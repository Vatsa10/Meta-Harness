"""USPTO-50k reaction classification (paper: 180 classes).

Source verified 2026-10-11 against the HF datasets-server: dataset `pingzhili/uspto-50k`, config
`default`, splits `train` (49015 rows) and `validation` (1001 rows); no `test` split is served.
Fields: `class` (int 1-10), `id` (patent id), `prod_smiles`, `rxn_smiles` (atom-mapped
reactants>>product), `prod_smiles_pop`, `keep`.

DEVIATION FROM THE PAPER: no served field has 180 values. `class` is the 10-way USPTO reaction
type (validation counts: 1:319 2:224 3:101 4:17 5:16 6:190 7:92 8:14 9:23 10:5), and distinct
reactant sets are near-unique per row. The 180-way label of the paper is not reproducible from
this copy, so `class` (as a string "1".."10") is used, with the product SMILES as `input`.
labels = the classes of the train split sorted numerically.

Splits: the served `validation` split is the test split; val is a seeded `val_fraction` carve of
`train`.
"""

from __future__ import annotations

from pathlib import Path

from ..httpdata import Http, carve, hf_rows, urllib_get, write_jsonl

NAME, DOMAIN = "uspto50k", "classification"
SOURCE = "hf:pingzhili/uspto-50k (train, validation)"
DATASET, CONFIG = "pingzhili/uspto-50k", "default"
PAGE = 100
SPLITS = ("train", "val", "test")


def _cached(cache_dir: Path) -> dict[str, Path] | None:
    paths = {split: Path(cache_dir) / f"{split}.jsonl" for split in SPLITS}
    return paths if all(p.is_file() for p in paths.values()) else None


def fetch(cache_dir: Path, http: Http = urllib_get, val_fraction: float = 0.2,
          seed: int = 0) -> dict[str, Path]:
    cached = _cached(cache_dir)
    if cached:
        return cached
    train_raw = hf_rows(DATASET, CONFIG, "train", http, page=PAGE)
    test_raw = hf_rows(DATASET, CONFIG, "validation", http, page=PAGE)
    labels = [str(c) for c in sorted({int(r["class"]) for r in train_raw})]

    def rows(raw):
        return [{"input": str(r["prod_smiles"]), "label": str(int(r["class"])), "labels": labels}
                for r in raw]

    train, val = carve(rows(train_raw), val_fraction, seed)
    out = {"train": train, "val": val, "test": rows(test_raw)}
    return {split: write_jsonl(Path(cache_dir) / f"{split}.jsonl", out[split]) for split in SPLITS}
