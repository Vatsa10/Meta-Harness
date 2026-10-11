"""Parameter-keyed split cache shared by the classification loaders (not a loader itself)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

SPLITS = ("train", "val", "test")
DEFAULT_LIMIT = 200


def cached(cache_dir: Path, key: Mapping) -> dict[str, Path] | None:
    """The split files when all exist and meta.json equals `key`, else None."""
    paths = {split: Path(cache_dir) / f"{split}.jsonl" for split in SPLITS}
    meta = Path(cache_dir) / "meta.json"
    if not (meta.is_file() and all(p.is_file() for p in paths.values())):
        return None
    try:
        stored = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return paths if stored == json.loads(json.dumps(key)) else None


def store(cache_dir: Path, key: Mapping, splits: Mapping[str, Sequence[dict]]) -> dict[str, Path]:
    from ..httpdata import write_jsonl
    meta = Path(cache_dir) / "meta.json"
    if meta.exists():
        meta.unlink()  # a torn write must not leave stale meta next to new splits
    paths = {s: write_jsonl(Path(cache_dir) / f"{s}.jsonl", splits[s]) for s in SPLITS}
    meta.write_text(json.dumps(key, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return paths
