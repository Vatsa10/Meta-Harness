"""`meta-harness math-corpus build`: fetch, deduplicate and decontaminate the math retrieval corpus.

Pipeline: fetch the eight Table 10 sources -> dedup inside the corpus -> drop every problem that
matches a search-set or IMO test problem (exact 60-character prefix, then shingle Jaccard >= 0.8)
-> `corpus.clean.jsonl` plus `decontam_report.json` in the cache directory. The BM25 index is
not stored: the math harness builds it lazily. Small by default (200 rows per source, 20 search, 20 test problems);
only --full downloads everything (about 535K problems).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .. import decontam
from ..httpdata import CACHE_ROOT, read_jsonl, urllib_get, write_jsonl
from ..loaders import imo, math_corpus, math_search

NAME = "math-corpus"
SMALL_LIMIT = 200  # owner small-run cap; the full corpus needs an explicit --full
HELP = "build the deduplicated, decontaminated math retrieval corpus"


def add_arguments(parser) -> None:
    parser.add_argument("action", choices=["build"])
    parser.add_argument("--limit-per-source", type=int, default=SMALL_LIMIT,
                        help=f"cap rows fetched from each of the eight sources (default {SMALL_LIMIT})")
    parser.add_argument("--full", action="store_true",
                        help="no caps: the whole corpus, the 250-problem search set and the 200-problem test set")
    parser.add_argument("--cache-dir", default=str(CACHE_ROOT / "math_corpus"))


def build(cache_dir, limit_per_source: int | None = SMALL_LIMIT, http=urllib_get,
          full: bool = False) -> dict:
    """Run the pipeline; returns the report that is also written to decontam_report.json."""
    cache = Path(cache_dir)
    if full:
        limit_per_source = None
    raw = read_jsonl(math_corpus.fetch(cache, http, per_source_limit=limit_per_source)["corpus"])
    held_out: list[str] = []
    for rows in (read_jsonl(math_search.fetch(cache / "math_search", http, limit=None if full else 20)["search"]),
                 read_jsonl(imo.fetch(cache / "imo", http, limit=None if full else 20)["test"])):
        held_out.extend(str(row["problem"]) for row in rows)
    unique = decontam.dedup(raw)
    kept, removed = decontam.decontaminate(unique, held_out)
    write_jsonl(cache / "corpus.clean.jsonl", kept)
    report = {
        "fetched": len(raw),
        "after_dedup": len(unique),
        "held_out_problems": len(held_out),
        "kept": len(kept),
        "removed": len(removed),
        "removed_ids": [row["id"] for row in removed],
        "removed_by_source": dict(Counter(row.get("source", "") for row in removed)),
        "prefix_chars": 60,
        "jaccard_threshold": 0.8,
    }
    (cache / "decontam_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run(args) -> int:
    report = build(args.cache_dir, args.limit_per_source, full=args.full)
    print(f"fetched {report['fetched']}, after dedup {report['after_dedup']}, "
          f"removed {report['removed']} as held-out matches, kept {report['kept']} -> "
          f"{Path(args.cache_dir) / 'corpus.clean.jsonl'}")
    return 0
