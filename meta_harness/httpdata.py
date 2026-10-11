"""Stdlib download and JSONL helpers shared by every dataset loader."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

Http = Callable[[str], bytes]
USER_AGENT = "meta-harness/0.2"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
CACHE_ROOT = Path(__file__).resolve().parent.parent / "data" / "cache"


def urllib_get(url: str, timeout: float = 60.0, attempts: int = 3) -> bytes:
    """GET with a User-Agent header; retries 429, 5xx and connection errors with backoff."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if (error.code != 429 and error.code < 500) or attempt == attempts - 1:
                raise
        except urllib.error.URLError:
            if attempt == attempts - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError("attempts must be >= 1")


def get_json(url: str, http: Http = urllib_get) -> Any:
    return json.loads(http(url).decode("utf-8"))


def hf_rows(dataset: str, config: str, split: str, http: Http = urllib_get, page: int = 100,
            limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while limit is None or offset < limit:
        length = page if limit is None else min(page, limit - offset)
        query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                        "offset": offset, "length": length})
        payload = get_json(f"{HF_ROWS}?{query}", http)
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= int(payload.get("num_rows_total", offset)):
            break
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path, tolerate_torn: bool = True) -> list[dict]:
    """Read JSONL. When `tolerate_torn`, a final line that fails to parse (torn write)
    is skipped; a bad line anywhere else raises ValueError."""
    lines = [(number, line) for number, line in
             enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1) if line.strip()]
    rows = []
    for position, (number, line) in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if tolerate_torn and position == len(lines) - 1:
                break
            raise ValueError(f"{path}:{number}: invalid JSON line") from error
    return rows


def carve(rows: Sequence[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded, disjoint split: returns (rest, carved) with round(len * fraction) carved rows."""
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    chosen = set(order[:int(round(len(rows) * fraction))])
    rest = [dict(rows[i]) for i in range(len(rows)) if i not in chosen]
    carved = [dict(rows[i]) for i in sorted(chosen)]
    return rest, carved


__all__ = ["Http", "urllib_get", "get_json", "HF_ROWS", "hf_rows", "write_jsonl", "read_jsonl",
           "carve", "CACHE_ROOT"]
