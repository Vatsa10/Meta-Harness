"""Cross-run lessons library: deterministic distillation from a finished run, keyword query.

A lesson is one JSONL record {cue, lesson, evidence, confidence, source_run, created}. The record
shape follows google-research/harness2 (Apache-2.0), core/library.py and selection._distil; here the
lessons are derived without an LLM from frontier candidates, their parents and score deltas.
Run roots are only read; the library lives elsewhere. All stored text is redacted.
"""

from __future__ import annotations

import ast
import difflib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from . import experience_query as q
from .redact import bounded, redact_obj

LIBRARY_ENV = "META_HARNESS_LESSONS_DIR"
LIBRARY_FILE = "lessons.jsonl"
FIELDS = ("cue", "lesson", "evidence", "confidence", "source_run", "created")


def library_dir() -> Path:
    override = os.environ.get(LIBRARY_ENV)
    return Path(override) if override else Path.home() / ".meta-harness" / "lessons"


def _top_level(source: str) -> dict[str, str] | None:
    """name -> normalised dump of each top-level def/class/assignment, or None if unparsable."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    out: dict[str, str] = {}
    other: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = ast.dump(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            other.append(ast.dump(node))
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
            out["=".join(t.id for t in node.targets)] = ast.dump(node)  # type: ignore[attr-defined]
        else:
            other.append(ast.dump(node))
    out["<module>"] = "|".join(other)
    return out


def changed_components(parent: str, child: str) -> list[str]:
    """Sorted names of top-level components that differ (added, removed or edited)."""
    a, b = _top_level(parent), _top_level(child)
    if a is None or b is None:
        return ["<source>"] if parent != child else []
    names = sorted(name for name in set(a) | set(b) if a.get(name) != b.get(name))
    return [n for n in names if n != "<module>"] + (["<module>"] if "<module>" in names else [])


def _line_counts(parent: str, child: str) -> tuple[int, int]:
    added = removed = 0
    for line in difflib.unified_diff(parent.splitlines(), child.splitlines(), lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed


def _read_source(root: Path, candidate_id: str) -> str | None:
    try:
        return (q._candidate_path(root, candidate_id) / "harness.py").read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return None


def _parent_of(root: Path, candidate_id: str) -> str | None:
    receipt = q._json(Path(root) / "candidates" / candidate_id / "receipt.json")
    parents = receipt.get("parents_read") if isinstance(receipt, dict) else None
    for parent in parents or []:
        if isinstance(parent, str) and parent != candidate_id and (Path(root) / "candidates" / parent).is_dir():
            try:
                q._candidate_path(root, parent)
            except (ValueError, FileNotFoundError):
                continue
            return parent
    return None


def _confidence(delta: float, n: int) -> float:
    """Grows with the size of the score change and with the number of repeats behind it."""
    return round(max(0.0, min(1.0, abs(delta) * 4.0)) * (n / (n + 1.0)), 3)


def distil(root: Path, created: str | None = None) -> list[dict]:
    """Lessons for a finished run: each frontier candidate (or, with no frontier file, each valid
    candidate) that has a parent yields one lesson from its diff and score delta."""
    root = Path(root)
    if not (root / "candidates").is_dir():
        raise ValueError(f"not a run root (no candidates/): {bounded(str(root), 200)}")
    scores = {row["candidate_id"]: row for row in q._scores(root)}
    frontier = q._json(root / "frontier.json")
    wanted = [f.get("candidate_id") for f in frontier if isinstance(f, dict)] if isinstance(frontier, list) else []
    wanted = [c for c in wanted if isinstance(c, str)] or sorted(scores)
    stamp = created or datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_name = bounded(root.name, 120)
    records = []
    for cid in wanted:
        row = scores.get(cid)
        parent = _parent_of(root, cid) if row is not None else None
        prow = scores.get(parent) if parent else None
        if (row is None or prow is None or row.get("valid") is False or prow.get("valid") is False
                or not isinstance(row.get("score"), (int, float)) or not isinstance(prow.get("score"), (int, float))):
            continue
        child, base = _read_source(root, cid), _read_source(root, parent)
        if child is None or base is None:
            continue
        components = changed_components(base, child)
        if not components:
            continue
        delta = float(row["score"]) - float(prow["score"])
        added, removed = _line_counts(base, child)
        n = int(q._num(row.get("repeats"), 1)) or 1
        cue = "changed " + ", ".join(components[:4]) + (f" (+{len(components) - 4} more)" if len(components) > 4 else "")
        verb = "improved" if delta > 0 else "regressed" if delta < 0 else "left unchanged"
        lesson = (f"Editing {', '.join(components[:4])} ({added} lines added, {removed} removed) {verb} the score "
                  f"{prow['score']:.4g} -> {row['score']:.4g} ({delta:+.4g}).")
        records.append(redact_obj({"cue": cue, "lesson": lesson, "evidence": [cid, parent],
                                   "confidence": _confidence(delta, n), "source_run": run_name,
                                   "created": stamp}))
    return records


# -- storage -------------------------------------------------------------------------------------

def _key(record: dict) -> tuple:
    return (record.get("source_run"), record.get("cue"), tuple(record.get("evidence") or ()))


def load(directory: Path | None = None) -> list[dict]:
    path = Path(directory or library_dir()) / LIBRARY_FILE
    out = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and all(f in row for f in FIELDS):
            out.append(row)
    return out


def add(records: list[dict], directory: Path | None = None) -> int:
    """Append records not already stored (same source_run, cue and evidence); returns how many."""
    target = Path(directory or library_dir())
    seen = {_key(r) for r in load(target)}
    fresh = []
    for record in records:
        record = redact_obj(record)
        if _key(record) in seen:
            continue
        seen.add(_key(record))
        fresh.append(record)
    if fresh:
        target.mkdir(parents=True, exist_ok=True)
        with open(target / LIBRARY_FILE, "a", encoding="utf-8", newline="\n") as handle:
            for record in fresh:
                handle.write(json.dumps({f: record[f] for f in FIELDS}, sort_keys=True) + "\n")
    return len(fresh)


def query(text: str | None = None, directory: Path | None = None, limit: int = 20) -> list[dict]:
    """Lessons whose cue or lesson contains any query word (case-insensitive), best match then
    confidence first. No query lists everything by confidence."""
    rows = load(directory)
    words = [w for w in re.findall(r"\w+", (text or "").lower())]
    scored = []
    for row in rows:
        haystack = f"{row['cue']} {row['lesson']}".lower()
        hits = sum(1 for w in words if w in haystack)
        if words and not hits:
            continue
        scored.append((hits, q._num(row.get("confidence")), row))
    scored.sort(key=lambda t: (-t[0], -t[1], str(t[2].get("created")), str(t[2].get("cue"))))
    return redact_obj([row for _, _, row in scored[:max(0, int(limit))]])


def distil_to_library(root: Path, directory: Path | None = None) -> tuple[int, int]:
    """(derived, newly stored). Refuses a library inside the run root (run roots are read-only)."""
    target = Path(directory or library_dir())
    if q._inside(root, target):
        raise ValueError("lessons library must be outside the run root (run directories are read-only)")
    records = distil(root)
    return len(records), add(records, target)


__all__ = ["add", "changed_components", "distil", "distil_to_library", "library_dir", "load", "query"]
