"""Read-only queries over a finished (or running) experience directory.

Nothing here writes under the run root. Torn JSONL lines and torn JSON files are skipped, and every
returned string passes through the redactor.
"""

from __future__ import annotations

import difflib
import json
import re
import shutil
import statistics
from pathlib import Path, PurePosixPath
from typing import Any

from .httpdata import read_jsonl
from .redact import bounded, redact, redact_obj, safe_id

CATEGORIES = ("source", "scores", "traces", "reasoning", "summary", "run_meta", "other")
RUN_META = {"run.json", "frontier.json", "receipt.json", "proposal.json", "parent.json", "outcomes.json",
            "test_results.json", "manifest.json", "proposer_log.jsonl"}
VERDICTS = ("accepted", "rejected_invalid_source", "rejected_split_violation", "stale_parent",
            "budget_expired", "failed_eval", "incomplete")
OUTCOME_KEYS = ("dispatched", "denied", "denied_split", "failed", "aborted")
_ITERATION = re.compile(r"iteration-(\d+)")
_STEM = re.compile(r"candidate-(\d+)\.proposer_log\.jsonl$")


# -- tolerant readers --------------------------------------------------------------------------

def _json(path: Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None


def _rows(path: Path) -> list[dict]:
    try:
        return [r for r in read_jsonl(path, tolerate_torn=True) if isinstance(r, dict)]
    except (OSError, ValueError):
        pass
    rows = []  # a bad line in the middle: keep every line that parses
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _candidate_dirs(root: Path) -> list[Path]:
    base = Path(root) / "candidates"
    return sorted(d for d in base.iterdir() if d.is_dir()) if base.is_dir() else []


def _num(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default


def _scores(root: Path) -> list[dict]:
    out = []
    for directory in _candidate_dirs(root):
        data = _json(directory / "scores.json")
        if isinstance(data, dict):
            out.append({**data, "candidate_id": data.get("candidate_id") or directory.name})
    return out


def _candidate_path(root: Path, candidate_id: str) -> Path:
    if not candidate_id or Path(candidate_id).name != candidate_id or candidate_id in (".", ".."):
        raise ValueError(f"not a candidate id: {bounded(candidate_id, 80)}")
    directory = Path(root) / "candidates" / candidate_id
    if not directory.is_dir():
        raise FileNotFoundError(f"no candidate {bounded(candidate_id, 80)} under {bounded(str(root), 200)}")
    return directory


# -- top-k, diffs, warm start --------------------------------------------------------------------

def top_k(root: Path, k: int = 5, by: str = "score") -> list[dict]:
    """Valid candidates only. by="score": best score first, ties by lower context_cost.
    by="cost": lowest context_cost first, ties by higher score."""
    if by not in ("score", "cost"):
        raise ValueError("by must be 'score' or 'cost'")
    rows = []
    for item in _scores(root):
        score = _num(item.get("score"), float("-inf"))
        if item.get("valid") is False or score == float("-inf") or score != score:
            continue
        rows.append({"candidate_id": item["candidate_id"], "score": score,
                     "context_cost": _num(item.get("context_cost")), "split": item.get("split", "search"),
                     "repeats": item.get("repeats", 1)})
    if by == "score":
        rows.sort(key=lambda r: (-r["score"], r["context_cost"], r["candidate_id"]))
    else:
        rows.sort(key=lambda r: (r["context_cost"], -r["score"], r["candidate_id"]))
    return redact_obj(rows[:max(0, int(k))])


def _score_of(root: Path, candidate_id: str) -> tuple[float | None, float | None]:
    data = _json(_candidate_path(root, candidate_id) / "scores.json")
    if not isinstance(data, dict):
        return None, None
    return _num(data.get("score")), _num(data.get("context_cost"))


def diff_candidates(root: Path, a: str, b: str) -> str:
    """Unified diff of harness.py between two candidates, plus score and context-cost deltas."""
    def source(candidate_id: str) -> list[str]:
        path = _candidate_path(root, candidate_id) / "harness.py"
        return path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    (sa, ca), (sb, cb) = _score_of(root, a), _score_of(root, b)

    def delta(x: float | None, y: float | None) -> str:
        return "n/a" if x is None or y is None else f"{y - x:+.6g}"

    lines = list(difflib.unified_diff(source(a), source(b), f"{a}/harness.py", f"{b}/harness.py"))
    head = [f"# score: {a}={sa} {b}={sb} delta={delta(sa, sb)}",
            f"# context_cost: {a}={ca} {b}={cb} delta={delta(ca, cb)}"]
    body = "".join(line if line.endswith("\n") else line + "\n" for line in lines)
    return redact("\n".join(head) + "\n" + body)


def _run_summary(root: Path) -> dict:
    root = Path(root)
    frontier = _json(root / "frontier.json")
    run = _json(root / "run.json")
    run = run if isinstance(run, dict) else {}
    best = top_k(root, 1)
    return {"root": bounded(str(root), 300),
            "frontier": [{"candidate_id": f.get("candidate_id"), "score": f.get("score"),
                          "context_cost": f.get("context_cost")}
                         for f in frontier if isinstance(f, dict)] if isinstance(frontier, list) else [],
            "best_score": best[0]["score"] if best else None,
            "best_candidate": best[0]["candidate_id"] if best else None,
            "candidates": len(_candidate_dirs(root)),
            "budget": run.get("budget") if isinstance(run.get("budget"), dict) else {}}


def diff_runs(root_a: Path, root_b: Path) -> dict:
    """Frontier, best score and budget of two runs side by side."""
    a, b = _run_summary(root_a), _run_summary(root_b)
    keys = sorted(set(a["budget"]) | set(b["budget"]))
    budget = {k: {"a": a["budget"].get(k), "b": b["budget"].get(k)} for k in keys}
    best = None
    if a["best_score"] is not None and b["best_score"] is not None:
        best = b["best_score"] - a["best_score"]
    return redact_obj({"a": a, "b": b, "best_score_delta": best, "budget": budget})


def _inside(root: Path, path: Path) -> bool:
    root_r, path_r = Path(root).resolve(), Path(path).resolve()
    return path_r == root_r or root_r in path_r.parents


def import_candidates(source_root: Path, dest_seeds: Path, k: int = 3) -> list[Path]:
    """Copy the top-k harness.py files of a run into dest_seeds as seed files (warm start).
    The destination may not be inside the run root (run directories are read-only)."""
    if _inside(source_root, dest_seeds):
        raise ValueError("destination must be outside the run root (run directories are read-only)")
    dest = Path(dest_seeds)
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for rank, row in enumerate(top_k(source_root, k), 1):
        source = Path(source_root) / "candidates" / row["candidate_id"] / "harness.py"
        if not source.is_file():
            continue
        target = dest / f"warm-{rank:02d}-{safe_id(row['candidate_id'])}.py"
        shutil.copyfile(source, target)
        out.append(target)
    return out


# -- proposer read statistics (paper Table 8) ------------------------------------------------------

def categorise(path: str) -> str:
    """Bucket a path the proposer touched."""
    parts = [p for p in PurePosixPath(str(path).replace("\\", "/")).parts if p not in (".", "/")]
    name = parts[-1].lower() if parts else ""
    if name.endswith(".py"):
        return "source"
    if name == "scores.json":
        return "scores"
    if (name.startswith("traces") and name.endswith(".jsonl")) or any(p.lower().startswith("traces")
                                                                       for p in parts[:-1]):
        return "traces"
    if "reasoning" in name:
        return "reasoning"
    if "summary" in name:
        return "summary"
    if name in RUN_META:
        return "run_meta"
    return "other"


def _iteration_of(text: str) -> int:
    match = _ITERATION.search(text)
    return int(match.group(1)) if match else 0


def _proposer_logs(root: Path) -> list[tuple[int, Path]]:
    """(iteration, path); a proposal log already copied into a candidate counts once."""
    root = Path(root)
    found: list[tuple[int, Path]] = []
    labels = []
    mapped: set[tuple[int, str]] = set()  # (iteration, proposal stem) already owned by a candidate
    for directory in _candidate_dirs(root):
        for name in ("receipt.json", "proposal.json"):
            data = _json(directory / name)
            if isinstance(data, dict):
                for key in ("proposal_id", "candidate"):
                    if isinstance(data.get(key), str):
                        mapped.add((_iteration_of(directory.name), data[key]))
        log = directory / "proposer_log.jsonl"
        if log.is_file():
            found.append((_iteration_of(directory.name), log))
            labels.append(directory.name)
    for log in sorted(root.glob("proposals/iteration-*/*.proposer_log.jsonl")):
        match = _STEM.search(log.name)
        label = f"{log.parent.name}-{int(match.group(1)):02d}" if match else None
        stem = log.name[:-len(".proposer_log.jsonl")]
        if (_iteration_of(log.parent.name), stem) in mapped:
            continue
        if label and any(c == label or c.startswith(label + "-") for c in labels):
            continue
        found.append((_iteration_of(log.parent.name), log))
    return found


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def read_stats(root: Path) -> dict:
    """Per iteration: successful reads, distinct files, grep and list calls, and the category
    breakdown of the files read; plus run-level medians over iterations. Refused reads (not ok)
    are counted separately and never as reads."""
    per: dict[int, dict[str, Any]] = {}
    for iteration, log in _proposer_logs(root):
        entry = per.setdefault(iteration, {"iteration": iteration, "proposals": 0, "files_read": 0,
                                           "grep_calls": 0, "list_calls": 0, "refused_calls": 0,
                                           "_files": set(), "categories": {c: 0 for c in CATEGORIES}})
        entry["proposals"] += 1
        for row in _rows(log):
            tool = row.get("tool")
            if row.get("role") != "tool" or tool not in ("read", "grep", "list"):
                continue
            if row.get("ok") is not True:
                entry["refused_calls"] += 1
                continue
            if tool == "read":
                args = row.get("arguments") if isinstance(row.get("arguments"), dict) else {}
                path = str(row.get("path") or args.get("path") or "")
                entry["files_read"] += 1
                entry["_files"].add(path.replace("\\", "/"))
                entry["categories"][categorise(path)] += 1
            elif tool == "grep":
                entry["grep_calls"] += 1
            else:
                entry["list_calls"] += 1
    iterations = []
    for key in sorted(per):
        entry = per[key]
        entry["distinct_files"] = len(entry.pop("_files"))
        iterations.append(entry)
    medians: dict[str, Any] = {name: _median([e[name] for e in iterations])
                               for name in ("files_read", "distinct_files", "grep_calls", "list_calls")}
    medians["categories"] = {c: _median([e["categories"][c] for e in iterations]) for c in CATEGORIES}
    totals = {c: sum(e["categories"][c] for e in iterations) for c in CATEGORIES}
    total = sum(totals.values())
    buckets = {"source": totals["source"], "traces": totals["traces"],
               "score_summary": totals["scores"] + totals["summary"],
               "other": totals["reasoning"] + totals["run_meta"] + totals["other"]}
    reads = [e["files_read"] for e in iterations]
    table8 = {"median_files_read": _median(reads), "min_files_read": min(reads, default=0),
              "max_files_read": max(reads, default=0), "total_reads": total,
              "shares_pct": {b: (100.0 * n / total if total else 0.0) for b, n in buckets.items()}}
    return redact_obj({"iterations": iterations, "medians": medians, "table8": table8})


# -- receipts (spec section 11) --------------------------------------------------------------------

def _receipts(root: Path) -> list[tuple[Path, dict]]:
    out = []
    for directory in _candidate_dirs(root):
        data = _json(directory / "receipt.json")
        if isinstance(data, dict):
            data.setdefault("candidate_id", directory.name)
            out.append((directory, data))
    return out


def _stage(receipt: dict) -> str:
    if isinstance(receipt.get("evaluation"), dict):
        return "evaluated"
    validation = receipt.get("validation")
    if isinstance(validation, dict):
        return "validated" if validation.get("ok") else "validation_failed"
    return "proposed"


def report(root: Path) -> dict:
    """Receipt verdict and stage counts (overall and per optimizer iteration), and tool outcomes
    joined to scores per verdict (Table 8 with denied / failed / aborted)."""
    verdicts = {v: 0 for v in VERDICTS}
    stages: dict[str, int] = {}
    by_iteration: dict[str, dict[str, int]] = {}
    joined: dict[str, dict[str, Any]] = {}
    for directory, receipt in _receipts(root):
        verdict = bounded(str(receipt.get("verdict") or "incomplete"), 80)
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        stage = _stage(receipt)
        stages[stage] = stages.get(stage, 0) + 1
        row = by_iteration.setdefault(f"{_iteration_of(str(receipt['candidate_id'])):04d}", {})
        row[verdict] = row.get(verdict, 0) + 1
        group = joined.setdefault(verdict, {"receipts": 0, "scored": 0, "score_sum": 0.0,
                                            **{k: 0 for k in OUTCOME_KEYS}})
        group["receipts"] += 1
        outcomes = receipt.get("tool_outcomes") if isinstance(receipt.get("tool_outcomes"), dict) else {}
        for key in OUTCOME_KEYS:
            group[key] += int(_num(outcomes.get(key)))
        scores = _json(directory / "scores.json")
        if isinstance(scores, dict) and isinstance(scores.get("score"), (int, float)) \
                and scores.get("valid") is not False:
            group["scored"] += 1
            group["score_sum"] += float(scores["score"])
    for group in joined.values():
        group["mean_score"] = group["score_sum"] / group["scored"] if group["scored"] else None
        del group["score_sum"]
    return redact_obj({"receipts": sum(verdicts.values()), "verdicts": verdicts, "stages": stages,
                       "by_iteration": dict(sorted(by_iteration.items())), "tool_outcomes_by_verdict": joined})


def export_cases(root: Path, out: Path) -> int:
    """One read-only case per receipt, as JSONL. Refuses to write inside the run root."""
    out_r = Path(out).resolve()
    if _inside(root, out):
        raise ValueError("--out must be outside the run root (run directories are read-only)")
    lines = []
    for directory, receipt in _receipts(root):
        scores = _json(directory / "scores.json")
        if not isinstance(scores, dict):
            scores = receipt.get("evaluation") if isinstance(receipt.get("evaluation"), dict) else {}
        try:
            rationale = bounded((directory / "proposer_reasoning.md").read_text(encoding="utf-8",
                                                                                  errors="replace"), 2000)
        except OSError:
            rationale = ""
        case = {"id": safe_id(str(receipt["candidate_id"])), "verdict": receipt.get("verdict") or "incomplete",
                "parents": list(receipt.get("parents_read") or []), "fingerprint": receipt.get("fingerprint"),
                "scores": {k: scores.get(k) for k in ("score", "context_cost", "valid", "split", "repeats")
                           if k in scores},
                "rationale": rationale}
        lines.append(json.dumps(redact_obj(case), sort_keys=True))
    out_r.parent.mkdir(parents=True, exist_ok=True)
    out_r.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)
