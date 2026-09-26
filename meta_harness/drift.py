"""Drift detection: judges scored offline against real transcripts, behind a ship gate that
is allowed to refuse all of them (see tools/tune_drift.py).

Three judges share one Verdict shape:
- judge_overlap: does recent work still touch the files the request first opened.
- judge_knn: does the shape of this stretch resemble past stretches that ended in a correction.
- judge_model: an optional model call, injected never imported, that degrades to a fallback.
"""
from __future__ import annotations

import collections
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Verdict:
    """A judge's opinion on whether a stretch has drifted off course."""

    drifting: bool
    score: float
    reason: str


def anchor_paths(stretch, anchor_calls: int = 3) -> set[str]:
    """The distinct paths seen in the first `anchor_calls` calls of the stretch."""
    return set(stretch.paths[:anchor_calls])


def judge_overlap(stretch, at_call: int, min_calls: int = 8, anchor_calls: int = 3) -> Verdict:
    """Judge by how much of the work since the anchor still touches the anchor's files.

    Biased toward long silent stretches on purpose: fewer than `min_calls` calls never drifts,
    because that is where the measured cost is (p90 = 41) and short stretches are where false
    positives annoy.
    """
    if at_call < min_calls:
        return Verdict(False, 0.0, "too few calls to judge")

    anchor = anchor_paths(stretch, anchor_calls=anchor_calls)
    if not anchor:
        return Verdict(False, 0.0, "no anchor paths - nothing to compare against")

    since_anchor = stretch.paths[anchor_calls:at_call]
    if not since_anchor:
        return Verdict(False, 0.0, "no paths touched since the anchor")

    overlapping = sum(1 for p in since_anchor if p in anchor)
    fraction = overlapping / len(since_anchor)
    drifting = fraction < 0.2
    reason = (f"anchor={sorted(anchor)} overlap_fraction={fraction:.2f}")
    return Verdict(drifting, fraction, reason)


def shape(stretch, at_call: int | None = None, n: int = 2) -> collections.Counter[str]:
    """Tool-name n-gram counts over the stretch's calls - the similarity feature for kNN.

    `at_call` limits the calls considered to the first `at_call` (a stretch judged mid-flight
    should only see its own history up to that point); `None` uses every call recorded so far.
    """
    calls = stretch.calls if at_call is None else stretch.calls[:at_call]
    counts: collections.Counter[str] = collections.Counter()
    for i in range(len(calls) - n + 1):
        counts[">".join(calls[i:i + n])] += 1
    return counts


def _cosine(a: collections.Counter[str], b: collections.Counter[str]) -> float:
    if not a or not b:
        return 0.0
    keys = set(a) & set(b)
    dot = sum(a[k] * b[k] for k in keys)
    if dot == 0:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_index(home: Path | None = None, limit: int | None = None) -> list[tuple["collections.Counter[str]", bool]]:
    """(shape, ended_in_correction) over every stretch this machine's transcript store has seen.

    An absent or missing home returns an empty index, not an error - a fresh install has no
    history yet.
    """
    from .cc_history import claude_home, is_artifact_project
    from .waste import iter_stretches

    root = Path(home) if home is not None else claude_home() / "projects"
    if not root.exists():
        return []

    paths: list[Path] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or is_artifact_project(directory.name):
            continue
        paths.extend(sorted(directory.glob("*.jsonl")))
    if limit:
        paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    index: list[tuple[collections.Counter[str], bool]] = []
    for path in paths:
        for stretch in iter_stretches(path):
            index.append((shape(stretch), stretch.ended_by == "correction"))
    return index


def judge_knn(stretch, at_call: int, index: Sequence[tuple["collections.Counter[str]", bool]],
              k: int = 9, min_calls: int = 8, threshold: float = 0.6) -> Verdict:
    """Judge by how similar past stretches ended.

    Cosine similarity over tool-name n-gram counts; `drifting` when the similarity-weighted
    share of the k nearest neighbours that ended in a correction is at or above `threshold`.
    An empty index never reports drift - a fresh install has no history, and firing on no
    evidence is a nag.
    """
    if at_call < min_calls:
        return Verdict(False, 0.0, "too few calls to judge")
    if not index:
        return Verdict(False, 0.0, "no history to compare against")

    query = shape(stretch, at_call=at_call)
    scored = [(_cosine(query, other), ended_in_correction) for other, ended_in_correction in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    neighbours = scored[:k]

    total_weight = sum(sim for sim, _ in neighbours)
    if total_weight == 0:
        return Verdict(False, 0.0, "no similar history to compare against")

    correction_weight = sum(sim for sim, ended in neighbours if ended)
    fraction = correction_weight / total_weight
    drifting = fraction >= threshold
    reason = f"{len(neighbours)} neighbours, correction-weighted fraction={fraction:.2f}"
    return Verdict(drifting, fraction, reason)


def judge_model(stretch, at_call: int, request: str,
                 complete: Callable[[str], str] | None, fallback: Callable[[], Verdict]) -> Verdict:
    """A model judge, injected never imported: `complete` is a callable this call never
    creates, so tests need no real model. Degrades to `fallback` whenever the capability is
    unverified - `complete` is None, raises, or answers with anything but a leading
    ``DRIFT:``/``OK:`` token - which is what makes depending on it safe.

    The prompt carries the request text, the tool-name sequence and distinct path BASENAMES
    only - never file contents, never full paths.
    """
    if complete is None:
        return fallback()

    calls = stretch.calls[:at_call]
    basenames = sorted({Path(p).name for p in stretch.paths[:at_call] if p})
    prompt = (
        "A coding agent is mid-task. Judge whether it has drifted off the original request.\n"
        f"Request: {request}\n"
        f"Tool calls so far: {', '.join(calls)}\n"
        f"Files touched (names only): {', '.join(basenames)}\n"
        "Answer with a leading 'DRIFT: <reason>' or 'OK: <reason>'."
    )

    try:
        answer = complete(prompt)
    except Exception:
        return fallback()

    if not isinstance(answer, str):
        return fallback()

    if answer.startswith("DRIFT:"):
        return Verdict(True, 1.0, answer[len("DRIFT:"):].strip())
    if answer.startswith("OK:"):
        return Verdict(False, 0.0, answer[len("OK:"):].strip())
    return fallback()
