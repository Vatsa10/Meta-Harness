"""Drift detection: judges scored offline against real transcripts, behind a ship gate that
is allowed to refuse all of them (see tools/tune_drift.py).

Three judges share one Verdict shape:
- judge_overlap: does recent work still touch the files the request first opened.
- judge_knn: does the shape of this stretch resemble past stretches that ended in a correction.
- judge_model: an optional model call, injected never imported, that degrades to a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass


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
