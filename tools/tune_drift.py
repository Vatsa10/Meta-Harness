"""Tune the drift judges against real corrections, behind a gate that can refuse all of them.

The correction labels `iter_stretches` produces are noisy - a hand-labelled audit found only
11 of 21 flagged stretches were genuine corrections (52%). This tool does NOT trust
`ended_by == "correction"` as ground truth for scoring. It joins each stretch against a
hand-labelled gold set instead (correction-gold.json: {session_id, project, start_turn, calls,
label, why}, label "a" = genuine correction, "c" = not a correction) and scores against that.

A stretch is counted as a ground-truth POSITIVE only when it joins a gold "a" record on
(session_id, start_turn). Every other stretch - a gold "c" match, or one the detector never
flagged and that has no gold record at all - is a NEGATIVE. Some gold keys no longer exist on
disk (the transcript store rotates); those are skipped and reported, not treated as either.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from meta_harness.cc_history import claude_home, is_artifact_project
from meta_harness.drift import judge_knn, judge_overlap, shape
from meta_harness.waste import iter_stretches

# Fixed location of the hand-labelled gold set (an sdd planning artifact, not part of the
# package - shared across every lane's worktree, so it is referenced by its one true path
# rather than something relative to this file).
GOLD_PATH = Path(
    r"D:\Files\Vatsa\Projects\Meta-Harness\.superpowers\sdd\2026-09-26-drift-and-waste"
    r"\correction-gold.json"
)

PRECISION_BAR: float = 0.5
MIN_CALLS_SAVED: int = 3

# The judge's own min_calls gate - a stretch shorter than this is never even considered, in
# either judge, so it is excluded from evaluation entirely rather than counted as a miss.
JUDGE_MIN_CALLS = 8


def gate(precision: float, median_calls_saved: float) -> bool:
    """Whether a judge ships. Stated before tuning, and not softened after seeing the numbers:
    a judge ships only when it clears BOTH bars. No threshold argument - this exists to be able
    to refuse, and a caller that could pass one in could tune the gate away."""
    return precision >= PRECISION_BAR and median_calls_saved >= MIN_CALLS_SAVED


def _load_gold(gold_path: Path = GOLD_PATH) -> list[dict[str, Any]]:
    try:
        with gold_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return []


def _transcripts(root: Path, limit: int | None) -> list[Path]:
    if not root.exists():
        return []
    paths: list[Path] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or is_artifact_project(directory.name):
            continue
        paths.extend(sorted(directory.glob("*.jsonl")))
    if limit:
        paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return paths


def _all_stretches(paths: list[Path]) -> list:
    """Every stretch on disk, keyed session-by-session so the kNN index can be built
    leave-one-session-out later."""
    stretches = []
    for path in paths:
        stretches.extend(iter_stretches(path))
    return stretches


def evaluate(judge_name: str, home: Path | None = None, limit: int | None = None) -> dict[str, Any]:
    """Score one judge ("overlap" or "knn") against the hand-labelled gold set, not against the
    noisy `ended_by` label. Returns fired/corrections/caught/recall/precision/median_calls_saved
    and whether the judge ships, plus how many gold records had no matching stretch on disk."""
    root = Path(home) if home is not None else claude_home() / "projects"
    paths = _transcripts(root, limit)
    stretches = _all_stretches(paths)

    gold = _load_gold()
    gold_by_key = {(record["session_id"], record["start_turn"]): record["label"] for record in gold}

    on_disk_keys = {(s.session_id, s.start_turn) for s in stretches}
    matched_gold = {key for key in gold_by_key if key in on_disk_keys}
    gold_skipped = len(gold_by_key) - len(matched_gold)

    # Build the kNN training index once, tagged by session, for leave-one-session-out lookups.
    knn_index_by_session: dict[str, list] = {}
    if judge_name == "knn":
        for s in stretches:
            knn_index_by_session.setdefault(s.session_id, []).append(
                (shape(s), s.ended_by == "correction"))

    fired = 0
    corrections = 0
    caught = 0
    calls_saved: list[int] = []

    for s in stretches:
        n = len(s.calls)
        if n < JUDGE_MIN_CALLS:
            continue

        key = (s.session_id, s.start_turn)
        label = gold_by_key.get(key)
        ground_truth_positive = label == "a"

        first_fire = None
        if judge_name == "overlap":
            for at_call in range(JUDGE_MIN_CALLS, n + 1):
                if judge_overlap(s, at_call=at_call, min_calls=JUDGE_MIN_CALLS).drifting:
                    first_fire = at_call
                    break
        elif judge_name == "knn":
            # Leave-one-session-out: this session's own stretches never appear in its own index.
            index = [entry for sid, entries in knn_index_by_session.items()
                     if sid != s.session_id for entry in entries]
            for at_call in range(JUDGE_MIN_CALLS, n + 1):
                if judge_knn(s, at_call=at_call, index=index, min_calls=JUDGE_MIN_CALLS).drifting:
                    first_fire = at_call
                    break
        else:
            raise ValueError(f"unknown judge: {judge_name!r}")

        if ground_truth_positive:
            corrections += 1
        if first_fire is not None:
            fired += 1
            if ground_truth_positive:
                caught += 1
                calls_saved.append(n - first_fire)

    recall = caught / corrections if corrections else 0.0
    precision = caught / fired if fired else 0.0
    median_calls_saved = statistics.median(calls_saved) if calls_saved else 0

    return {
        "judge": judge_name,
        "fired": fired,
        "corrections": corrections,
        "caught": caught,
        "recall": recall,
        "precision": precision,
        "median_calls_saved": median_calls_saved,
        "ships": gate(precision, median_calls_saved),
        "gold_skipped": gold_skipped,
        "gold_total": len(gold_by_key),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    rows = [evaluate(name, home=args.home, limit=args.limit) for name in ("overlap", "knn")]

    header = (f"{'judge':<10} {'fired':>6} {'corr':>6} {'caught':>7} {'recall':>7} "
              f"{'prec':>6} {'calls_saved':>12} {'ships':>6}")
    print(header)
    for row in rows:
        print(f"{row['judge']:<10} {row['fired']:>6} {row['corrections']:>6} {row['caught']:>7} "
              f"{row['recall']:>7.2f} {row['precision']:>6.2f} {row['median_calls_saved']:>12} "
              f"{str(row['ships']):>6}")
    if rows:
        skipped = rows[0]["gold_skipped"]
        total = rows[0]["gold_total"]
        print(f"\ngold records: {total}, skipped (no stretch on disk): {skipped}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
