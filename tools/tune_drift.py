"""Tune the drift judges against real corrections, behind a gate that can refuse all of them.

The correction labels `iter_stretches` produces are noisy - a hand-labelled audit found only
11 of 21 flagged stretches were genuine corrections (52%). This tool does NOT trust
`ended_by == "correction"` as ground truth for scoring. It joins each stretch against a
hand-labelled gold set instead (correction-gold.json: {session_id, project, start_turn, calls,
label, why}, label "a" = genuine correction, "c" = not a correction) and scores against that.

The join is by CONTAINMENT, not exact match: a gold record's (session_id, start_turn) is joined
to the stretch, in that same session, whose span actually contains it - `stretch.start_turn <=
gold_turn <= stretch.end_turn`. A stretch splitter fix can move a gold turn into an earlier,
merged stretch - it does not delete it - so an exact-match join would silently miscount some
genuine corrections as unmatched, or worse, as a false positive on the wrong stretch. Checking
only "the largest start_turn <= gold_turn" (with no upper bound) is not enough either: a gold
turn that falls in a GAP between stretches, or inside a stretch `iter_stretches` filtered out
for being under its own `min_calls`, would then be misjoined to whichever earlier, unrelated
stretch happens to start before it. `Stretch.end_turn` (meta_harness/waste.py) exists so this
tool can tell the difference. A stretch is a ground-truth POSITIVE if ANY gold record it
contains is labelled "a"; it is a NEGATIVE otherwise (a gold "c" only, or no gold record at all -
which is the case for the vast majority of stretches, since gold covers only what was
hand-labelled). A gold turn contained in no stretch at all fails to join, and is reported with a
reason, never silently dropped or misattributed.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from meta_harness.cc_history import claude_home, is_artifact_project
from meta_harness.drift import judge_knn, judge_overlap, shape
from meta_harness.waste import iter_stretches

# Fixed location of the hand-labelled gold set (an sdd planning artifact, not part of the
# package - shared across every lane's worktree). Overridable via `evaluate(gold_path=...)` or
# `--gold` so a missing/relocated file is the caller's problem to fix, not this tool's to
# silently paper over.
GOLD_PATH = Path(
    r"D:\Files\Vatsa\Projects\Meta-Harness\.superpowers\sdd\2026-09-26-drift-and-waste"
    r"\correction-gold.json"
)

PRECISION_BAR: float = 0.5
MIN_CALLS_SAVED: int = 3

# The judge's own min_calls gate - a stretch shorter than this is never even considered, in
# either judge, so it is excluded from evaluation entirely rather than counted as a miss.
JUDGE_MIN_CALLS = 8

# The model judge (meta_harness.drift.judge_model) needs a live `complete` callable and cannot
# be scored offline against a transcript store - there is nothing to inject it with here. It is
# reported as unevaluated, not silently left out of the table.
UNEVALUATED_JUDGES = ("model",)


def gate(precision: float, median_calls_saved: float) -> bool:
    """Whether a judge ships. Stated before tuning, and not softened after seeing the numbers:
    a judge ships only when it clears BOTH bars. No threshold argument - this exists to be able
    to refuse, and a caller that could pass one in could tune the gate away."""
    return precision >= PRECISION_BAR and median_calls_saved >= MIN_CALLS_SAVED


class GoldFileMissing(RuntimeError):
    """Raised when the gold set can't be read - never treated as "zero positives"."""


def _load_gold(gold_path: Path) -> list[dict[str, Any]]:
    if not gold_path.exists():
        raise GoldFileMissing(f"gold file not found: {gold_path}")
    try:
        with gold_path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except ValueError as exc:
        raise GoldFileMissing(f"gold file at {gold_path} is not valid JSON: {exc}") from exc


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


def _join_gold_by_containment(gold: list[dict[str, Any]], stretches: list) -> dict[str, Any]:
    """Join each gold record to the stretch, in the same session, whose span actually CONTAINS
    its start_turn: `stretch.start_turn <= gold_turn <= stretch.end_turn`. A gold turn that sits
    in a gap between stretches - or inside a stretch short enough that `iter_stretches` filtered
    it out entirely - joins to nothing: it is reported as FAILED, with a reason, never silently
    attributed to the nearest unrelated stretch that merely starts before it.

    Returns the set of stretch identities that are ground-truth positive, plus join accounting:
    how many gold records joined, how many joined to a DIFFERENT start_turn than their own
    (moved by a splitter change that merged stretches), and the records (with reasons) that
    failed to join at all.
    """
    by_session: dict[str, list] = collections.defaultdict(list)
    for s in stretches:
        by_session[s.session_id].append(s)
    for entries in by_session.values():
        entries.sort(key=lambda s: s.start_turn)

    contained_labels: dict[int, list[str]] = collections.defaultdict(list)
    moved = 0
    failed: list[dict[str, Any]] = []

    for record in gold:
        turn = record["start_turn"]
        candidates = by_session.get(record["session_id"], [])
        containing = next(
            (s for s in candidates if s.start_turn <= turn <= s.end_turn), None)
        if containing is None:
            failed.append({
                **record,
                "reason": ("no stretch on disk contains turn " + str(turn) + " for this "
                           "session - it falls in a gap between stretches, or inside a stretch "
                           "iter_stretches filtered out for being under min_calls"),
            })
            continue
        if containing.start_turn != turn:
            moved += 1
        contained_labels[id(containing)].append(record["label"])

    positive_ids = {stretch_id for stretch_id, labels in contained_labels.items() if "a" in labels}
    matched = len(gold) - len(failed)
    return {
        "positive_ids": positive_ids,
        "matched": matched,
        "moved": moved,
        "failed": failed,
    }


def evaluate(judge_name: str, home: Path | None = None, limit: int | None = None,
             gold_path: Path = GOLD_PATH) -> dict[str, Any]:
    """Score one judge ("overlap" or "knn") against the hand-labelled gold set, not against the
    noisy `ended_by` label. Returns fired/corrections/caught/recall/precision/median_calls_saved,
    whether the judge ships, and the gold join accounting (matched/moved/failed)."""
    root = Path(home) if home is not None else claude_home() / "projects"
    paths = _transcripts(root, limit)
    stretches = _all_stretches(paths)

    gold = _load_gold(gold_path)
    join = _join_gold_by_containment(gold, stretches)
    positive_ids = join["positive_ids"]

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

        ground_truth_positive = id(s) in positive_ids

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
        "gold_total": len(gold),
        "gold_matched": join["matched"],
        "gold_moved": join["moved"],
        "gold_failed": len(join["failed"]),
        "gold_failed_records": join["failed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gold", type=Path, default=GOLD_PATH, dest="gold_path")
    args = parser.parse_args(argv)

    try:
        rows = [evaluate(name, home=args.home, limit=args.limit, gold_path=args.gold_path)
                for name in ("overlap", "knn")]
    except GoldFileMissing as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    header = (f"{'judge':<10} {'fired':>6} {'corr':>6} {'caught':>7} {'recall':>7} "
              f"{'prec':>6} {'calls_saved':>12} {'ships':>6}")
    print(header)
    for row in rows:
        print(f"{row['judge']:<10} {row['fired']:>6} {row['corrections']:>6} {row['caught']:>7} "
              f"{row['recall']:>7.2f} {row['precision']:>6.2f} {row['median_calls_saved']:>12} "
              f"{str(row['ships']):>6}")
    for name in UNEVALUATED_JUDGES:
        print(f"{name:<10} {'n/a':>6} {'n/a':>6} {'n/a':>7} {'n/a':>7} {'n/a':>6} {'n/a':>12} "
              f"{'False':>6}  (unevaluated: needs live model calls, cannot be scored offline)")

    if rows:
        row = rows[0]
        print(f"\ngold records: {row['gold_total']}, joined: {row['gold_matched']} "
              f"(moved to a different stretch than their own start_turn: {row['gold_moved']}), "
              f"failed to join: {row['gold_failed']}")
        for record in row["gold_failed_records"]:
            print(f"  failed: session={record['session_id']} start_turn={record['start_turn']} "
                  f"label={record['label']} - {record['reason']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
