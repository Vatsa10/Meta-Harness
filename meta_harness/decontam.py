"""Corpus deduplication and decontamination (paper appendix C.2).

A corpus problem is dropped when its normalised text shares the first `prefix_chars`
characters with a held-out problem, or when the Jaccard similarity of their word-shingle sets
reaches `threshold` (paper: exact prefix match, then fuzzy Jaccard at 0.8). Candidate pairs come
from an inverted shingle index, so the cost is not O(corpus * held-out).
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

_DISPLAY = re.compile(r"\\[\(\)\[\]]|\$+")
_WRAPPER = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|boxed|mbox|operatorname)\s*\{([^{}]*)\}")
_SPACING = re.compile(r"\\(?:left|right|,|;|:|!|quad|qquad)(?![a-zA-Z])")
_SPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Lowercase, strip LaTeX math delimiters and text wrappers, collapse whitespace."""
    text = str(text).lower()
    for _ in range(3):  # nested wrappers such as \textbf{\text{x}}
        text = _WRAPPER.sub(r"\1", text)
    text = _DISPLAY.sub(" ", text)
    text = _SPACING.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def shingles(text: str, n: int = 5) -> set[str]:
    """Word n-grams of `normalise(text)`; a text shorter than n words is one shingle."""
    words = normalise(text).split()
    if not words:
        return set()
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prefix(normalised: str, prefix_chars: int) -> str | None:
    """Key for the prefix test; None when the text is too short to be a meaningful prefix."""
    if prefix_chars <= 0 or not normalised:
        return None
    return normalised[:prefix_chars] if len(normalised) >= prefix_chars else normalised


class _Index:
    """Inverted shingle index plus a prefix table over a growing set of reference problems."""

    def __init__(self, prefix_chars: int, threshold: float):
        self.prefix_chars = prefix_chars
        self.threshold = threshold
        self.prefixes: set[str] = set()
        self.postings: dict[str, list[int]] = {}
        self.sets: list[set[str]] = []

    def add(self, text: str) -> None:
        key = _prefix(normalise(text), self.prefix_chars)
        if key is not None:
            self.prefixes.add(key)
        members = shingles(text)
        number = len(self.sets)
        self.sets.append(members)
        for shingle in members:
            self.postings.setdefault(shingle, []).append(number)

    def matches(self, text: str) -> bool:
        key = _prefix(normalise(text), self.prefix_chars)
        if key is not None and key in self.prefixes:
            return True
        members = shingles(text)
        candidates: set[int] = set()
        for shingle in members:
            candidates.update(self.postings.get(shingle, ()))
        return any(jaccard(members, self.sets[number]) >= self.threshold for number in candidates)


def decontaminate(corpus: Iterable[dict], held_out: Sequence[str], prefix_chars: int = 60,
                  threshold: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Drop a corpus item when its normalised problem shares the first `prefix_chars` with any
    held-out problem, or its shingle Jaccard with any held-out problem is >= threshold.
    Candidate pairs come from an inverted shingle index (no O(N*M) scan).
    Returns (kept, removed) and the caller writes removed ids to decontam_report.json."""
    index = _Index(prefix_chars, threshold)
    for problem in held_out:
        index.add(problem)
    kept: list[dict] = []
    removed: list[dict] = []
    for item in corpus:
        (removed if index.matches(str(item.get("problem", ""))) else kept).append(item)
    return kept, removed


def dedup(corpus: Iterable[dict], threshold: float = 0.8, prefix_chars: int = 60) -> list[dict]:
    """Same test, within the corpus: the first of any near-duplicate group is kept."""
    index = _Index(prefix_chars, threshold)
    kept: list[dict] = []
    for item in corpus:
        problem = str(item.get("problem", ""))
        if index.matches(problem):
            continue
        index.add(problem)
        kept.append(item)
    return kept


__all__ = ["normalise", "shingles", "jaccard", "decontaminate", "dedup"]
