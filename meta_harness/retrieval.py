"""Standard-library TF-IDF and BM25 indexes used by the paper's harnesses."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

_MATH_TOKEN = re.compile(r"\\[A-Za-z]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\s]")
_TEXT_TOKEN = re.compile(r"[\w]+(?:'[\w]+)?", re.UNICODE)


def tokenize(text: str, math_mode: bool = False) -> list[str]:
    return [x.lower() for x in ( _MATH_TOKEN if math_mode else _TEXT_TOKEN).findall(str(text))]


@dataclass(frozen=True)
class Retrieved:
    index: int
    score: float
    item: Any


class TfidfIndex:
    def __init__(self, items: Sequence[Any], texts: Sequence[str] | None = None):
        self.items = list(items)
        self.texts = list(texts) if texts is not None else [str(x) for x in items]
        if len(self.items) != len(self.texts):
            raise ValueError("items and texts must have the same length")
        self.documents = [Counter(tokenize(text)) for text in self.texts]
        self.n = len(self.documents)
        self.document_frequency = Counter()
        for document in self.documents:
            self.document_frequency.update(document.keys())
        self.norms = [self._norm(document) for document in self.documents]

    def _idf(self, token: str) -> float:
        return math.log((1 + self.n) / (1 + self.document_frequency.get(token, 0))) + 1.0

    def _norm(self, document: Counter[str]) -> float:
        return math.sqrt(sum((count * self._idf(token)) ** 2 for token, count in document.items())) or 1.0

    def search(self, query: str, k: int = 5) -> list[Retrieved]:
        if k <= 0 or not self.items:
            return []
        query_counts = Counter(tokenize(query))
        weights = {token: count * self._idf(token) for token, count in query_counts.items()}
        query_norm = math.sqrt(sum(value * value for value in weights.values())) or 1.0
        result = []
        for index, document in enumerate(self.documents):
            numerator = sum(weights.get(token, 0.0) * count * self._idf(token) for token, count in document.items())
            result.append(Retrieved(index, numerator / (query_norm * self.norms[index]), self.items[index]))
        return sorted(result, key=lambda x: (-x.score, x.index))[:k]


class BM25Index:
    def __init__(self, items: Sequence[Any], texts: Sequence[str] | None = None, k1: float = 1.5, b: float = 0.75, math_mode: bool = True):
        self.items = list(items)
        self.texts = list(texts) if texts is not None else [str(x) for x in items]
        if len(self.items) != len(self.texts):
            raise ValueError("items and texts must have the same length")
        self.k1, self.b, self.math_mode = k1, b, math_mode
        self.documents = [Counter(tokenize(text, math_mode)) for text in self.texts]
        self.lengths = [sum(document.values()) for document in self.documents]
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))
        self.n = len(self.documents)
        self.document_frequency = Counter()
        for document in self.documents:
            self.document_frequency.update(document.keys())

    def _idf(self, token: str) -> float:
        frequency = self.document_frequency.get(token, 0)
        return math.log(1.0 + (self.n - frequency + 0.5) / (frequency + 0.5))

    def search(self, query: str, k: int = 10) -> list[Retrieved]:
        if k <= 0 or not self.items:
            return []
        query_terms = set(tokenize(query, self.math_mode))
        result = []
        for index, document in enumerate(self.documents):
            length_factor = 1 - self.b + self.b * self.lengths[index] / max(1.0, self.average_length)
            score = 0.0
            for token in query_terms:
                count = document.get(token, 0)
                if count:
                    score += self._idf(token) * (count * (self.k1 + 1)) / (count + self.k1 * length_factor)
            result.append(Retrieved(index, score, self.items[index]))
        return sorted(result, key=lambda x: (-x.score, x.index))[:k]


def reciprocal_rank_fusion(result_lists: Iterable[Sequence[Retrieved]], k: int = 60) -> list[Retrieved]:
    scores: dict[int, float] = {}
    items: dict[int, Any] = {}
    for results in result_lists:
        for rank, result in enumerate(results, 1):
            scores[result.index] = scores.get(result.index, 0.0) + 1.0 / (k + rank)
            items[result.index] = result.item
    return [Retrieved(index, score, items[index]) for index, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]
