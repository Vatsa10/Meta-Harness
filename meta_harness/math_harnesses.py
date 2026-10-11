"""Math retrieval harness from the paper appendices (section 4.2, App. B.2 and C.2).

Spec (paper pages 19-21 and 23):

* Each problem is assigned to exactly one of four routes by lightweight lexical predicates
  (keyword sets plus a few regexes for geometry notation). The paper lists the routes in the
  order combinatorics, geometry, number theory, default (algebra and other); `route` tests them
  in that order. Routes never aggregate outputs: only the selected route retrieves examples.
* All routes use BM25 with a math-aware tokenizer that keeps LaTeX tokens atomic, over a corpus
  restricted to entries with non-empty solutions of at most 4000 characters. Retrieved
  solutions are truncated to 3000 characters when inserted into the prompt.
* Combinatorics: BM25@20, deduplicate to 8, rerank by lexical score and difficulty, keep 3.
* Geometry: one hard reference plus up to two raw BM25 neighbours, no rerank. The hard
  reference comes from a SEPARATE index built only from NuminaMath-sourced items with
  difficulty > 6; it is not a filter over the main top-10, so it can be an item the main
  index would not have ranked at all. Results are deduplicated by id.
* Number theory: BM25@12, rerank by lexical score, difficulty and a bonus for solutions that
  state a technique early (a technique-lexicon word in the first 300 characters), keep 3.
* Default: BM25@10, rerank by lexical score and difficulty, adaptive K: K grows from 1 to
  `max_k` while the next candidate's score stays within `ratio` of the top score.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .cls_harnesses import _call
from .retrieval import BM25Index, Retrieved

MAX_ITEM_CHARS = 4000
SOLUTION_PROMPT_CHARS = 3000
TECHNIQUE_WINDOW = 300
HARD_DIFFICULTY = 6.0
DIFFICULTY_WEIGHT = 0.03
TECHNIQUE_BONUS = 0.25
ADAPTIVE_RATIO = 0.8

TECHNIQUE_LEXICON = (
    "induction", "pigeonhole", "invariant", "extremal", "vieta", "am-gm", "amgm", "cauchy",
    "modulo", "bijection", "contradiction", "infinite descent", "double counting", "telescop",
    "jensen", "schur", "chinese remainder", "fermat", "lifting the exponent",
    "generating function", "parity",
)

_COMBINATORICS = re.compile(
    r"\b(?:choose|binomial|subsets?|permutations?|arrangements?|probability|colou?rings?|"
    r"colou?red|pigeonhole|combinatori\w*|tournament|vertices|edges|distinct ways|"
    r"number of ways|how many ways)\b")
_GEOMETRY = re.compile(
    r"\b(?:circles?|triangles?|angles?|polygons?|perpendicular|parallel|geometry|areas?|"
    r"circumcircle|incircle|tangent|chord|quadrilateral|hexagon|bisector|altitude|"
    r"circumcent\w*|incent\w*)\b|\\angle|\\triangle|\\parallel|\\perp|\\odot|\\overline\{[A-Z]{2}\}")
_NUMBER_THEORY = re.compile(
    r"\b(?:primes?|divisib\w*|divides|divisors?|congruen\w*|modulo|integers?|gcd|lcm|"
    r"diophantine|number theory|remainders?|residues?)\b|\\pmod|\\equiv|\\mid")


def technique_stated_early(solution: str) -> bool:
    """True when a technique-lexicon word appears in the first 300 characters of a solution."""
    head = str(solution)[:TECHNIQUE_WINDOW].lower()
    return any(term in head for term in TECHNIQUE_LEXICON)


def adaptive_k(scores: Sequence[float], max_k: int = 3, ratio: float = ADAPTIVE_RATIO) -> int:
    """Grow K from 1 to max_k while the next score is within `ratio` of the top score."""
    if not scores:
        return 0
    top = scores[0]
    k = 1
    while k < min(max_k, len(scores)) and top > 0 and scores[k] >= ratio * top:
        k += 1
    return k


def _is_numina(item: Mapping[str, Any]) -> bool:
    labels = [str(item[key]) for key in ("source", "dataset", "origin") if item.get(key)]
    return not labels or any("numina" in label.lower() for label in labels)


def _difficulty(item: Mapping[str, Any]) -> float:
    try:
        return float(item.get("difficulty", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


class MathRetrievalHarness:
    """The paper's lexical router with geometry, number-theory, combinatorics, and default routes."""

    def __init__(self, corpus: Sequence[Mapping[str, Any]], default_k: int = 3,
                 adaptive_ratio: float = ADAPTIVE_RATIO):
        self.corpus = []
        for position, item in enumerate(corpus):
            problem, solution = str(item.get("problem") or ""), str(item.get("solution") or "")
            if not problem or not solution or len(problem) > MAX_ITEM_CHARS or len(solution) > MAX_ITEM_CHARS:
                continue
            record = dict(item)
            if record.get("id") in (None, ""):
                record["id"] = position
            self.corpus.append(record)
        self.index = BM25Index(self.corpus, [str(x["problem"]) for x in self.corpus], math_mode=True)
        hard = [x for x in self.corpus if _difficulty(x) > HARD_DIFFICULTY and _is_numina(x)]
        self.hard_index = BM25Index(hard, [str(x["problem"]) for x in hard], math_mode=True)
        self.default_k = default_k
        self.adaptive_ratio = adaptive_ratio

    @staticmethod
    def route(problem: str) -> str:
        text = problem.lower()
        if _COMBINATORICS.search(text):
            return "combinatorics"
        if _GEOMETRY.search(text):
            return "geometry"
        if _NUMBER_THEORY.search(text):
            return "number_theory"
        return "default"

    def _geometry(self, problem: str) -> list[Retrieved]:
        chosen, seen = [], set()
        for result in self.hard_index.search(problem, 1) + self.index.search(problem, 10):
            ident = result.item["id"]
            if ident in seen:
                continue
            seen.add(ident)
            chosen.append(result)
            if len(chosen) == 3:
                break
        return chosen

    @staticmethod
    def _rank(result: Retrieved, route: str) -> float:
        bonus = TECHNIQUE_BONUS if route == "number_theory" and technique_stated_early(result.item["solution"]) else 0.0
        return result.score + DIFFICULTY_WEIGHT * _difficulty(result.item) + bonus

    def _retrieve(self, problem: str, route: str) -> list[Retrieved]:
        if route == "geometry":
            return self._geometry(problem)
        raw = self.index.search(problem, {"combinatorics": 20, "number_theory": 12, "default": 10}[route])
        if route == "combinatorics":
            unique, seen_ids, seen = [], set(), set()
            for result in raw:
                signature = re.sub(r"\W+", " ", str(result.item["problem"]).lower()).strip()[:120]
                if result.item["id"] in seen_ids or signature in seen:
                    continue
                seen_ids.add(result.item["id"])
                seen.add(signature)
                unique.append(result)
            raw = unique[:8]
        ranked = sorted(raw, key=lambda x: (-self._rank(x, route), x.index))
        if route == "default":
            scores = [self._rank(x, route) for x in ranked]
            return ranked[:adaptive_k(scores, self.default_k, self.adaptive_ratio)]
        return ranked[:3]

    def run(self, task: Mapping[str, Any], model: Callable[..., Any], trace: Any) -> str:
        problem = str(task.get("problem", task.get("input", "")))
        route = self.route(problem)
        examples = self._retrieve(problem, route)
        context = "\n\n".join(
            f"Reference problem:\n{x.item['problem']}\nSolution:\n{str(x.item['solution'])[:SOLUTION_PROMPT_CHARS]}"
            for x in examples)
        prompt = (f"Solve rigorously. End with the final answer inside \\boxed{{}}. Route: {route}\n\n"
                  f"{context}\n\nProblem:\n{problem}\nSolution:")
        trace.event("math_retrieval", {"route": route, "retrieved": [x.item["id"] for x in examples]})
        answer = _call(model, prompt)
        trace.event("math_result", {"answer": answer})
        return answer


__all__ = ["MathRetrievalHarness", "adaptive_k", "technique_stated_early"]
