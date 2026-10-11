"""Math retrieval harness from the paper appendices (section 4.2)."""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping, Sequence

from .cls_harnesses import _call
from .retrieval import BM25Index, Retrieved


class MathRetrievalHarness:
    """The paper's lexical router with geometry, number-theory, combinatorics, and default routes."""

    def __init__(self, corpus: Sequence[Mapping[str, Any]], default_k: int = 3):
        self.corpus = [dict(item) for item in corpus if item.get("problem") and item.get("solution")]
        self.index = BM25Index(self.corpus, [str(item["problem"]) for item in self.corpus], math_mode=True)
        self.default_k = default_k

    @staticmethod
    def route(problem: str) -> str:
        text = problem.lower()
        if re.search(r"\b(circle|triangle|angle|polygon|perpendicular|parallel|geometry|area)\b|\\angle|\\triangle", text):
            return "geometry"
        if re.search(r"\b(prime|divisibility|congruen|modulo|integer|gcd|diophantine|number theory)\b", text):
            return "number_theory"
        if re.search(r"\b(choose|subset|permutation|probability|graph|color|pigeonhole|combinator)\b", text):
            return "combinatorics"
        return "default"

    def _retrieve(self, problem: str, route: str) -> list[Retrieved]:
        if route == "geometry":
            raw = self.index.search(problem, 10)
            hard = [x for x in raw if float(x.item.get("difficulty", 0) or 0) >= 6]
            return (hard[:1] + raw[:3])[:3]
        raw = self.index.search(problem, {"combinatorics": 20, "number_theory": 12, "default": 10}[route])
        if route == "combinatorics":
            unique, seen = [], set()
            for result in raw:
                signature = re.sub(r"\W+", " ", str(result.item["problem"]).lower()).strip()[:120]
                if signature not in seen:
                    seen.add(signature)
                    unique.append(result)
            raw = unique[:8]
        def rank(result: Retrieved) -> float:
            difficulty = float(result.item.get("difficulty", 0) or 0)
            technique = 0.25 if "therefore" in str(result.item.get("solution", "")).lower() else 0.0
            return result.score + 0.03 * difficulty + (technique if route == "number_theory" else 0.0)
        return sorted(raw, key=lambda x: (-rank(x), x.index))[:self.default_k]

    def run(self, task: Mapping[str, Any], model: Callable[..., Any], trace: Any) -> str:
        problem = str(task.get("problem", task.get("input", "")))
        route = self.route(problem)
        examples = self._retrieve(problem, route)
        context = "\n\n".join(f"Reference problem:\n{x.item['problem']}\nSolution:\n{str(x.item['solution'])[:3000]}" for x in examples)
        prompt = f"Solve rigorously. Route: {route}\n\n{context}\n\nProblem:\n{problem}\nSolution:"
        trace.event("math_retrieval", {"route": route, "retrieved": [x.index for x in examples]})
        answer = _call(model, prompt)
        trace.event("math_result", {"answer": answer})
        return answer


__all__ = ["MathRetrievalHarness"]
