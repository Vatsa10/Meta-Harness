"""Random few-shot math baseline: k corpus examples drawn uniformly, seeded by the problem text."""

import json
import os
import random


def _load_corpus():
    path = os.environ.get("META_HARNESS_CORPUS", "")
    if not path or not os.path.isfile(path):
        return []
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if (isinstance(record, dict) and record.get("problem") and record.get("solution")
                    and len(str(record["solution"])) <= 4000):
                items.append(record)
    return items


class Harness:
    def __init__(self, k=None):
        self.k = int(k if k is not None else os.environ.get("META_HARNESS_RETRIEVAL_K", "3"))
        self.corpus = _load_corpus()

    def run(self, task, model, trace):
        problem = str(task.get("problem", task.get("input", "")))
        rng = random.Random(problem)
        shots = rng.sample(self.corpus, min(self.k, len(self.corpus))) if self.corpus else []
        context = "\n\n".join(
            f"Reference problem:\n{x['problem']}\nSolution:\n{str(x['solution'])[:3000]}" for x in shots)
        prompt = ("Solve the problem rigorously. End with the final answer inside \\boxed{}.\n\n" +
                  (context + "\n\n" if context else "") + f"Problem:\n{problem}\n\nSolution:")
        trace.event("math_retrieval", {"retrieved": [x.get("id") for x in shots], "k": self.k})
        answer = str(model(prompt)).strip()
        trace.event("math_result", {"answer": answer})
        return answer
