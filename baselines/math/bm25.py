"""BM25 retrieval math baseline (paper Table 6, "BM25 Retrieval").

Corpus path comes from META_HARNESS_CORPUS (JSONL with "problem" and "solution" fields).
With no corpus configured the harness degrades to zero-shot, which keeps it valid.
"""

import json
import os

from meta_harness.retrieval import BM25Index


def _load_corpus():
    path = os.environ.get("META_HARNESS_CORPUS", "")
    if not path or not os.path.isfile(path):
        return []
    items = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("problem") and record.get("solution"):
                items.append(record)
    return items


class Harness:
    def __init__(self, k=None):
        self.k = int(k if k is not None else os.environ.get("META_HARNESS_RETRIEVAL_K", "3"))
        self.corpus = _load_corpus()
        self.index = BM25Index(self.corpus, [str(x["problem"]) for x in self.corpus],
                               math_mode=True) if self.corpus else None

    def run(self, task, model, trace):
        problem = str(task.get("problem", task.get("input", "")))
        retrieved = self.index.search(problem, self.k) if self.index else []
        context = "\n\n".join(
            f"Reference problem:\n{x.item['problem']}\nSolution:\n{str(x.item['solution'])[:3000]}"
            for x in retrieved)
        prompt = ("Solve the problem rigorously. Reference solutions may help. "
                  "End with the final answer inside \\boxed{}.\n\n" +
                  (context + "\n\n" if context else "") + f"Problem:\n{problem}\n\nSolution:")
        trace.event("math_retrieval", {"retrieved": [x.index for x in retrieved], "k": self.k})
        answer = str(model(prompt)).strip()
        trace.event("math_result", {"answer": answer})
        return answer
