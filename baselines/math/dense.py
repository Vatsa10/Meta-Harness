"""Dense-retrieval math baseline (paper Table 6): embedding nearest neighbours.

k comes from META_HARNESS_RETRIEVAL_K (the paper compares k=1 and k=5). Embeddings use
meta_harness.embeddings.embed (provider META_HARNESS_EMBED_PROVIDER, model
META_HARNESS_EMBED_MODEL). If embedding fails the harness degrades to zero-shot.
"""

import json
import os

from meta_harness.embeddings import cosine, embed


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
        self.provider = os.environ.get("META_HARNESS_EMBED_PROVIDER", "openai")
        self.embed_model = os.environ.get("META_HARNESS_EMBED_MODEL", "text-embedding-3-small")
        self.corpus = _load_corpus()
        self.vectors = None

    def _nearest(self, problem):
        if not self.corpus:
            return []
        if self.vectors is None:
            self.vectors = embed([str(x["problem"]) for x in self.corpus], self.provider, self.embed_model)
        query = embed([problem], self.provider, self.embed_model)[0]
        order = sorted(range(len(self.corpus)), key=lambda i: (-cosine(query, self.vectors[i]), i))
        return [self.corpus[i] for i in order[: self.k]]

    def run(self, task, model, trace):
        problem = str(task.get("problem", task.get("input", "")))
        try:
            shots = self._nearest(problem)
        except Exception as error:
            trace.event("dense_error", {"error": type(error).__name__})
            shots = []
        context = "\n\n".join(
            f"Reference problem:\n{x['problem']}\nSolution:\n{str(x['solution'])[:3000]}" for x in shots)
        prompt = ("Solve the problem rigorously. End with the final answer inside \\boxed{}.\n\n" +
                  (context + "\n\n" if context else "") + f"Problem:\n{problem}\n\nSolution:")
        trace.event("math_retrieval", {"retrieved": [x.get("id") for x in shots], "k": self.k})
        answer = str(model(prompt)).strip()
        trace.event("math_result", {"answer": answer})
        return answer
