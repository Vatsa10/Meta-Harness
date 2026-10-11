"""Cluster Coverage (paper Table 9, Pareto variant "Cluster Coverage"; behaviour per App. B.1).

Broad coverage of the memory instead of only the neighbourhood of the query. Memory is embedded as TF-IDF
vectors and greedily k-center clustered: start from the first example, repeatedly add the example least similar
to every chosen centre, up to `clusters` centres. The prompt shows one representative (the centre) per cluster
plus the `k` nearest examples to the query, deduplicated. Memory grows only in `learn`.

Design inferred from the name in Table 9; App. B.1 specifies only draft_verification and label_primed_query. Chosen constants: clusters=5 centres, k=4 nearest examples, first memory row seeds the k-center.
"""

import math
from collections import Counter

from meta_harness.retrieval import TfidfIndex, tokenize


def _query(task):
    return str(task.get("input", task.get("text", "")))


def _labels(task, memory):
    declared = task.get("labels")
    if declared:
        return [str(x) for x in declared]
    return sorted({m["label"] for m in memory})


def _pick(output, labels):
    text = str(output).strip()
    low = text.lower()
    hits = [(low.find(x.lower()), -len(x), x) for x in labels if x.lower() in low]
    if hits:
        return min(hits)[2]
    return text.splitlines()[0].strip() if text else (labels[0] if labels else text)


def _rank(memory, query):
    """Every memory row ordered by TF-IDF similarity to the query."""
    if not memory:
        return []
    return TfidfIndex(memory, [m["text"] for m in memory]).search(query, len(memory))


def _show(m):
    return f"Example: {m['text']}\nLabel: {m['label']}"


def _result(label, calls, examples, prompts):
    return {"output": label, "calls": calls, "examples": examples,
            "prompt_chars": sum(len(p) for p in prompts)}


def _vectors(texts):
    docs = [Counter(tokenize(t)) for t in texts]
    df = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    vectors = []
    for d in docs:
        v = {tok: c * (math.log((1 + n) / (1 + df[tok])) + 1.0) for tok, c in d.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vectors.append({tok: x / norm for tok, x in v.items()})
    return vectors


def _cos(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b.get(tok, 0.0) for tok, x in a.items())


def _k_center(vectors, count):
    """Greedy k-center: indices of the chosen centres, first example first."""
    if not vectors:
        return []
    centres = [0]
    nearest = [_cos(v, vectors[0]) for v in vectors]
    while len(centres) < min(count, len(vectors)):
        far = min((i for i in range(len(vectors)) if i not in centres), key=lambda i: (nearest[i], i))
        centres.append(far)
        nearest = [max(nearest[i], _cos(vectors[i], vectors[far])) for i in range(len(vectors))]
    return centres


class Harness:
    def __init__(self, clusters=5, k=4):
        self.clusters, self.k = int(clusters), int(k)
        self.memory = []

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        centres = [self.memory[i] for i in _k_center(_vectors([m["text"] for m in self.memory]), self.clusters)]
        near = [r.item for r in _rank(self.memory, query)[: self.k]]
        chosen = centres + [m for m in near if m not in centres]
        prompt = ("Valid labels: " + ", ".join(labels) + "\n\n" + "\n".join(_show(m) for m in chosen) +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        label = _pick(model(prompt), labels)
        return _result(label, 1, len(chosen), [prompt])

    def learn(self, task, result):
        if "label" in task:
            self.memory.append({"text": _query(task), "label": str(task["label"])})
