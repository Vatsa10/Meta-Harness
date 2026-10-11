"""Label-Primed Query (paper App. B.1; highest-accuracy frontier point, label_primed_query_anchored.py).

One larger single call built from three parts. (1) A label primer lists every valid output label before any
example. (2) A coverage block holds, for each known label, the single memory example of that label most
similar to the query; similarity is TF-IDF over ALL of memory, so a label whose examples are all far from the
query still gets its best (if distant) example. (3) A contrastive block of up to `pair_count` query-anchored
pairs: each of the nearest examples is paired with the nearest example (to the query) carrying a different
label, exposing the local decision boundary. Memory grows only in `learn`.
"""

from meta_harness.retrieval import TfidfIndex


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


class Harness:
    def __init__(self, pair_count=4):
        self.pair_count = int(pair_count)
        self.memory = []

    def _coverage(self, ranked, labels):
        best = {}
        for r in ranked:
            best.setdefault(r.item["label"], r.item)
        return [best[x] for x in labels if x in best]

    def _pairs(self, ranked):
        pairs = []
        for r in ranked[: self.pair_count]:
            other = next((x.item for x in ranked if x.item["label"] != r.item["label"]), None)
            if other is not None and (r.item, other) not in pairs and (other, r.item) not in pairs:
                pairs.append((r.item, other))
        return pairs

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        ranked = _rank(self.memory, query)
        coverage = self._coverage(ranked, labels)
        pairs = self._pairs(ranked)
        prompt = "Valid labels: " + ", ".join(labels) + "\n\n"
        prompt += "Best example per label:\n" + "\n".join(_show(m) for m in coverage) + "\n\n"
        prompt += "Contrastive pairs:\n" + "\n".join(_show(a) + "\n" + _show(b) + "\n--" for a, b in pairs)
        prompt += f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:"
        label = _pick(model(prompt), labels)
        return _result(label, 1, len(coverage) + 2 * len(pairs), [prompt])

    def learn(self, task, result):
        if "label" in task:
            self.memory.append({"text": _query(task), "label": str(task["label"])})
