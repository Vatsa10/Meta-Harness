"""Relevance + Contrastive (paper Table 9, Pareto variant "Relevance + Contrastive"; behaviour per App. B.1).

Relevance-ranked examples plus targeted counter-evidence. The prompt shows the `k` nearest examples by TF-IDF.
The leading label is that of the nearest example; the remaining labels are ranked by how similar their best
example is to the query, and the top 3 of them are the confusable labels. For each, the nearest example of that
label is added in a separate block, so the model sees the closest competitor of every likely confusion.
Memory grows only in `learn`.

Design inferred from the name in Table 9; App. B.1 specifies only draft_verification and label_primed_query. Chosen constants: k=6 nearest examples, 3 confusable labels.
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
    def __init__(self, k=6, confusable=3):
        self.k, self.confusable = int(k), int(confusable)
        self.memory = []

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        ranked = _rank(self.memory, query)
        alternatives = []
        if ranked:
            seen = {ranked[0].item["label"]}
            for r in ranked:
                if r.item["label"] not in seen:
                    seen.add(r.item["label"])
                    alternatives.append(r.item)
                if len(alternatives) >= self.confusable:
                    break
        top = [r.item for r in ranked[: self.k] if r.item not in alternatives]
        prompt = ("Valid labels: " + ", ".join(labels) + "\n\nMost relevant examples:\n" +
                  "\n".join(_show(m) for m in top) + "\n\nNearest example of each confusable label:\n" +
                  "\n".join(_show(m) for m in alternatives) +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        label = _pick(model(prompt), labels)
        return _result(label, 1, len(top) + len(alternatives), [prompt])

    def learn(self, task, result):
        if "label" in task:
            self.memory.append({"text": _query(task), "label": str(task["label"])})
