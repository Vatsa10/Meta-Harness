"""Error-Annotated (paper Table 9, Pareto variant "Error-Annotated"; behaviour per App. B.1).

A memory harness that records the model's own mistakes. In `learn`, each training example is stored with the
label the model predicted earlier (taken from the run result) when that prediction was wrong. At inference the
`k` nearest examples by TF-IDF are shown, and any example with a recorded wrong prediction is annotated
"commonly confused with X", so the prompt exposes the confusions the model actually makes near this query.

Design inferred from the name in Table 9; App. B.1 specifies only draft_verification and label_primed_query. Chosen constants: k=8 nearest examples.
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
    def __init__(self, k=8):
        self.k = int(k)
        self.memory = []

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        near = [r.item for r in _rank(self.memory, query)[: self.k]]
        blocks = []
        for m in near:
            note = f"\n(commonly confused with {m['confused_with']})" if m.get("confused_with") else ""
            blocks.append(_show(m) + note)
        prompt = ("Valid labels: " + ", ".join(labels) + "\n\n" + "\n".join(blocks) +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        label = _pick(model(prompt), labels)
        return _result(label, 1, len(near), [prompt])

    def learn(self, task, result):
        if "label" not in task:
            return
        truth = str(task["label"])
        predicted = str(result.get("output", "")).strip() if isinstance(result, dict) else str(result).strip()
        row = {"text": _query(task), "label": truth}
        if predicted and predicted != truth:
            row["confused_with"] = predicted
        self.memory.append(row)
