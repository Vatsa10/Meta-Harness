"""CoT Replay (paper Table 9, Pareto variant "CoT Replay"; behaviour per App. B.1).

Memory stores a short model-written rationale next to each labelled example. `learn(task, result, model)` spends
one extra model call per training example asking for a one-sentence reason the input has its true label, and
keeps the reply. At inference the `k` nearest examples by TF-IDF are shown with their rationales replayed
("Reasoning: ..."), so the model sees worked reasoning rather than bare labels.

Design inferred from the name in Table 9; App. B.1 specifies only draft_verification and label_primed_query. Chosen constants: k=6 nearest examples, rationale capped at 300 characters.
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
    def __init__(self, k=6, max_rationale_chars=300):
        self.k = int(k)
        self.max_chars = int(max_rationale_chars)
        self.memory = []

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        near = [r.item for r in _rank(self.memory, query)[: self.k]]
        blocks = [_show(m) + (f"\nReasoning: {m['rationale']}" if m.get("rationale") else "") for m in near]
        prompt = ("Valid labels: " + ", ".join(labels) + "\n\n" + "\n\n".join(blocks) +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        label = _pick(model(prompt), labels)
        return _result(label, 1, len(near), [prompt])

    def learn(self, task, result, model):
        if "label" not in task:
            return
        truth = str(task["label"])
        text = _query(task)
        ask = (f"Input: {text}\nTrue label: {truth}\n"
               "In one short sentence, explain why this input has that label.\nReason:")
        rationale = str(model(ask)).strip().replace("\n", " ")[: self.max_chars]
        self.memory.append({"text": text, "label": truth, "rationale": rationale})
