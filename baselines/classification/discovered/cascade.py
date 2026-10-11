"""Cascade Retrieval (paper Table 9, Pareto variant "Cascade Retrieval"; behaviour per App. B.1).

A cheap pass first, retrieval only when needed. Pass 1 is zero-shot (no examples): the model returns a label and
a confidence in [0, 1]. If the label is not one of the valid labels, or the stated confidence is missing or below
`threshold` (0.7), and memory is non-empty, the harness escalates to pass 2: a prompt with the `k` nearest
examples by TF-IDF. Otherwise pass 1 is the answer. Memory grows only in `learn`.

Design inferred from the name in Table 9; App. B.1 specifies only draft_verification and label_primed_query. Chosen constants: threshold=0.7, k=8 nearest examples on escalation.
"""

import re

from meta_harness.retrieval import TfidfIndex

_CONFIDENCE = re.compile(r"confidence\s*[:=]\s*([0-9]*\.?[0-9]+)\s*(%?)", re.IGNORECASE)


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


def _parse_zero_shot(output, labels):
    """(valid label or None, confidence in [0, 1] or None)."""
    text = str(output)
    lines = text.splitlines()
    line = next((x for x in lines if x.lower().startswith("label")), lines[0] if lines else "")
    answer = line.split(":", 1)[-1].strip().strip(".\"'") if ":" in line else line.strip()
    label = next((x for x in labels if x.lower() == answer.lower()), None)
    match = _CONFIDENCE.search(text)
    confidence = None
    if match:
        confidence = float(match.group(1)) / (100.0 if match.group(2) else 1.0)
    return label, confidence


class Harness:
    def __init__(self, threshold=0.7, k=8):
        self.threshold, self.k = float(threshold), int(k)
        self.memory = []

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        cheap = ("Valid labels: " + ", ".join(labels) + f"\n\nInput: {query}\n"
                 "Reply on two lines:\nLabel: <one valid label>\nConfidence: <number from 0 to 1>")
        output = str(model(cheap))
        label, confidence = _parse_zero_shot(output, labels)
        if label is not None and confidence is not None and confidence >= self.threshold:
            return _result(label, 1, 0, [cheap])
        if not self.memory:
            return _result(label if label is not None else _pick(output, labels), 1, 0, [cheap])
        near = [r.item for r in _rank(self.memory, query)[: self.k]]
        prompt = ("Valid labels: " + ", ".join(labels) + "\n\n" + "\n".join(_show(m) for m in near) +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        final = _pick(model(prompt), labels)
        return _result(final, 2, len(near), [cheap, prompt])

    def learn(self, task, result):
        if "label" in task:
            self.memory.append({"text": _query(task), "label": str(task["label"])})
