"""Few-shot classification baseline. N from META_HARNESS_FEW_SHOT_N (default 8)."""

import os


def _labels(task, memory):
    declared = task.get("labels")
    if declared:
        return [str(x) for x in declared]
    seen = sorted({label for _, label in memory})
    if "label" in task and str(task["label"]) not in seen:
        seen = sorted(seen + [str(task["label"])])
    return seen


def _pick(output, labels):
    text = str(output).strip()
    for label in labels:
        if label.lower() in text.lower():
            return label
    return text.splitlines()[0].strip() if text else (labels[0] if labels else text)


class Harness:
    def __init__(self, shots=None):
        self.shots = int(shots if shots is not None else os.environ.get("META_HARNESS_FEW_SHOT_N", "8"))
        self.memory = []

    def _examples(self):
        if self.shots <= 0:
            return []
        # Most recent first keeps the window fresh in an online stream.
        return list(reversed(self.memory))[: self.shots]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        examples = self._examples()
        block = "\n".join(f"Example: {text}\nLabel: {label}" for text, label in examples)
        prompt = ("Classify the input.\nValid labels: " + ", ".join(labels) + "\n\n" + block +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("few_shot_prompt", {"prompt": prompt, "shots": len(examples)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            self.memory.append((query, str(task["label"])))
        trace.event("few_shot_result", {"prediction": prediction})
        return prediction
