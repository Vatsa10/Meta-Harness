"""Few-shot classification baseline (paper Table 2, "Few-Shot (N)").

N comes from META_HARNESS_FEW_SHOT_N and must be one of 4, 8, 16, 32 or all (default 8); `all`
puts the whole memory in the prompt. `run` only reads memory; `learn` appends the labelled task.
"""

import os

VALID_N = ("4", "8", "16", "32", "all")


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


def parse_n(value):
    text = str(value).strip().lower()
    if text not in VALID_N:
        raise ValueError(f"META_HARNESS_FEW_SHOT_N must be one of {', '.join(VALID_N)}; got {value!r}")
    return None if text == "all" else int(text)


class Harness:
    def __init__(self, shots=None):
        raw = shots if shots is not None else os.environ.get("META_HARNESS_FEW_SHOT_N", "8")
        # an explicit integer (tests, programmatic use) bypasses the env value set
        self.shots = shots if isinstance(shots, int) and shots > 0 else parse_n(raw)  # None means the whole memory
        self.memory = []

    def _examples(self):
        if self.shots is None:
            return list(self.memory)
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
        trace.event("few_shot_result", {"prediction": prediction})
        return prediction

    def learn(self, task, result):
        if "label" in task:
            self.memory.append((str(task.get("input", "")), str(task["label"])))
