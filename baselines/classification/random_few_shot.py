"""Random few-shot baseline (paper Table 6, "Random Few-Shot", applied to classification).

N examples (META_HARNESS_FEW_SHOT_N, default 8; 4/8/16/32/all) are sampled from memory with
random.Random(seed derived from a SHA-256 of the input), so the draw is deterministic per task and
independent of PYTHONHASHSEED. `run` only reads memory; `learn` appends the labelled task.
"""

import hashlib
import os
import random

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


def seed_for(text):
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


class Harness:
    def __init__(self, shots=None):
        raw = shots if shots is not None else os.environ.get("META_HARNESS_FEW_SHOT_N", "8")
        # an explicit integer (tests, programmatic use) bypasses the env value set
        self.shots = shots if isinstance(shots, int) and shots > 0 else parse_n(raw)
        self.memory = []

    def _examples(self, query):
        if self.shots is None or self.shots >= len(self.memory):
            return list(self.memory)
        return random.Random(seed_for(query)).sample(self.memory, self.shots)

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        examples = self._examples(query)
        block = "\n".join(f"Example: {text}\nLabel: {label}" for text, label in examples)
        prompt = ("Classify the input.\nValid labels: " + ", ".join(labels) + "\n\n" + block +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("random_few_shot_prompt", {"prompt": prompt, "shots": len(examples)})
        prediction = _pick(model(prompt), labels)
        trace.event("random_few_shot_result", {"prediction": prediction})
        return prediction

    def learn(self, task, result):
        if "label" in task:
            self.memory.append((str(task.get("input", "")), str(task["label"])))
