"""Agentic Context Engineering baseline: a reflectively curated bullet playbook.

Follows Zhang et al. (paper ref [59]) in shape: predict, then on an error ask the model to
write one durable bullet, keeping a bounded deduplicated playbook.
"""

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
    def __init__(self, max_bullets=None):
        self.max_bullets = int(max_bullets if max_bullets is not None
                               else os.environ.get("META_HARNESS_ACE_BULLETS", "24"))
        self.playbook = []
        self.memory = []

    def _reflect(self, query, prediction, truth, model, trace):
        prompt = ("A classifier made a mistake.\n"
                  f"Input: {query}\nPredicted: {prediction}\nCorrect: {truth}\n"
                  "Write ONE short reusable rule (max 20 words) that would prevent this mistake. "
                  "Do not mention this specific input.\nRule:")
        lines = str(model(prompt)).strip().splitlines()
        bullet = lines[0].strip(" -*") if lines else ""
        trace.event("ace_reflection", {"bullet": bullet})
        if bullet and bullet not in self.playbook:
            self.playbook.append(bullet)
            del self.playbook[: max(0, len(self.playbook) - self.max_bullets)]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        rules = "\n".join(f"- {bullet}" for bullet in self.playbook)
        prompt = ("Classify the input using the playbook.\nValid labels: " + ", ".join(labels) +
                  (f"\n\nPlaybook:\n{rules}" if rules else "") +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("ace_prompt", {"prompt": prompt, "bullets": len(self.playbook)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            truth = str(task["label"])
            if prediction != truth:
                self._reflect(query, prediction, truth, model, trace)
            self.memory.append((query, truth))
        trace.event("ace_result", {"prediction": prediction})
        return prediction
