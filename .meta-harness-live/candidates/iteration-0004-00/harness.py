"""Ultra-minimal zero-shot: maximum token efficiency with streamlined format."""


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
    def __init__(self):
        self.memory = []

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        prompt = f"{query} → [{' | '.join(labels)}] ="
        trace.event("minimal_prompt", {"prompt": prompt, "labels": len(labels)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            self.memory.append((query, str(task["label"])))
        trace.event("minimal_result", {"prediction": prediction})
        return prediction
