"""Adaptive few-shot: start zero-shot, add retrieved examples only after mistakes."""

from meta_harness.retrieval import TfidfIndex


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
        self.index = None
        self.errors = set()

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        
        # Use retrieval only if we've made errors and have memory
        examples = []
        if self.errors and self.memory:
            if self.index is None:
                self.index = TfidfIndex()
                for text, label in self.memory:
                    self.index.add(text, (text, label))
            results = self.index.search(query, k=min(3, len(self.memory)))
            examples = [doc for _, doc in results]
        
        if examples:
            block = "\n".join(f"Example: {text}\nLabel: {label}" for text, label in examples)
            prompt = ("Classify the input.\nValid labels: " + ", ".join(labels) + 
                     f"\n\n{block}\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        else:
            prompt = ("Classify the input.\nValid labels: " + ", ".join(labels) +
                     f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        
        trace.event("adaptive_prompt", {"examples": len(examples), "has_errors": bool(self.errors)})
        prediction = _pick(model(prompt), labels)
        
        if "label" in task:
            truth = str(task["label"])
            if prediction != truth:
                self.errors.add(len(self.memory))
            self.memory.append((query, truth))
            # Rebuild index when memory grows significantly
            if self.index and len(self.memory) % 10 == 0:
                self.index = None
        
        trace.event("adaptive_result", {"prediction": prediction})
        return prediction
