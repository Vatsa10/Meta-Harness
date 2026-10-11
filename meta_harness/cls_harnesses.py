"""Classification harnesses from the paper appendices (text classification, section 4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .retrieval import TfidfIndex


def _call(model: Callable[..., Any], prompt: str, **kwargs: Any) -> str:
    result = model(prompt, **kwargs)
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, Mapping):
        for key in ("label", "answer", "prediction", "content", "text"):
            if key in result:
                return str(result[key]).strip()
    return str(result).strip()


def _label(output: str, labels: Sequence[str]) -> str:
    for label in labels:
        if label.lower() in output.lower():
            return label
    return output.splitlines()[0].strip() if output else (labels[0] if labels else output)


@dataclass
class MemoryExample:
    text: str
    label: str


class LabelPrimedQueryHarness:
    """Label primer, per-class coverage, and query-local contrastive examples."""

    def __init__(self, labels: Sequence[str] | None = None, pair_count: int = 4):
        self.labels = list(labels or [])
        self.pair_count = pair_count
        self.memory: list[MemoryExample] = []

    def _context(self, query: str) -> list[MemoryExample]:
        if not self.memory:
            return []
        retrieved = TfidfIndex(self.memory, [x.text for x in self.memory]).search(query, max(8, len(self.labels)))
        selected: list[MemoryExample] = []
        for label in self.labels:
            match = next((x.item for x in retrieved if x.item.label == label), None)
            if match and match not in selected:
                selected.append(match)
        for result in retrieved:
            if result.item not in selected:
                selected.append(result.item)
        pairs: list[MemoryExample] = []
        for left in selected:
            right = next((x for x in selected if x.label != left.label), None)
            if right:
                pairs.extend([left, right])
            if len(pairs) >= self.pair_count * 2:
                break
        return selected + [x for x in pairs if x not in selected]

    def run(self, task: Mapping[str, Any], model: Callable[..., Any], trace: Any) -> str:
        query = str(task.get("input", task.get("text", "")))
        if task.get("labels"):
            self.labels = [str(x) for x in task["labels"]]
        if not self.labels:
            self.labels = sorted({x.label for x in self.memory} | ({str(task["label"])} if "label" in task else set()))
        examples = self._context(query)
        prompt = "Valid labels: " + ", ".join(self.labels) + "\n\n"
        prompt += "\n".join(f"Example: {x.text}\nLabel: {x.label}" for x in examples)
        prompt += f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:"
        trace.event("classification_prompt", {"prompt": prompt, "retrieved": len(examples)})
        trace.add_context_cost(len(prompt))
        prediction = _label(_call(model, prompt), self.labels)
        if "label" in task:
            self.memory.append(MemoryExample(query, str(task["label"])))
        trace.event("classification_result", {"prediction": prediction})
        return prediction


class DraftVerificationHarness:
    """Draft prediction followed by confirmer/challenger verification."""

    def __init__(self, labels: Sequence[str] | None = None, draft_k: int = 5, verify_k: int = 5):
        self.labels = list(labels or [])
        self.draft_k, self.verify_k = draft_k, verify_k
        self.memory: list[MemoryExample] = []

    def run(self, task: Mapping[str, Any], model: Callable[..., Any], trace: Any) -> str:
        query = str(task.get("input", ""))
        if task.get("labels"):
            self.labels = [str(x) for x in task["labels"]]
        if not self.labels:
            self.labels = sorted({x.label for x in self.memory} | ({str(task["label"])} if "label" in task else set()))
        nearest = TfidfIndex(self.memory, [x.text for x in self.memory]).search(query, self.draft_k) if self.memory else []
        context = "\n".join(f"{x.item.text} => {x.item.label}" for x in nearest)
        draft_prompt = f"Labels: {', '.join(self.labels)}\nExamples:\n{context}\nInput: {query}\nGive one label."
        trace.event("draft_prompt", {"prompt": draft_prompt})
        trace.add_context_cost(len(draft_prompt))
        draft = _label(_call(model, draft_prompt), self.labels)
        final = draft
        if len(self.memory) >= self.draft_k:
            supporters = [x for x in self.memory if x.label == draft]
            challengers = [x for x in self.memory if x.label != draft]
            support = TfidfIndex(supporters, [x.text for x in supporters]).search(query, self.verify_k) if supporters else []
            challenge = TfidfIndex(challengers, [x.text for x in challengers]).search(query, self.verify_k) if challengers else []
            verification = "\n".join(f"Support: {x.item.text} => {x.item.label}" for x in support)
            verification += "\n" + "\n".join(f"Challenge: {x.item.text} => {x.item.label}" for x in challenge)
            verify_prompt = f"Initial label: {draft}\n{verification}\nInput: {query}\nKeep or revise; return one label."
            trace.event("verification_prompt", {"prompt": verify_prompt})
            trace.add_context_cost(len(verify_prompt))
            final = _label(_call(model, verify_prompt), self.labels)
        if "label" in task:
            self.memory.append(MemoryExample(query, str(task["label"])))
        trace.event("classification_result", {"draft": draft, "prediction": final})
        return final


__all__ = ["MemoryExample", "LabelPrimedQueryHarness", "DraftVerificationHarness", "_call", "_label"]
