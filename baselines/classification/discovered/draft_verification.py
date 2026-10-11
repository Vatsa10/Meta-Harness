"""Draft Verification (paper App. B.1; lowest-context frontier point, draft_verification.py).

Two short calls. Stage 1 (draft): retrieve the `draft_k` (5) nearest labelled examples by TF-IDF and ask for an
initial label D. Stage 2 (verification): re-query memory conditioned on D, taking `verify_k` (5) confirmers (same
label as D) and `verify_k` challengers (different label), nearest to the query first, and ask whether to keep or
revise D. Cold start: with fewer than `draft_k` labelled examples the program stops after the draft (a standard
single-call few-shot prompt). Memory grows only in `learn`.
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
    def __init__(self, draft_k=5, verify_k=5):
        self.draft_k, self.verify_k = int(draft_k), int(verify_k)
        self.memory = []

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        ranked = _rank(self.memory, query)
        near = [r.item for r in ranked[: self.draft_k]]
        draft_prompt = ("Valid labels: " + ", ".join(labels) + "\n\n" + "\n".join(_show(m) for m in near) +
                        f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        draft = _pick(model(draft_prompt), labels)
        if len(self.memory) < self.draft_k:
            return _result(draft, 1, len(near), [draft_prompt])
        support = [r.item for r in ranked if r.item["label"] == draft][: self.verify_k]
        challenge = [r.item for r in ranked if r.item["label"] != draft][: self.verify_k]
        verify_prompt = ("Valid labels: " + ", ".join(labels) + f"\n\nInitial label: {draft}\n\n"
                         "Examples with the same label:\n" + "\n".join(_show(m) for m in support) +
                         "\n\nExamples with different labels:\n" + "\n".join(_show(m) for m in challenge) +
                         f"\n\nInput: {query}\nKeep or revise the initial label. Return exactly one valid label.\nLabel:")
        final = _pick(model(verify_prompt), labels)
        return _result(final, 2, len(near) + len(support) + len(challenge), [draft_prompt, verify_prompt])

    def learn(self, task, result):
        if "label" in task:
            self.memory.append({"text": _query(task), "label": str(task["label"])})
