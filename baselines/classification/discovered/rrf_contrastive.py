"""RRF + Contrastive (paper Table 9, Pareto variant "RRF + Contrastive"; behaviour per App. B.1).

Hybrid retrieval. The memory is ranked twice, by TF-IDF cosine and by BM25, and the two rankings are merged with
reciprocal rank fusion (`meta_harness.retrieval.reciprocal_rank_fusion`). The prompt shows the top `k` fused
examples, then up to `pair_count` contrastive pairs: each of the top fused examples paired with the best-fused
example carrying a different label. Memory grows only in `learn`.

Design inferred from the name in Table 9; App. B.1 specifies only draft_verification and label_primed_query. Chosen constants: k=6 fused examples, pair_count=3, RRF constant 60, BM25 with default k1=1.5 b=0.75.
"""

from meta_harness.retrieval import BM25Index, TfidfIndex, reciprocal_rank_fusion


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


def _show(m):
    return f"Example: {m['text']}\nLabel: {m['label']}"


def _result(label, calls, examples, prompts):
    return {"output": label, "calls": calls, "examples": examples,
            "prompt_chars": sum(len(p) for p in prompts)}


class Harness:
    def __init__(self, k=6, pair_count=3):
        self.k, self.pair_count = int(k), int(pair_count)
        self.memory = []

    def run(self, task, model):
        query = _query(task)
        labels = _labels(task, self.memory)
        fused = []
        if self.memory:
            texts = [m["text"] for m in self.memory]
            dense = TfidfIndex(self.memory, texts).search(query, len(texts))
            sparse = BM25Index(self.memory, texts, math_mode=False).search(query, len(texts))
            fused = reciprocal_rank_fusion([dense, sparse])
        top = [r.item for r in fused[: self.k]]
        pairs = []
        for r in fused[: self.pair_count]:
            other = next((x.item for x in fused if x.item["label"] != r.item["label"]), None)
            if other is not None and (r.item, other) not in pairs and (other, r.item) not in pairs:
                pairs.append((r.item, other))
        shown, blocks = list(top), []
        for a, b in pairs:
            fresh = [m for m in (a, b) if m not in shown]
            if fresh:
                shown.extend(fresh)
                blocks.append("\n".join(_show(m) for m in fresh) + "\n--")
        prompt = ("Valid labels: " + ", ".join(labels) + "\n\nTop fused examples:\n" +
                  "\n".join(_show(m) for m in top) + "\n\nContrastive pairs:\n" +
                  "\n".join(blocks) +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        label = _pick(model(prompt), labels)
        return _result(label, 1, len(shown), [prompt])

    def learn(self, task, result):
        if "label" in task:
            self.memory.append({"text": _query(task), "label": str(task["label"])})
