"""Dense few-shot baseline (paper Table 6, "Dense Retrieval", applied to classification).

The top-k memory examples by cosine similarity over embeddings from meta_harness.embeddings.embed.
Provider and model come from META_HARNESS_EMBED_PROVIDER / META_HARNESS_EMBED_MODEL (default
openai / text-embedding-3-small). Embeddings are cached in memory per text. The query is embedded on
every run, so a missing embedding provider raises RuntimeError at the first call (the seed fails
validation visibly); there is no TF-IDF fallback. k comes from META_HARNESS_FEW_SHOT_N (default 8).
"""

import os

from meta_harness.embeddings import cosine, embed

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
    def __init__(self, shots=None, embed_fn=None):
        raw = shots if shots is not None else os.environ.get("META_HARNESS_FEW_SHOT_N", "8")
        # an explicit integer (tests, programmatic use) bypasses the env value set
        self.shots = shots if isinstance(shots, int) and shots > 0 else parse_n(raw)
        self.embed_fn = embed_fn
        self.memory = []
        self.cache = {}

    def _embed(self, texts):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            provider = os.environ.get("META_HARNESS_EMBED_PROVIDER", "openai")
            model = os.environ.get("META_HARNESS_EMBED_MODEL", "text-embedding-3-small")
            fn = self.embed_fn or embed
            try:
                vectors = fn(missing, provider=provider, model=model)
            except Exception as error:  # noqa: BLE001
                # Only a missing provider/key is a configuration problem; anything else keeps its text.
                if not (isinstance(error, ValueError) or "is not set" in str(error)):
                    raise
                raise RuntimeError(
                    f"dense few-shot needs an embedding provider ({provider}/{model}; set "
                    f"META_HARNESS_EMBED_PROVIDER, META_HARNESS_EMBED_MODEL and the provider API key): "
                    f"{type(error).__name__}: {error}") from error
            for text, vector in zip(missing, vectors):
                self.cache[text] = vector
        return [self.cache[t] for t in texts]

    def _examples(self, query):
        vectors = self._embed([query] + [text for text, _ in self.memory])
        sims = [cosine(vectors[0], vec) for vec in vectors[1:]]
        # Highest similarity first; ties go to the more recent example.
        order = sorted(range(len(sims)), key=lambda i: (-sims[i], -i))
        keep = len(order) if self.shots is None else self.shots
        return [self.memory[i] for i in order[:keep]]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        examples = self._examples(query)
        block = "\n".join(f"Example: {text}\nLabel: {label}" for text, label in examples)
        prompt = ("Classify the input.\nValid labels: " + ", ".join(labels) + "\n\n" + block +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("dense_few_shot_prompt", {"prompt": prompt, "shots": len(examples)})
        prediction = _pick(model(prompt), labels)
        trace.event("dense_few_shot_result", {"prediction": prediction})
        return prediction

    def learn(self, task, result):
        if "label" in task:
            self.memory.append((str(task.get("input", "")), str(task["label"])))
