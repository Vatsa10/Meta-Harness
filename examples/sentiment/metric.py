"""Custom metric for the sentiment example: `score(task, result) -> float` in [0, 1].

Use it with `--metric examples/sentiment/metric.py:score`.
"""


def score(task, result):
    predicted = str(result.get("output", "")).strip().lower()
    return float(predicted == str(task["label"]).strip().lower())
