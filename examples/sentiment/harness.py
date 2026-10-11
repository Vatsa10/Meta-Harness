"""Seed harness for the offline sentiment example.

Contract: a `Harness` class whose `run(task, model, trace)` returns {"output": ...}.
`model(prompt)` returns the reply text; tokens are metered by the engine, not by the harness.
This seed passes the raw reply through, so it scores poorly when the model is chatty.
"""


class Harness:
    def run(self, task, model, trace):
        reply = model("What is the sentiment of this review? Review: " + task["input"])
        return {"output": reply.strip()}
