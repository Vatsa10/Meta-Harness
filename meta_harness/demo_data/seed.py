"""Smoke seed harness: ask the model directly."""


class Harness:
    def run(self, task, model, trace):
        return {"output": model("classify: " + task["input"])}
