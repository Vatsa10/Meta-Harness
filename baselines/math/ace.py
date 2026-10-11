"""Agentic Context Engineering baseline for math: a reflectively curated playbook of strategy bullets.

Solve with the playbook in context; when the final answer is wrong (the task has an `answer`),
ask the model for one durable, problem-independent strategy bullet. Bounded and deduplicated.
"""

import os
import re

from meta_harness.metrics import math_equivalence


def _boxed(text):
    found = re.findall(r"\\boxed\{([^{}]*)\}", str(text))
    if found:
        return found[-1].strip()
    lines = str(text).strip().splitlines()
    return lines[-1].strip() if lines else ""


class Harness:
    def __init__(self, max_bullets=None):
        self.max_bullets = int(max_bullets if max_bullets is not None
                               else os.environ.get("META_HARNESS_ACE_BULLETS", "24"))
        self.playbook = []

    def _reflect(self, problem, got, truth, model, trace):
        prompt = ("A math solution reached a wrong final answer.\n"
                  f"Problem: {problem}\nGiven answer: {got}\nCorrect answer: {truth}\n"
                  "Write ONE short reusable proof or computation strategy rule (max 25 words) that "
                  "would avoid this mistake. Do not mention this specific problem.\nRule:")
        lines = str(model(prompt)).strip().splitlines()
        bullet = lines[0].strip(" -*") if lines else ""
        trace.event("ace_reflection", {"bullet": bullet})
        if bullet and bullet not in self.playbook:
            self.playbook.append(bullet)
            del self.playbook[: max(0, len(self.playbook) - self.max_bullets)]

    def run(self, task, model, trace):
        problem = str(task.get("problem", task.get("input", "")))
        rules = "\n".join(f"- {bullet}" for bullet in self.playbook)
        prompt = ("Solve the problem rigorously. End with the final answer inside \\boxed{}.\n\n" +
                  (f"Playbook of strategies:\n{rules}\n\n" if rules else "") +
                  f"Problem:\n{problem}\n\nSolution:")
        trace.event("ace_prompt", {"bullets": len(self.playbook)})
        answer = str(model(prompt)).strip()
        if "answer" in task and not math_equivalence(_boxed(answer), task):
            self._reflect(problem, _boxed(answer), task["answer"], model, trace)
        trace.event("ace_result", {"answer": answer})
        return answer
