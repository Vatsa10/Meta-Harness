"""Agentic Context Engineering (ACE) baseline: a curated, incrementally updated bullet playbook.

Reference: Zhang et al., "Agentic Context Engineering: Evolving Contexts for Self-Improving
Language Models" (paper ref [59], arXiv:2510.04618). The Meta-Harness paper (Table 2 and the
baselines paragraph) describes ACE as using "reflective memory curation to build context over
time", with the Table 2 row marked as implemented from Ye et al. [52]. The paper gives no further
update rule, so the rule below follows the ACE method and is what this file implements.

Roles
  generator  `run`: classify the input with the playbook in the prompt; remember which bullets
             were shown.
  reflector  `learn`, on a wrong prediction only: write tagged insights from the error, one per
             line as `TAG: insight` (TAG names the error type).
  curator    `learn`: turn the insights into a small delta over the playbook, one operation per
             line, applied incrementally to the numbered bullets:
               ADD: <text>          append a new bullet (new, never reused, number; counters 0/0)
               UPDATE <n>: <text>   replace the text of bullet n only; its counters are kept
               REMOVE <n>           delete bullet n
             Lines that are not operations, and operations naming an unknown number, are ignored.
             The playbook is never rewritten wholesale.

Counters and pruning
  Every bullet carries `helpful` and `harmful` counters. After each labelled task, each bullet
  shown in the prompt gets helpful += 1 if the prediction was right, harmful += 1 if wrong.
  A bullet with harmful > helpful is pruned. When the playbook exceeds META_HARNESS_ACE_BULLETS
  (default 24) the bullets with the lowest helpful - harmful (oldest first) are dropped.

`run` never mutates state; all updates happen in `learn(task, result, model)`.
"""

import os
import re

_OP = re.compile(r"^\s*[-*]?\s*(ADD|UPDATE|REMOVE)\b\s*(\d+)?\s*:?\s*(.*?)\s*$", re.IGNORECASE)


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
    def __init__(self, max_bullets=None):
        self.max_bullets = int(max_bullets if max_bullets is not None
                               else os.environ.get("META_HARNESS_ACE_BULLETS", "24"))
        self.playbook = []   # [{"id", "text", "helpful", "harmful"}]
        self.next_id = 1
        self.memory = []
        self._shown = []
        self.insights = []

    # -- curator ---------------------------------------------------------------------------

    def _find(self, bullet_id):
        for bullet in self.playbook:
            if bullet["id"] == bullet_id:
                return bullet
        return None

    def apply_delta(self, text):
        """Apply ADD / UPDATE n / REMOVE n lines; return the list of operations applied."""
        applied = []
        for line in str(text).splitlines():
            match = _OP.match(line)
            if not match:
                continue
            op, number, body = match.group(1).upper(), match.group(2), match.group(3).strip()
            if op == "ADD":
                if body and all(b["text"] != body for b in self.playbook):
                    self.playbook.append({"id": self.next_id, "text": body, "helpful": 0, "harmful": 0})
                    applied.append(("ADD", self.next_id))
                    self.next_id += 1
            elif number is not None:
                bullet = self._find(int(number))
                if bullet is None:
                    continue
                if op == "UPDATE" and body:
                    bullet["text"] = body
                    applied.append(("UPDATE", bullet["id"]))
                elif op == "REMOVE":
                    self.playbook.remove(bullet)
                    applied.append(("REMOVE", bullet["id"]))
        return applied

    def prune(self):
        """Drop bullets with harmful > helpful, then enforce the size bound."""
        self.playbook = [b for b in self.playbook if b["harmful"] <= b["helpful"]]
        overflow = len(self.playbook) - self.max_bullets
        if overflow > 0:
            ranked = sorted(self.playbook, key=lambda b: (b["helpful"] - b["harmful"], b["id"]))
            doomed = {b["id"] for b in ranked[:overflow]}
            self.playbook = [b for b in self.playbook if b["id"] not in doomed]

    def _render(self):
        return "\n".join(f"[{b['id']}] (helpful={b['helpful']}, harmful={b['harmful']}) {b['text']}"
                         for b in self.playbook)

    # -- generator -------------------------------------------------------------------------

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        rules = self._render()
        prompt = ("Classify the input using the playbook.\nValid labels: " + ", ".join(labels) +
                  (f"\n\nPlaybook:\n{rules}" if rules else "") +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("ace_prompt", {"prompt": prompt, "bullets": len(self.playbook)})
        self._shown = [b["id"] for b in self.playbook]
        prediction = _pick(model(prompt), labels)
        trace.event("ace_result", {"prediction": prediction})
        return prediction

    # -- reflector and curator ---------------------------------------------------------------

    def learn(self, task, result, model):
        if "label" not in task:
            return
        query, truth = str(task.get("input", "")), str(task["label"])
        output = result.get("output") if hasattr(result, "get") else result
        prediction = str(output).strip()
        correct = prediction == truth
        for bullet_id in self._shown:
            bullet = self._find(bullet_id)
            if bullet is not None:
                bullet["helpful" if correct else "harmful"] += 1
        self._shown = []
        self.memory.append((query, truth))
        if not correct:
            reflection = str(model(
                "A classifier made a mistake.\n"
                f"Input: {query}\nPredicted: {prediction}\nCorrect: {truth}\n"
                "Write 1-2 tagged insights, one per line as 'TAG: insight' (TAG names the error "
                "type; insight is a reusable rule of at most 20 words that does not mention this "
                "specific input).")).strip()
            insights = [line.strip(" -*") for line in reflection.splitlines() if line.strip()][:2]
            self.insights.extend(insights)
            if insights:
                delta = str(model(
                    "You curate a numbered playbook of classification rules.\n"
                    f"Current playbook:\n{self._render() or '(empty)'}\n\n"
                    "New insights:\n" + "\n".join(f"- {i}" for i in insights) +
                    "\n\nReply with delta operations only, one per line: 'ADD: <text>', "
                    "'UPDATE <n>: <text>' or 'REMOVE <n>'. Do not rewrite the playbook."))
                self.apply_delta(delta)
        self.prune()
