"""Meta Context Engineering (MCE) baseline: an evolving library of natural-language skills.

Reference: Ye et al., "Meta Context Engineering via Agentic Skill Evolution" (paper ref [52],
arXiv:2601.21557). The Meta-Harness paper (baselines paragraph, Table 2) describes MCE as one that
"maintains and evolves a library of natural-language skills for context construction", and its
Table 2 rows for ACE and MCE are marked as implementations from Ye et al. The paper gives no
further update rule, so this file records the reference behaviour it implements:

  * the context is a bounded library of skills, each `NAME: recipe` (a short construction recipe
    saying what evidence to look for);
  * `run` shows the library, the model names the skill it used on a first line `Skill: <n>` and
    answers on `Label: <label>` (a plain label answer is accepted and credits no skill);
  * `learn(task, result, model)`: the selected skill's `used` counter goes up, and `helpful` too
    when the prediction was right (measured usefulness = (helpful + 1) / (used + 2));
  * on an error a new skill is evolved from that error (one model call, `NAME: recipe`, duplicates
    ignored);
  * retention is by measured usefulness: when the library exceeds META_HARNESS_MCE_SKILLS
    (default 12) the skills with the lowest usefulness are dropped, oldest first.

`run` never mutates state; all updates happen in `learn`.
"""

import os
import re

_SKILL = re.compile(r"skill\s*:?\s*#?\s*(\d+)", re.IGNORECASE)


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
    def __init__(self, max_skills=None):
        self.max_skills = int(max_skills if max_skills is not None
                              else os.environ.get("META_HARNESS_MCE_SKILLS", "12"))
        self.skills = []   # [{"id", "text", "used", "helpful"}]
        self.next_id = 1
        self.memory = []
        self._selected = None

    @staticmethod
    def usefulness(skill):
        return (skill["helpful"] + 1) / (skill["used"] + 2)

    def _prune(self):
        overflow = len(self.skills) - self.max_skills
        if overflow > 0:
            ranked = sorted(self.skills, key=lambda s: (self.usefulness(s), s["id"]))
            doomed = {s["id"] for s in ranked[:overflow]}
            self.skills = [s for s in self.skills if s["id"] not in doomed]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        library = "\n".join(f"{s['id']}. {s['text']}" for s in self.skills)
        prompt = ("Classify the input. Pick the most relevant skill from the library, then apply it.\n"
                  "Valid labels: " + ", ".join(labels) +
                  (f"\n\nSkill library:\n{library}" if library else "") +
                  f"\n\nInput: {query}\nReply with 'Skill: <number>' then 'Label: <label>'.\nSkill:")
        trace.event("mce_prompt", {"prompt": prompt, "skills": len(self.skills)})
        raw = str(model(prompt))
        chosen = _SKILL.search("Skill: " + raw) if raw.strip()[:1].isdigit() else _SKILL.search(raw)
        self._selected = int(chosen.group(1)) if chosen else None
        answer = raw.rsplit("Label:", 1)[-1] if "Label:" in raw else raw
        prediction = _pick(answer, labels)
        trace.event("mce_result", {"prediction": prediction, "skill": self._selected})
        return prediction

    def learn(self, task, result, model):
        if "label" not in task:
            return
        query, truth = str(task.get("input", "")), str(task["label"])
        output = result.get("output") if hasattr(result, "get") else result
        prediction = str(output).strip()
        correct = prediction == truth
        for skill in self.skills:
            if skill["id"] == self._selected:
                skill["used"] += 1
                skill["helpful"] += int(correct)
        self._selected = None
        self.memory.append((query, truth))
        if not correct:
            lines = str(model(
                "You maintain a library of classification skills. A prediction was wrong.\n"
                f"Input: {query}\nPredicted: {prediction}\nCorrect: {truth}\n"
                "Write one skill as 'NAME: recipe' where the recipe is at most 25 words and "
                "describes what evidence to look for.\nSkill:")).strip().splitlines()
            text = lines[0].strip(" -*") if lines else ""
            if text and all(s["text"] != text for s in self.skills):
                self.skills.append({"id": self.next_id, "text": text, "used": 0, "helpful": 0})
                self.next_id += 1
        self._prune()
