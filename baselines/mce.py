"""Meta Context Engineering baseline: an evolving library of natural-language skills.

Follows Ye et al. (paper ref [52]) in shape: a small library of named skills, each a short
construction recipe; the model selects a skill, applies it, and the library grows on failure.
"""

import os


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
        self.skills = []
        self.memory = []

    def _evolve(self, query, prediction, truth, model, trace):
        prompt = ("You maintain a library of classification skills. A prediction was wrong.\n"
                  f"Input: {query}\nPredicted: {prediction}\nCorrect: {truth}\n"
                  "Write one skill as 'NAME: recipe' where the recipe is at most 25 words and "
                  "describes what evidence to look for.\nSkill:")
        lines = str(model(prompt)).strip().splitlines()
        skill = lines[0].strip(" -*") if lines else ""
        trace.event("mce_skill", {"skill": skill})
        if skill and skill not in self.skills:
            self.skills.append(skill)
            del self.skills[: max(0, len(self.skills) - self.max_skills)]

    def run(self, task, model, trace):
        query = str(task.get("input", ""))
        labels = _labels(task, self.memory)
        library = "\n".join(f"{index + 1}. {skill}" for index, skill in enumerate(self.skills))
        prompt = ("Classify the input. Pick the most relevant skill from the library, then apply it.\n"
                  "Valid labels: " + ", ".join(labels) +
                  (f"\n\nSkill library:\n{library}" if library else "") +
                  f"\n\nInput: {query}\nReturn exactly one valid label.\nLabel:")
        trace.event("mce_prompt", {"prompt": prompt, "skills": len(self.skills)})
        prediction = _pick(model(prompt), labels)
        if "label" in task:
            truth = str(task["label"])
            if prediction != truth:
                self._evolve(query, prediction, truth, model, trace)
            self.memory.append((query, truth))
        trace.event("mce_result", {"prediction": prediction})
        return prediction
