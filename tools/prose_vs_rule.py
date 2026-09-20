"""Is the harness the words, or the mechanism?

Runs one doctrine two ways against the same tasks: as prose in the system prompt, and as
tool.check rules with the prose removed. The paper asks whether scaffolding beats the model;
this asks whether the scaffolding has to be read to work.

The honest-reporting bar here is high on purpose: the whole value of this experiment is that
it is capable of reporting a result AGAINST the project's own thesis (prose wins, or the two
arms are indistinguishable). `summarize` therefore never declares an accuracy winner unless
the sample size and the effect size both clear an explicit, symmetric bar. Neither arm gets
special treatment: the same significance test runs regardless of which arm scored higher.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

# Below this many observations per arm, an accuracy difference is not evidence of anything -
# it is noise. This is a floor, not a tuned constant: raise it if anything, never lower it to
# make a particular run look more conclusive.
MIN_SAMPLE_PER_ARM = 5

# Two-sided 95% threshold on a standard normal z-score.
Z_SIGNIFICANCE = 1.96


def build_arms(doctrine_path: Path | str, tasks_path: Path | str) -> dict[str, dict[str, Any]]:
    """Two arms differing only in where the doctrine lives."""
    doctrine = Path(doctrine_path).read_text(encoding="utf-8").strip()
    tasks = str(tasks_path)
    return {
        "prose": {"append_system_prompt": doctrine, "rules_enabled": False, "tasks": tasks},
        "rule": {"append_system_prompt": "", "rules_enabled": True, "tasks": tasks},
    }


def _accuracy_verdict(prose: Mapping[str, Any], rule: Mapping[str, Any],
                       score_delta: float) -> str | None:
    """Decide whether an accuracy difference is trustworthy, symmetrically for both arms.

    Returns None when the scores are exactly tied (nothing to report). Otherwise returns a
    sentence: either an honest "no detectable difference" (sample too small, or the gap is
    within noise for the sample given) or a winner statement backed by a two-proportion
    z-test, computed the same way regardless of which arm is ahead.
    """
    if score_delta == 0:
        return None

    prose_score = float(prose.get("score", 0.0))
    rule_score = float(rule.get("score", 0.0))
    prose_n = int(prose.get("n", 1))
    rule_n = int(rule.get("n", 1))

    if prose_n < MIN_SAMPLE_PER_ARM or rule_n < MIN_SAMPLE_PER_ARM:
        return (f"no detectable difference in accuracy: sample too small to trust "
                f"(prose n={prose_n}, rule n={rule_n}, need >= {MIN_SAMPLE_PER_ARM} each)")

    # Two-proportion z-test using a pooled standard error, applied identically to both arms.
    pooled_p = ((prose_score * prose_n) + (rule_score * rule_n)) / (prose_n + rule_n)
    pooled_se = math.sqrt(pooled_p * (1 - pooled_p) * (1 / prose_n + 1 / rule_n)) \
        if 0 < pooled_p < 1 else 0.0

    if pooled_se == 0.0:
        z = float("inf") if score_delta != 0 else 0.0
    else:
        z = score_delta / pooled_se

    if abs(z) < Z_SIGNIFICANCE:
        return (f"no detectable difference in accuracy: gap of {score_delta:+.3f} is not "
                f"statistically significant at n={prose_n}/{rule_n} (z={z:.2f})")

    winner = "rule" if score_delta > 0 else "prose"
    return (f"score moved {score_delta:+.3f} (prose {prose_score} -> rule {rule_score}); "
            f"{winner} wins, statistically significant at n={prose_n}/{rule_n} (z={z:.2f})")


def summarize(results: Mapping[str, Mapping[str, float]]) -> str:
    prose, rule = results.get("prose", {}), results.get("rule", {})
    score_delta = float(rule.get("score", 0)) - float(prose.get("score", 0))
    context_delta = float(rule.get("context", 0)) - float(prose.get("context", 0))
    if score_delta == 0 and context_delta == 0:
        return "no difference between prose and mechanism on this task set"

    lines = []
    accuracy_line = _accuracy_verdict(prose, rule, score_delta)
    if accuracy_line:
        lines.append(accuracy_line)
    if context_delta:
        share = abs(context_delta) / max(1.0, float(prose.get("context", 1)))
        lines.append(f"context moved {context_delta:+.0f} tokens ({share:.0%})")
    if not score_delta and context_delta:
        lines.append("accuracy unchanged: this is a context result, not an accuracy one")
    return "; ".join(lines)


if __name__ == "__main__":
    import sys

    arms = build_arms(sys.argv[1], sys.argv[2])
    print(json.dumps(arms, indent=2))
