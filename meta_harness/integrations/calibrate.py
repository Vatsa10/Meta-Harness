"""Calibrate the blind judge to the owner's good/bad labels, then grade with the winner.

`calibrate(labels, judge_fn, prompts)` scores every candidate judge prompt twice - as given and with
few-shot examples drawn from the owner's reasons (leave-one-out: a labelled run never sees itself as
an example) - against the labels, reports agreement, Cohen's kappa and confusion counts, keeps the
best variant and saves it to `judge.json`. `calibrated_judge(task, result)` is a search metric
(`--metric meta_harness.integrations.calibrate:calibrated_judge`) that grades with that prompt using
the judge model from `meta_harness.judge`. Stdlib only.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .. import judge
from ..protocol import expected_value

JUDGE_FILE_ENV = "META_HARNESS_JUDGE_FILE"
DEFAULT_FILE = "judge.json"
MIN_AGREEMENT = 0.8
MIN_LABELS = 20
SHOTS = 2
EXAMPLE_CHARS = 500
FEW_SHOT_HEADER = "The owner graded these earlier answers; grade the same way.\n"


def load_labels(source: str | Path | Iterable[Mapping]) -> list[dict]:
    if isinstance(source, (str, Path)):
        rows = [json.loads(line) for line in Path(source).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = [dict(row) for row in source]
    return [row for row in rows if row.get("label") in ("good", "bad")]


def pick_examples(labels: list[dict], shots: int = SHOTS) -> list[dict]:
    """Up to `shots` bad and `shots` good labelled runs that carry a reason, in label order."""
    chosen: list[dict] = []
    for kind in ("bad", "good"):
        chosen += [row for row in labels if row.get("label") == kind and str(row.get("reason", "")).strip()][:shots]
    return [{"id": row.get("id"), "output": _clip(row.get("output", "")), "label": row["label"],
             "reason": str(row["reason"]).strip()} for row in chosen]


def _clip(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= EXAMPLE_CHARS else text[:EXAMPLE_CHARS] + "..."


def examples_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    parts = [FEW_SHOT_HEADER]
    for number, example in enumerate(examples, start=1):
        verdict = "PASS" if example["label"] == "good" else "FAIL"
        parts.append(f"Example {number}:\nAnswer:\n{example['output']}\nOwner: {verdict} - {example['reason']}\n")
    return "\n".join(parts) + "\n"


def render(template: str, examples: list[dict], task: Mapping, output: Any) -> str:
    """Blind prompt: few-shot block + template over the criteria and the answer only."""
    criteria = task.get("criteria")
    if criteria in (None, ""):
        criteria = expected_value(task)
    answer = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
    body = template.replace("{criteria}", "" if criteria is None else str(criteria)).replace("{answer}", answer)
    return examples_block(examples) + body


def score(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs = (owner_good, judge_pass). Agreement, Cohen's kappa and the confusion counts."""
    tp = sum(1 for owner, judged in pairs if owner and judged)
    fn = sum(1 for owner, judged in pairs if owner and not judged)
    fp = sum(1 for owner, judged in pairs if not owner and judged)
    tn = sum(1 for owner, judged in pairs if not owner and not judged)
    n = len(pairs)
    if not n:
        return {"n": 0, "agreement": 0.0, "kappa": 0.0, "confusion": {"tp": 0, "fn": 0, "fp": 0, "tn": 0}}
    observed = (tp + tn) / n
    expected = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else (1.0 if observed == 1 else 0.0)
    return {"n": n, "agreement": observed, "kappa": kappa, "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn}}


def _evaluate(labels: list[dict], judge_fn: Callable[[str], str], template: str, examples: list[dict]) -> dict:
    pairs = []
    for row in labels:
        own = [example for example in examples if example["id"] != row.get("id")]
        prompt = render(template, own, row, row.get("output", ""))
        pairs.append((row["label"] == "good", judge.parse_verdict(judge_fn(prompt)) >= 0.5))
    return score(pairs)


def calibrate(labels: str | Path | Iterable[Mapping], judge_fn: Callable[[str], str],
              prompts: list[str] | None = None, *, shots: int = SHOTS,
              out: str | Path | None = DEFAULT_FILE) -> dict:
    """Score each prompt (base and few-shot) against the labels; save and return the best."""
    rows = load_labels(labels)
    if not rows:
        raise ValueError("no good/bad labels to calibrate against")
    prompts = list(prompts or [judge.JUDGE_PROMPT])
    for template in prompts:
        if "{answer}" not in template:
            raise ValueError("every judge prompt needs an {answer} placeholder")
    examples = pick_examples(rows, shots)
    candidates = []
    for index, template in enumerate(prompts):
        variants = [("base", [])] + ([("few-shot", examples)] if examples else [])
        for variant, shot in variants:
            result = _evaluate(rows, judge_fn, template, shot)
            candidates.append({"prompt_index": index, "variant": variant, "prompt": template,
                               "few_shot": shot, **result})
    best = max(enumerate(candidates), key=lambda pair: (pair[1]["agreement"], pair[1]["kappa"], -pair[0]))[1]
    notes = []
    if best["n"] < MIN_LABELS:
        notes.append(f"only {best['n']} labelled runs; label at least {MIN_LABELS} for a trustworthy judge")
    if best["agreement"] < MIN_AGREEMENT:
        notes.append(f"best agreement {best['agreement']:.2f} is below {MIN_AGREEMENT}; "
                     "the judge does not yet grade like you - label more runs or add prompt variants")
    for note in notes:
        warnings.warn(note, stacklevel=2)
    report = {key: best[key] for key in ("prompt", "few_shot", "variant", "prompt_index", "n", "agreement",
                                          "kappa", "confusion")}
    report["warnings"] = notes
    report["candidates"] = [{key: c[key] for key in ("prompt_index", "variant", "n", "agreement", "kappa",
                                                     "confusion")} for c in candidates]
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["path"] = str(path)
    return report


_CACHE: dict[str, tuple[float, dict]] = {}


def load_judge(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get(JUDGE_FILE_ENV) or DEFAULT_FILE)
    try:
        stamp = path.stat().st_mtime
    except OSError:
        raise FileNotFoundError(f"no calibrated judge at {path}; run calibrate first "
                                f"or set {JUDGE_FILE_ENV}") from None
    cached = _CACHE.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    config = json.loads(path.read_text(encoding="utf-8"))
    if "prompt" not in config:
        raise ValueError(f"{path} has no prompt")
    _CACHE[str(path)] = (stamp, config)
    return config


def calibrated_judge(task: Mapping, result: Mapping) -> float:
    """Search metric: grade result['output'] with the calibrated prompt (judge.json)."""
    config = load_judge()
    prompt = render(config["prompt"], config.get("few_shot") or [], task, result.get("output", ""))
    model = judge.judge_model()
    for attempt in range(judge.JUDGE_RETRIES + 1):
        try:
            return judge.parse_verdict(model(prompt))
        except Exception as exc:
            if not judge.retryable(judge.http_status(exc)) or attempt == judge.JUDGE_RETRIES:
                raise
            time.sleep(judge.RETRY_DELAY)
    return 0.0
