"""IMO-level test set: 200 problems (paper section 4.2, Table 11).

    IMO-AnswerBench     100  (stratified by Category from the full set; seeded)
    IMO-ProofBench       60  (proof-only: no short answer, so not gradable by a string metric)
    ArXivMath Dec. 2025  17
    ArXivMath Jan. 2026  23

Sources (pinned from the benchmark repos/cards, not re-verified offline):
    IMO-AnswerBench / ProofBench: CSV in github.com/google-deepmind/superhuman (imobench/)
        answerbench.csv columns: Problem ID, Problem, Short Answer, Category
        proofbench.csv columns: Problem ID, Problem, ...
    ArXivMath: HF MathArena/arxivmath-2512 and MathArena/arxivmath-2601, split train:
        problem, answer
Answer-gradable items go to test.jsonl as {"problem", "answer"}. Proof-only items (every
ProofBench row, and any row without an answer) are listed in skipped.json: proof grading needs a
judge, which a string metric cannot replace (README says so).
"""

from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

from ..httpdata import hf_rows, urllib_get, write_jsonl

NAME, DOMAIN = "imo", "math"
RAW = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/"
ANSWERBENCH_URL = RAW + "answerbench.csv"
PROOFBENCH_URL = RAW + "proofbench.csv"
ARXIVMATH = {"ArXivMath Dec. 2025": "MathArena/arxivmath-2512",
             "ArXivMath Jan. 2026": "MathArena/arxivmath-2601"}
ANSWERBENCH_SIZE = 100


def _csv(http, url: str) -> list[dict]:
    text = http(url).decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(io.StringIO(text, newline=""))]


def stratified_sample(rows: list[dict], key: str, n: int, seed: int) -> list[dict]:
    """Seeded sample of n rows, proportional per `key` group (largest remainder)."""
    if n >= len(rows):
        return list(rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "")), []).append(row)
    rng = random.Random(seed)
    names = sorted(groups)
    quota = {g: len(groups[g]) * n / len(rows) for g in names}
    take = {g: int(quota[g]) for g in names}
    for g in sorted(names, key=lambda g: (-(quota[g] - take[g]), g))[: n - sum(take.values())]:
        take[g] += 1
    picked: list[dict] = []
    for g in names:
        members = list(groups[g])
        rng.shuffle(members)
        picked.extend(members[: take[g]])
    return picked


def fetch(cache_dir, http=urllib_get, seed: int = 0) -> dict[str, Path]:
    cache = Path(cache_dir)
    test, skipped = cache / "test.jsonl", cache / "skipped.json"
    if test.exists() and skipped.exists():
        return {"test": test}
    items: list[dict] = []
    skips: list[dict] = []

    def take(source: str, ident: str, problem: str, answer: str) -> None:
        problem, answer = (problem or "").strip(), (answer or "").strip()
        if problem and answer:
            items.append({"problem": problem, "answer": answer})
        elif problem:
            skips.append({"source": source, "id": ident, "reason": "proof-only: no short answer"})

    answerbench = _csv(http, ANSWERBENCH_URL)
    for row in stratified_sample(answerbench, "Category", ANSWERBENCH_SIZE, seed):
        take("IMO-AnswerBench", row.get("Problem ID", ""), row.get("Problem", ""),
             row.get("Short Answer", ""))
    for row in _csv(http, PROOFBENCH_URL):  # ProofBench has no short answers: all proof-only
        take("IMO-ProofBench", row.get("Problem ID", ""), row.get("Problem", ""),
             row.get("Short Answer", ""))
    for label, dataset in ARXIVMATH.items():
        for position, row in enumerate(hf_rows(dataset, "default", "train", http)):
            take(label, str(row.get("problem_idx", position)), row.get("problem", ""),
                 str(row.get("answer") or ""))
    write_jsonl(test, items)
    skipped.write_text(json.dumps(skips, indent=2), encoding="utf-8")
    return {"test": test}
