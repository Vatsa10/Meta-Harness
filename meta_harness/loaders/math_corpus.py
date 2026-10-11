"""Math retrieval corpus (paper appendix C.2, Table 10): eight open datasets, merged.

Table 10 (problems in the final corpus; 535,356 in total):
    OpenMathReasoning 281,743 | DeepMath-103K 103,021 | NuminaMath-1.5 129,520 | PolyMath 11,083
    Omni-MATH 933 | FineProofs-SFT 4,289 | AIME 1983-2024 4,275 | Putnam-AXIOM 492
The table text in paper.pdf extracts out of column order; the Omni-MATH (933) and Putnam (492)
counts are recovered by making the column sum to the stated total. Omni-MATH: the paper's 933 is a
subset whose filter is not stated, so this corpus keeps every row (about 4,428) and the count
differs from Table 10; the search-set items are removed by decontamination either way.

SOURCES maps each dataset to its Hugging Face id. The ids and field names below are pinned from
the dataset cards. Putnam-AXIOM was verified against the live hub on 2026-10-11; the others were
not, and a wrong pin fails loudly at fetch time (HTTP error), not with an empty source.
Row field names read per source (first present wins):
    OpenMathReasoning  nvidia/OpenMathReasoning  split cot: problem, generated_solution,
                       pass_rate_72b_tir, problem_source
    DeepMath-103K      zwhe99/DeepMath-103K      train: question, r1_solution_1, difficulty
    NuminaMath-1.5     AI-MO/NuminaMath-1.5      train: problem, solution, source (no difficulty)
    PolyMath           Qwen/PolyMath             config en, splits low/medium/high/top: question, answer
    Omni-MATH          KbsdJames/Omni-MATH       test: problem, solution, difficulty
    FineProofs-SFT     lm-provers/FineProofs-SFT train: problem, solution
    AIME 1983-2024     di-zhang-fdu/AIME_1983_2024 train: Question, Answer
    Putnam-AXIOM       kaifronsdal/Putnam-AXIOM  config original, split train (501 rows): problem, solution

Filters from the paper: NuminaMath keeps competition sources only; OpenMathReasoning drops
problems whose source is an evaluation family (IMO, AIME, HMMT, SMT, USAMO, Putnam) and keeps one
solution per problem (the highest pass rate); OpenMathReasoning and DeepMath solutions are cut to
5,000 characters. Output rows: {"id", "problem", "solution", "source", "difficulty"}.
Pass `per_source_limit` to cap every source for a small run; None fetches everything.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..httpdata import hf_rows, urllib_get, write_jsonl

NAME, DOMAIN = "math_corpus", "math"

SOURCES: dict[str, str] = {
    "OpenMathReasoning": "nvidia/OpenMathReasoning",
    "DeepMath-103K": "zwhe99/DeepMath-103K",
    "NuminaMath-1.5": "AI-MO/NuminaMath-1.5",
    "PolyMath": "Qwen/PolyMath",
    "Omni-MATH": "KbsdJames/Omni-MATH",
    "FineProofs-SFT": "lm-provers/FineProofs-SFT",
    "AIME-1983-2024": "di-zhang-fdu/AIME_1983_2024",
    "Putnam-AXIOM": "kaifronsdal/Putnam-AXIOM",
}

SOLUTION_CAP = 5000
EVAL_FAMILIES = ("imo", "aime", "hmmt", "smt", "usamo", "putnam")
NUMINA_COMPETITION = {"olympiads", "olympiads_ref", "amc_aime", "aops_forum", "number_theory",
                      "inequalities", "cn_contest"}

# name -> (config, splits, problem keys, solution keys, difficulty keys)
_LAYOUT: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    "OpenMathReasoning": ("default", ("cot",), ("problem",), ("generated_solution",), ()),
    "DeepMath-103K": ("default", ("train",), ("question", "problem"), ("r1_solution_1", "solution"),
                      ("difficulty",)),
    "NuminaMath-1.5": ("default", ("train",), ("problem",), ("solution",), ()),
    "PolyMath": ("en", ("low", "medium", "high", "top"), ("question", "problem"), ("answer",), ()),
    "Omni-MATH": ("default", ("test",), ("problem",), ("solution",), ("difficulty",)),
    "FineProofs-SFT": ("default", ("train",), ("problem", "question"), ("solution", "proof"), ()),
    "AIME-1983-2024": ("default", ("train",), ("Question", "question", "problem"), ("Answer", "answer"), ()),
    "Putnam-AXIOM": ("original", ("train",), ("problem",), ("solution",), ()),
}


def _first(row: dict, keys: tuple[str, ...]):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _number(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_eval_family(source) -> bool:
    text = str(source or "").lower()
    return any(family in text for family in EVAL_FAMILIES)


def _adapt(name: str, row: dict, position: int) -> tuple[dict, float] | None:
    """One raw row -> (corpus row, tie-break score) or None when the row is filtered out."""
    _config, _splits, problem_keys, solution_keys, difficulty_keys = _LAYOUT[name]
    problem = _first(row, problem_keys)
    if problem is None:
        return None
    solution = str(_first(row, solution_keys) or "")
    score = 0.0
    if name == "NuminaMath-1.5":
        if str(row.get("source", "")).lower() not in NUMINA_COMPETITION:
            return None
    if name == "OpenMathReasoning":
        if _is_eval_family(row.get("problem_source")):
            return None
        rate = _number(row.get("pass_rate_72b_tir"))
        score = -1.0 if rate is None else rate
    if name in ("OpenMathReasoning", "DeepMath-103K"):
        solution = solution[:SOLUTION_CAP]
    difficulty = _number(_first(row, difficulty_keys)) if difficulty_keys else None
    return ({"id": f"{name}:{position}", "problem": str(problem), "solution": solution,
             "source": name, "difficulty": difficulty}, score)


def _source_rows(name: str, http, limit: int | None) -> list[dict]:
    config, splits, *_ = _LAYOUT[name]
    raw: list[dict] = []
    for split in splits:
        raw.extend(hf_rows(SOURCES[name], config, split, http, limit=limit))
    if limit is not None:
        raw = raw[:limit]
    adapted = [pair for position, row in enumerate(raw) if (pair := _adapt(name, row, position))]
    if name == "OpenMathReasoning":  # one solution per problem: the highest pass rate wins
        best: dict[str, tuple[dict, float]] = {}
        for item, score in adapted:
            current = best.get(item["problem"])
            if current is None or score > current[1]:
                best[item["problem"]] = (item, score)
        return [item for item, _ in best.values()]
    return [item for item, _ in adapted]


def fetch(cache_dir, http=urllib_get, per_source_limit: int | None = None) -> dict[str, Path]:
    """Download and merge the eight sources -> {"corpus": corpus.jsonl}.

    The cache is reused only when it was built with the same `per_source_limit`."""
    cache = Path(cache_dir)
    corpus, meta = cache / "corpus.jsonl", cache / "corpus.meta.json"
    if corpus.exists() and meta.exists():
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("per_source_limit") == per_source_limit:
                return {"corpus": corpus}
        except ValueError:
            pass
    rows: list[dict] = []
    for name in SOURCES:
        rows.extend(_source_rows(name, http, per_source_limit))
    write_jsonl(corpus, rows)
    meta.write_text(json.dumps({"per_source_limit": per_source_limit, "rows": len(rows)}),
                    encoding="utf-8")
    return {"corpus": corpus}
