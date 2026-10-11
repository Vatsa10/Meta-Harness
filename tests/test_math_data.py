import json
import random
import urllib.parse
from pathlib import Path

from meta_harness import decontam
from meta_harness.commands import math_corpus as command
from meta_harness.httpdata import read_jsonl
from meta_harness.loaders import imo, math_corpus, math_search

FIXTURES = Path(__file__).parent / "fixtures" / "math"
HF = json.loads((FIXTURES / "hf_rows.json").read_text(encoding="utf-8"))
CSV = json.loads((FIXTURES / "imo_csv.json").read_text(encoding="utf-8"))

LONG = ("Let $a_1, a_2, \\ldots, a_n$ be a sequence of positive integers such that every term after "
        "the first is the sum of the squares of the digits of the previous term, and suppose that the "
        "sequence eventually becomes periodic with period dividing twelve. Determine, with complete "
        "justification, the number of starting values below one million for which the period of the "
        "resulting cycle equals exactly eight, and express your final answer as a single positive "
        "integer written in base ten.")


def fake_http(url: str) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.path.endswith(".csv"):
        return CSV[parsed.path.rsplit("/", 1)[1]].encode("utf-8")
    query = urllib.parse.parse_qs(parsed.query)
    rows = HF[query["dataset"][0]]
    offset, length = int(query["offset"][0]), int(query["length"][0])
    page = [{"row": row} for row in rows[offset:offset + length]]
    return json.dumps({"rows": page, "num_rows_total": len(rows)}).encode("utf-8")


def item(ident, problem):
    return {"id": ident, "problem": problem}


# ---- decontamination -------------------------------------------------------------------

def test_removes_exact_prefix_and_near_duplicate_keeps_unrelated():
    near = LONG.replace("exactly eight", "exactly seven")
    prefix = LONG[:80] + " a completely different ending that shares only the opening."
    corpus = [item("exact", LONG), item("near", near), item("prefix", prefix),
              item("other", "Compute the area of a triangle with sides three, four and five units.")]
    assert decontam.jaccard(decontam.shingles(LONG), decontam.shingles(near)) >= 0.8
    kept, removed = decontam.decontaminate(corpus, [LONG])
    assert [r["id"] for r in removed] == ["exact", "near", "prefix"]
    assert [k["id"] for k in kept] == ["other"]


def test_prefix_only_match_is_removed_without_jaccard():
    held = "Show that for every positive integer n the number n^3 - n is divisible by six always."
    corpus = [item("p", held[:70] + " zzz yyy xxx www vvv uuu ttt sss rrr qqq")]
    kept, removed = decontam.decontaminate(corpus, [held], threshold=1.1)
    assert [r["id"] for r in removed] == ["p"] and not kept


def test_whitespace_and_latex_variants_still_match():
    variant = "  " + LONG.replace("$", "").upper().replace("\\LDOTS", "\\ldots").replace(" ", "  ") + "\n"
    assert decontam.normalise(variant) == decontam.normalise(LONG.replace("$", ""))
    kept, removed = decontam.decontaminate([item("v", variant)], [LONG])
    assert not kept and [r["id"] for r in removed] == ["v"]
    wrapped = "$$\\text{Find}$$ \\(x\\) such that \\textbf{x} is prime and even"
    assert decontam.normalise(wrapped) == "find x such that x is prime and even"


def test_inverted_index_matches_brute_force_reference():
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(40)]

    def make():
        return " ".join(rng.choice(vocab) for _ in range(rng.randint(8, 30)))

    held = [make() for _ in range(12)]
    corpus = [item(str(i), rng.choice([make(), held[rng.randrange(12)], make()])) for i in range(50)]
    # a few mutated copies so near matches occur
    for i in range(0, 50, 7):
        words = held[i % 12].split()
        words[len(words) // 2] = "mutated"
        corpus[i]["problem"] = " ".join(words)

    def brute(problem):
        a = decontam.normalise(problem)
        for h in held:
            b = decontam.normalise(h)
            if a[:60] == b[:60] and (len(a) >= 60 and len(b) >= 60 or a == b):
                return True
            if decontam.jaccard(decontam.shingles(problem), decontam.shingles(h)) >= 0.8:
                return True
        return False

    _, removed = decontam.decontaminate(corpus, held)
    expected = [c["id"] for c in corpus if brute(c["problem"])]
    assert [r["id"] for r in removed] == expected
    assert 0 < len(expected) < 50


def test_dedup_keeps_one_of_two_near_duplicates():
    near = LONG.replace("exactly eight", "exactly seven")
    other = item("c", "Compute the area of a triangle with sides three, four and five units.")
    kept = decontam.dedup([item("a", LONG), item("b", near), other])
    assert [k["id"] for k in kept] == ["a", "c"]


# ---- loaders ---------------------------------------------------------------------------

def test_corpus_loader_maps_rows_filters_and_truncates(tmp_path):
    rows = read_jsonl(math_corpus.fetch(tmp_path, fake_http)["corpus"])
    assert set(rows[0]) == {"id", "problem", "solution", "source", "difficulty"}
    by_source = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    assert set(by_source) == set(math_corpus.SOURCES)
    omr = by_source["OpenMathReasoning"]
    assert [r["solution"][:8] for r in omr if r["problem"].startswith("Problem 1:")] == ["sol high"]
    assert not any(r["solution"] == "from imo" for r in omr)          # eval-family source dropped
    assert max(len(r["solution"]) for r in omr) == 5000                # truncated
    assert [r["problem"][:9] for r in by_source["NuminaMath-1.5"]] == ["Problem 5"]
    assert {r["difficulty"] for r in by_source["DeepMath-103K"]} == {8.5, 5.0}
    assert by_source["AIME-1983-2024"][0]["solution"] == "7"


def test_corpus_loader_limit_and_cache(tmp_path):
    rows = read_jsonl(math_corpus.fetch(tmp_path, fake_http, per_source_limit=1)["corpus"])
    assert [r["source"] for r in rows].count("DeepMath-103K") == 1
    calls = []
    math_corpus.fetch(tmp_path, lambda url: calls.append(url) or fake_http(url), per_source_limit=1)
    assert calls == []                                                # cached for the same limit
    math_corpus.fetch(tmp_path, lambda url: calls.append(url) or fake_http(url))
    assert calls                                                      # a different limit refetches


def test_search_loader_maps_problem_answer_and_hard_filter(tmp_path):
    rows = read_jsonl(math_search.fetch(tmp_path, fake_http, size=250, seed=0)["search"])
    assert all(set(r) == {"problem", "answer"} for r in rows)
    problems = {r["problem"]: r["answer"] for r in rows}
    assert problems[LONG] == "1234"                                   # list answer, $ stripped
    assert not any(p.startswith("Problem 8") for p in problems)       # easy Omni row excluded
    assert len(rows) == 3
    small = read_jsonl(math_search.fetch(tmp_path / "s", fake_http, size=2, seed=1)["search"])
    assert len(small) == 2


def test_imo_loader_answer_gradable_and_skipped(tmp_path):
    paths = imo.fetch(tmp_path, fake_http)
    rows = read_jsonl(paths["test"])
    assert all(set(r) == {"problem", "answer"} for r in rows)
    assert len(rows) == 10
    skipped = json.loads((tmp_path / "skipped.json").read_text(encoding="utf-8"))
    assert {s["source"] for s in skipped} == {"IMO-ProofBench"} and len(skipped) == 2
    assert not any("Prove" in r["problem"] for r in rows)
    assert any("a comma\nand a newline" in r["problem"] for r in rows)


def test_imo_stratified_sample_is_proportional_and_seeded():
    rows = [{"Category": "A"}] * 6 + [{"Category": "B"}] * 4
    picked = imo.stratified_sample(rows, "Category", 5, seed=0)
    assert [r["Category"] for r in picked].count("A") == 3 and len(picked) == 5


# ---- build command ---------------------------------------------------------------------

def test_build_writes_clean_corpus_without_any_held_out_problem(tmp_path):
    report = command.build(tmp_path, http=fake_http)
    clean = read_jsonl(tmp_path / "corpus.clean.jsonl")
    held = [r["problem"] for r in read_jsonl(tmp_path / "math_search" / "search.jsonl")]
    held += [r["problem"] for r in read_jsonl(tmp_path / "imo" / "test.jsonl")]
    norm_held = {decontam.normalise(h) for h in held}
    assert clean and not any(decontam.normalise(c["problem"]) in norm_held for c in clean)
    assert LONG not in [c["problem"] for c in clean]
    saved = json.loads((tmp_path / "decontam_report.json").read_text(encoding="utf-8"))
    assert saved == report and report["removed"] >= 2
    assert set(report["removed_ids"]).isdisjoint({c["id"] for c in clean})
    assert report["kept"] == len(clean)


def test_command_is_registered():
    from meta_harness.commands import all_commands
    assert "math-corpus" in all_commands()
