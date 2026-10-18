"""Build the research_ops corpus and task file.

Corpus: the U.S. Constitution (Project Gutenberg eBook #5) split into its 24 article sections, plus
the Bill of Rights (eBook #2) as 10 amendments - 34 public-domain documents. Only the public-domain
text is kept (the Project Gutenberg header/footer and trademark are stripped).

    python examples/usecases/research_ops/build_data.py --fetch   # download into data/cache/constitution
    python examples/usecases/research_ops/build_data.py           # write data/corpus.jsonl + tasks.jsonl
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

from usecases._common import split_tasks, write_jsonl  # noqa: E402
from usecases.research_ops.questions import TASKS  # noqa: E402

REPO = HERE.parents[2]
CACHE = REPO / "data" / "cache" / "constitution"
SOURCES = {"pg5.txt": "https://www.gutenberg.org/cache/epub/5/pg5.txt",
           "pg2.txt": "https://www.gutenberg.org/cache/epub/2/pg2.txt"}
SEED = 20261018
ARTICLE_WORDS = {"1": 1, "2": 2, "THREE": 3, "FOUR": 4, "FIVE": 5, "SIX": 6, "SEVEN": 7}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]


def fetch() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        with urllib.request.urlopen(url, timeout=60) as r:
            (CACHE / name).write_bytes(r.read())


def _body(path: Path) -> str:
    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    start = text.index("*** START OF")
    start = text.index("\n", start) + 1
    return text[start:text.index("*** END OF")]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def constitution_docs() -> list[dict]:
    body = _body(CACHE / "pg5.txt")
    parts = re.split(r"\n(Article 1|ARTICLE (?:2|THREE|FOUR|FIVE|SIX|SEVEN))\s*\n", body)
    docs = []
    for header, text in zip(parts[1::2], parts[2::2]):
        art = ARTICLE_WORDS[header.split()[-1]]
        sections = re.split(r"\n\s*Section (\d+)\.\s+", "\n" + text)
        if len(sections) == 1:
            docs.append({"id": f"art{art}", "title": f"Article {art}", "text": _clean(text)})
            continue
        for num, sec in zip(sections[1::2], sections[2::2]):
            docs.append({"id": f"art{art}-sec{num}", "title": f"Article {art}, Section {num}", "text": _clean(sec)})
    return docs


def bill_of_rights_docs() -> list[dict]:
    body = _body(CACHE / "pg2.txt")
    parts = re.split(r"\n(" + "|".join(sorted(ROMAN, key=len, reverse=True)) + r")\s*\n", body)
    docs = []
    for numeral, text in zip(parts[1::2], parts[2::2]):
        n = ROMAN.index(numeral) + 1
        docs.append({"id": f"amend{n}", "title": f"Amendment {n} (Bill of Rights)", "text": _clean(text)})
    return docs


def build() -> None:
    docs = constitution_docs() + bill_of_rights_docs()
    ids = [d["id"] for d in docs]
    assert len(ids) == len(set(ids)) == 34, ids
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    write_jsonl(data / "corpus.jsonl", docs)
    tasks = []
    for i, t in enumerate(TASKS):
        if t["ticket"] and t["ticket"]["doc_id"] not in ids:
            raise SystemExit(f"task {i}: unknown doc {t['ticket']['doc_id']}")
        if t["source"] not in ids:
            raise SystemExit(f"task {i}: unknown source {t['source']}")
        tasks.append({"id": f"ops-{i:03d}", **t})
    write_jsonl(HERE / "tasks.jsonl", split_tasks(tasks, SEED, (len(tasks) - 25, 10, 15)))
    print(f"{len(docs)} docs, {len(tasks)} tasks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    if ap.parse_args().fetch:
        fetch()
    build()
