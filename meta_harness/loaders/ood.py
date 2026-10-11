"""The nine out-of-distribution text classification sets of the paper (Table 5, App. C.1).

Every set becomes a list of classification tasks `{"input", "label", "labels"}` written to
`<cache_dir>/<name>/test.jsonl`. The paper evaluates a fixed subsample, so each set is capped at
`OodSpec.limit` rows chosen with a seeded shuffle (same seed, same rows).

Sources, verified against what the servers actually return (2026-10-11):

* scicite    raw tar.gz `scicite/test.jsonl` (1861 rows); fields `string`, `label`
             (background / method / result).
             https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz
* finer139   HF nlpaueb/finer-139, config finer-139, split test (108378 rows); fields `tokens`,
             `ner_tags` (279 BIO tags over 139 XBRL types, names served in the features).
* amazon5    HF SetFit/amazon_reviews_multi_en, config default, split test (5000 rows); fields
             `text`, int `label` 0..4 (0-based star rating; `label_text` is just that number as a string),
             rendered as "1 star".."5 stars".
* fpb        zip `FinancialPhraseBank-v1.0/Sentences_AllAgree.txt` (2264 lines, latin-1,
             `sentence@label`); there is no official split, so the whole file is the pool.
             https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip
* goemotions HF google-research-datasets/go_emotions, config simplified, split test (5427 rows);
             `labels` is a list of ClassLabel ids (28 names served in the features).
* banking77  raw csv `banking_data/test.csv` (3080 rows); fields `text`, `category` (77 intents).
             https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
* agnews     HF fancyzhx/ag_news, config default, split test (7600 rows); `text`, ClassLabel `label`.
* scitail    HF allenai/scitail, config tsv_format, split test (2126 rows); `premise`, `hypothesis`,
             string `label` (entails / neutral).
* tweet_hate HF cardiffnlp/tweet_eval, config hate, split test (2970 rows); `text`, ClassLabel `label`.

Framing decisions:

* GoEmotions is multi-label; only examples with exactly one label are kept, so the task is the
  paper's 28-way single-label classification.
* FiNER-139 is word-level NER with 139 entity types. The paper (App. C.1) describes it as "word-level
  annotations ... 139 fine-grained XBRL entity types" and as testing whether a model "can identify and
  classify numeric financial entities from context", but gives no further framing. Here each
  sentence is one example: its input is the sentence text, its label is the XBRL entity type it
  carries, and only sentences carrying exactly one distinct entity type are kept (sentences with
  no entity or with several types have no single sentence-level label). `labels` is the 139 type
  names, without the B-/I- prefix.
* Where an int label is served, it is mapped to its name: first from `OodSpec.label_names`, else
  from the ClassLabel names served with the rows.
* `labels` of amazon5 and scitail is the dataset's full class set, fixed in the spec; for the raw files (scicite, fpb, banking77) it is the sorted set over the whole file.
* HF sets are sampled by `block`-sized pages in a seeded order until `limit` kept rows are
  collected (at most `MAX_PAGES` page requests), then shuffled and cut to `limit`.
"""

from __future__ import annotations

import csv
import io
import json
import random
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..httpdata import HF_ROWS, Http, get_json, urllib_get, write_jsonl

NAME = "ood"
DOMAIN = "classification"
SOURCE = "nine OOD sets, see OOD_SPECS (paper Table 5, App. C.1)"

BLOCK = 100
MAX_PAGES = 60

SCICITE_URL = "https://s3-us-west-2.amazonaws.com/ai2-s2-research/scicite/scicite.tar.gz"
FPB_URL = ("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/"
           "FinancialPhraseBank-v1.0.zip")
BANKING_URL = ("https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/"
               "banking_data/test.csv")
FPB_MEMBER = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"
SCICITE_MEMBER = "scicite/test.jsonl"


@dataclass(frozen=True)
class OodSpec:
    name: str
    source: str
    text: str | tuple[str, ...]
    label: str
    label_names: tuple[str, ...] | None = None    # int label -> name
    split: str = "test"
    limit: int = 1000                             # paper evaluates a fixed subsample; seeded


OOD_SPECS: dict[str, OodSpec] = {
    "scicite": OodSpec("scicite", "raw: allenai scicite jsonl (scicite.tar.gz)", "string", "label"),
    "finer139": OodSpec("finer139", "hf:nlpaueb/finer-139/finer-139", "tokens", "ner_tags"),
    "amazon5": OodSpec("amazon5", "hf:SetFit/amazon_reviews_multi_en/default", "text", "label",
                       ("1 star", "2 stars", "3 stars", "4 stars", "5 stars")),
    "fpb": OodSpec("fpb", "zip: takala/financial_phrasebank FinancialPhraseBank-v1.0 "
                          "Sentences_AllAgree.txt", "sentence", "label"),
    "goemotions": OodSpec("goemotions", "hf:google-research-datasets/go_emotions/simplified",
                          "text", "labels"),
    "banking77": OodSpec("banking77", "raw: PolyAI task-specific-datasets banking_data/test.csv",
                         "text", "category"),
    "agnews": OodSpec("agnews", "hf:fancyzhx/ag_news/default", "text", "label",
                      ("World", "Sports", "Business", "Sci/Tech")),
    "scitail": OodSpec("scitail", "hf:allenai/scitail/tsv_format", ("premise", "hypothesis"), "label",
                        ("entails", "neutral")),
    "tweet_hate": OodSpec("tweet_hate", "hf:cardiffnlp/tweet_eval/hate", "text", "label",
                          ("non-hate", "hate")),
}


# -- row conversion ---------------------------------------------------------------------------
def _join_text(spec: OodSpec, row: dict) -> str:
    if isinstance(spec.text, tuple):
        return "\n".join(f"{field}: {row[field]}" for field in spec.text)
    return str(row[spec.text])


def _class_names(features: Any, field: str) -> tuple[str, ...] | None:
    """ClassLabel names served for `field` (scalar or list-of-ClassLabel), else None."""
    if isinstance(features, dict):
        features = [{"name": key, "type": value} for key, value in features.items()]
    for feature in features or []:
        if feature.get("name") != field:
            continue
        kind = feature.get("type") or {}
        if isinstance(kind.get("feature"), dict):
            kind = kind["feature"]
        names = kind.get("names")
        return tuple(names) if names else None
    return None


def _finer_types(names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({name[2:] for name in names if name[:2] in ("B-", "I-")}))


def convert_hf_row(spec: OodSpec, row: dict, names: tuple[str, ...] | None) -> dict | None:
    """One HF row -> {"input", "label"}; None when the framing drops the row."""
    raw = row[spec.label]
    if spec.name == "finer139":
        kinds = {names[tag][2:] for tag in raw if tag}
        if len(kinds) != 1:
            return None
        return {"input": " ".join(row["tokens"]), "label": kinds.pop()}
    if spec.name == "goemotions":
        if len(raw) != 1:
            return None
        raw = raw[0]
    if isinstance(raw, int):
        table = spec.label_names or names
        if not table:
            raise ValueError(f"{spec.name}: integer label {raw} but no label names")
        raw = table[raw]
    return {"input": _join_text(spec, row), "label": str(raw)}


# -- sources ----------------------------------------------------------------------------------
def _hf_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    from urllib.parse import urlencode
    return f"{HF_ROWS}?" + urlencode({"dataset": dataset, "config": config, "split": split,
                                      "offset": offset, "length": length})


def _hf_tasks(spec: OodSpec, limit: int, seed: int, http: Http) -> tuple[list[dict], list[str]]:
    dataset, config = spec.source[len("hf:"):].rsplit("/", 1)
    meta = get_json(_hf_url(dataset, config, spec.split, 0, 1), http)
    total = int(meta.get("num_rows_total", 0))
    features = meta.get("features", [])
    names = _class_names(features, spec.label)
    blocks = list(range((total + BLOCK - 1) // BLOCK))
    random.Random(seed).shuffle(blocks)
    kept: list[dict] = []
    for pages, block in enumerate(blocks):
        if len(kept) >= limit or pages >= MAX_PAGES:
            break
        payload = get_json(_hf_url(dataset, config, spec.split, block * BLOCK, BLOCK), http)
        for item in payload.get("rows", []):
            task = convert_hf_row(spec, item["row"], names)
            if task is not None:
                kept.append(task)
    if spec.name == "finer139":
        labels = list(_finer_types(names or ()))
    elif spec.label_names:
        labels = list(spec.label_names)
    elif names:
        labels = list(names)
    else:
        labels = sorted({task["label"] for task in kept})
    return kept, labels


def _scicite_tasks(spec: OodSpec, http: Http) -> list[dict]:
    with tarfile.open(fileobj=io.BytesIO(http(SCICITE_URL)), mode="r:gz") as archive:
        member = archive.extractfile(SCICITE_MEMBER)
        if member is None:
            raise ValueError(f"{SCICITE_MEMBER} missing from the scicite archive")
        text = member.read().decode("utf-8")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return [{"input": _join_text(spec, row), "label": str(row[spec.label])} for row in rows]


def _fpb_tasks(spec: OodSpec, http: Http) -> list[dict]:
    with zipfile.ZipFile(io.BytesIO(http(FPB_URL))) as archive:
        raw = archive.read(FPB_MEMBER)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    tasks = []
    for line in text.splitlines():
        if "@" in line:
            sentence, label = line.rsplit("@", 1)
            tasks.append({"input": sentence.strip(), "label": label.strip()})
    return tasks


def _banking_tasks(spec: OodSpec, http: Http) -> list[dict]:
    text = http(BANKING_URL).decode("utf-8")
    return [{"input": _join_text(spec, row), "label": row[spec.label]}
            for row in csv.DictReader(io.StringIO(text))]


_RAW: dict[str, Callable[[OodSpec, Http], list[dict]]] = {
    "scicite": _scicite_tasks, "fpb": _fpb_tasks, "banking77": _banking_tasks,
}


def subsample(tasks: list[dict], limit: int, seed: int) -> list[dict]:
    """Seeded, order-stable subsample: the kept rows keep their original relative order."""
    if len(tasks) <= limit:
        return list(tasks)
    order = list(range(len(tasks)))
    random.Random(seed).shuffle(order)
    return [tasks[i] for i in sorted(order[:limit])]


def build_tasks(name: str, http: Http = urllib_get, limit: int | None = None, seed: int = 0) -> list[dict]:
    """Download one set and return its tasks, each with `input`, `label` and `labels`."""
    if name not in OOD_SPECS:
        raise KeyError(f"unknown OOD dataset {name!r}; known: {', '.join(sorted(OOD_SPECS))}")
    spec = OOD_SPECS[name]
    limit = spec.limit if limit is None else limit
    if name in _RAW:
        tasks = _RAW[name](spec, http)
        labels = sorted({task["label"] for task in tasks})
    else:
        tasks, labels = _hf_tasks(spec, limit, seed, http)
    tasks = subsample(tasks, limit, seed)
    return [{**task, "labels": list(labels)} for task in tasks]


def fetch_one(name: str, cache_dir: Path, http: Http = urllib_get, limit: int | None = None,
              seed: int = 0) -> Path:
    """Write (or reuse) `<cache_dir>/<name>/test.jsonl` and return its path.

    The default limit and seed use `test.jsonl`; any other choice gets its own file so a cached
    default is never mistaken for a different subsample."""
    if name not in OOD_SPECS:
        raise KeyError(f"unknown OOD dataset {name!r}; known: {', '.join(sorted(OOD_SPECS))}")
    spec = OOD_SPECS[name]
    custom = (limit is not None and limit != spec.limit) or seed != 0
    filename = f"test.limit{limit or spec.limit}.seed{seed}.jsonl" if custom else "test.jsonl"
    path = Path(cache_dir) / name / filename
    if path.is_file():
        return path
    return write_jsonl(path, build_tasks(name, http, limit, seed))


def fetch(cache_dir: Path, http: Http = urllib_get) -> dict[str, Path]:
    """Loader contract: every OOD set, name -> test.jsonl."""
    return {name: fetch_one(name, cache_dir, http) for name in OOD_SPECS}
