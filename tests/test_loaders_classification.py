import json
import urllib.parse
from pathlib import Path

import pytest

from meta_harness import __main__ as cli
from meta_harness.domains import get_domain
from meta_harness.httpdata import read_jsonl
from meta_harness.loaders import all_loaders, lawbench, symptom2disease, uspto50k

FIX = Path(__file__).parent / "fixtures" / "loaders"
LAW_URL = lawbench.SOURCE


def _load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


class FakeHttp:
    """Serves fixtures: HF rows by (dataset, split, offset), raw files by exact URL."""

    def __init__(self, hf: dict, raw: dict | None = None):
        self.hf, self.raw, self.calls = hf, raw or {}, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if url in self.raw:
            return self.raw[url]
        query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        key = (query["dataset"][0], query["split"][0], int(query["offset"][0]))
        if key not in self.hf:
            raise AssertionError(f"unexpected request {url}")
        return json.dumps(_load(self.hf[key])).encode("utf-8")


def s2d_http():
    return FakeHttp({
        ("gretelai/symptom_to_diagnosis", "train", 0): "s2d_train_p0.json",
        ("gretelai/symptom_to_diagnosis", "train", 5): "s2d_train_p1.json",
        ("gretelai/symptom_to_diagnosis", "test", 0): "s2d_test_p0.json",
    })


def uspto_http():
    return FakeHttp({
        ("pingzhili/uspto-50k", "train", 0): "uspto_train_p0.json",
        ("pingzhili/uspto-50k", "validation", 0): "uspto_validation_p0.json",
    })


def law_http():
    return FakeHttp({}, {LAW_URL: (FIX / "lawbench_3-3.json").read_bytes()})


@pytest.fixture
def small_pages(monkeypatch):
    monkeypatch.setattr(symptom2disease, "PAGE", 5)
    monkeypatch.setattr(uspto50k, "PAGE", 10)


CASES = [
    (symptom2disease, s2d_http, {"train": 8, "val": 2, "test": 5}),
    (lawbench, law_http, {"train": 2, "val": 1, "test": 2}),
    (uspto50k, uspto_http, {"train": 8, "val": 2, "test": 5}),
]


@pytest.mark.parametrize("module,make_http,sizes", CASES, ids=lambda c: getattr(c, "NAME", None))
def test_loader_writes_splits_and_caches(module, make_http, sizes, tmp_path, small_pages):
    http = make_http()
    paths = module.fetch(tmp_path, http=http)
    assert set(paths) == {"train", "val", "test"}
    rows = {split: read_jsonl(path) for split, path in paths.items()}
    assert {s: len(r) for s, r in rows.items()} == sizes
    for split, split_rows in rows.items():
        for row in split_rows:
            assert set(row) == {"input", "label", "labels"}
            assert row["input"] and row["label"] and row["labels"] == sorted(row["labels"], key=lambda x: (len(x), x)) \
                or row["labels"] == sorted(row["labels"])
    train_inputs = {r["input"] for r in rows["train"]}
    assert not train_inputs & {r["input"] for r in rows["val"]}
    assert module.NAME in all_loaders() and module.DOMAIN == "classification"
    first_calls = len(http.calls)
    assert first_calls > 0
    again = module.fetch(tmp_path, http=http)
    assert again == paths and len(http.calls) == first_calls


def test_symptom2disease_labels_and_seed(tmp_path, small_pages):
    paths = symptom2disease.fetch(tmp_path / "a", http=s2d_http())
    train = read_jsonl(paths["train"]) + read_jsonl(paths["val"])
    expected = sorted({r["row"]["output_text"] for p in ("s2d_train_p0.json", "s2d_train_p1.json")
                       for r in _load(p)["rows"]})
    assert train[0]["labels"] == expected
    again = symptom2disease.fetch(tmp_path / "b", http=s2d_http())
    assert read_jsonl(again["val"]) == read_jsonl(paths["val"])
    other = symptom2disease.fetch(tmp_path / "c", http=s2d_http(), seed=1)
    assert read_jsonl(other["val"]) != read_jsonl(paths["val"])
    test_inputs = [r["row"]["input_text"] for r in _load("s2d_test_p0.json")["rows"]]
    assert [r["input"] for r in read_jsonl(paths["test"])] == test_inputs


def test_val_comes_from_train_not_test(tmp_path, small_pages):
    paths = symptom2disease.fetch(tmp_path, http=s2d_http())
    test_inputs = {r["input"] for r in read_jsonl(paths["test"])}
    assert not test_inputs & {r["input"] for r in read_jsonl(paths["val"])}
    assert not test_inputs & {r["input"] for r in read_jsonl(paths["train"])}


def test_lawbench_label_is_bare_charge_and_labels_exclude_test_only(tmp_path):
    paths = lawbench.fetch(tmp_path, http=law_http())
    rows = {s: read_jsonl(p) for s, p in paths.items()}
    raw = _load("lawbench_3-3.json")
    all_labels = {r["answer"].split(":", 1)[1] for r in raw}
    seen = {r["label"] for p in paths.values() for r in read_jsonl(p)}
    assert seen == all_labels and all(":" not in label for label in seen)
    trained = {r["label"] for r in rows["train"] + rows["val"]}
    for row in rows["train"] + rows["val"] + rows["test"]:
        assert row["labels"] == sorted(trained)
    assert {r["label"] for r in rows["test"]} - trained  # fixture has test-only labels
    assert lawbench.normalise_answer("罪名:盗窃") == "盗窃"


def test_lawbench_limit_and_parameter_keyed_cache(tmp_path):
    http = law_http()
    paths = lawbench.fetch(tmp_path, http=http, limit=3)
    assert sum(len(read_jsonl(p)) for p in paths.values()) == 3
    lawbench.fetch(tmp_path, http=http, limit=3)
    assert len(http.calls) == 1
    lawbench.fetch(tmp_path, http=http, limit=None, test_fraction=0.4)
    assert len(http.calls) == 2


def test_seed_change_resplits_same_dir(tmp_path, small_pages):
    http = s2d_http()
    a = symptom2disease.fetch(tmp_path, http=http, seed=0)
    first = read_jsonl(a["val"])
    n = len(http.calls)
    b = symptom2disease.fetch(tmp_path, http=http, seed=1)
    assert len(http.calls) > n and read_jsonl(b["val"]) != first
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["seed"] == 1


def test_default_limit_bounds_http_calls(tmp_path):
    import inspect
    assert inspect.signature(symptom2disease.fetch).parameters["limit"].default == 200
    served = []

    def http(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        off, length = int(q["offset"][0]), int(q["length"][0])
        served.append((q["split"][0], off, length))
        rows = [{"row_idx": i, "row": {"input_text": f"t{q['split'][0]}{i}", "output_text": f"d{i % 3}"}}
                for i in range(off, off + length)]
        return json.dumps({"rows": rows, "num_rows_total": 5000}).encode("utf-8")

    paths = symptom2disease.fetch(tmp_path, http=http)
    assert len(served) == 4  # 2 pages of 100 per split, never the 5000
    assert sum(len(read_jsonl(p)) for k, p in paths.items() if k != "test") == 200
    assert len(read_jsonl(paths["test"])) == 200


def test_uspto_uses_validation_as_test_and_numeric_labels(tmp_path, small_pages):
    paths = uspto50k.fetch(tmp_path, http=uspto_http())
    test = read_jsonl(paths["test"])
    served = _load("uspto_validation_p0.json")["rows"]
    assert [r["input"] for r in test] == [r["row"]["prod_smiles"] for r in served]
    train_classes = {str(r["row"]["class"]) for r in _load("uspto_train_p0.json")["rows"]}
    assert test[0]["labels"] == sorted(train_classes, key=int) == ["1", "2", "3", "6", "7", "9"]
    assert all(r["label"] in {"1", "5", "6"} for r in test)


def test_fetch_command_unknown_loader_exits_2(capsys):
    assert cli.main(["fetch", "nope"]) == 2
    err = capsys.readouterr().err
    assert "symptom2disease" in err and "lawbench" in err and "uspto50k" in err


def test_fetch_command_prints_split_paths(monkeypatch, tmp_path, capsys):
    seen = {}

    def stub(cache_dir, **kwargs):
        seen["dir"] = cache_dir
        seen["kwargs"] = kwargs
        return {"train": cache_dir / "train.jsonl"}

    monkeypatch.setattr(symptom2disease, "fetch", stub)
    assert cli.main(["fetch", "symptom2disease", "--dir", str(tmp_path)]) == 0
    first_out = capsys.readouterr().out
    assert seen["dir"] == tmp_path / "symptom2disease" and seen["kwargs"] == {}
    assert cli.main(["fetch", "symptom2disease", "--dir", str(tmp_path), "--full"]) == 0
    assert seen["kwargs"] == {"limit": None}
    assert json.loads(first_out) == {"train": str(tmp_path / "symptom2disease" / "train.jsonl")}


def test_classification_domain_contract():
    d = get_domain("classification")
    assert d.PROTOCOL == "online" and d.SEED_DIR.name == "classification" and "accuracy" in d.METRICS
    ok = d.adapt([{"input": "x", "label": "a", "labels": ["a", "b"]}])
    assert ok == [{"input": "x", "label": "a", "labels": ["a", "b"]}]
    with pytest.raises(ValueError):
        d.adapt([{"input": "x"}])
