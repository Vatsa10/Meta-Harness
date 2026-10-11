import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from meta_harness import __main__ as cli
from meta_harness.loaders import ood
from meta_harness.loaders.ood import OOD_SPECS, OodSpec, convert_hf_row, fetch_one, subsample

FIXTURES = Path(__file__).parent / "fixtures" / "ood"
BASELINE = Path(__file__).resolve().parent.parent / "baselines" / "classification" / "zero_shot.py"


def fixture(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class FakeHttp:
    """Serves the recorded fixtures by URL and counts calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url == ood.SCICITE_URL:
            buffer = io.BytesIO()
            data = "\n".join(fixture("scicite")["lines"]).encode("utf-8")
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                info = tarfile.TarInfo(ood.SCICITE_MEMBER)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            return buffer.getvalue()
        if url == ood.FPB_URL:
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as archive:
                archive.writestr(ood.FPB_MEMBER, "\n".join(fixture("fpb")["lines"]).encode("latin-1"))
            return buffer.getvalue()
        if url == ood.BANKING_URL:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, ["text", "category"])
            writer.writeheader()
            writer.writerows(fixture("banking77")["rows"])
            return buffer.getvalue().encode("utf-8")
        query = parse_qs(urlparse(url).query)
        for name, spec in OOD_SPECS.items():
            if spec.source.startswith("hf:") and query["dataset"][0] == spec.source[3:].rsplit("/", 1)[0]:
                recorded = fixture(name)
                offset, length = int(query["offset"][0]), int(query["length"][0])
                # a spec that names its labels must not depend on the server serving ClassLabel names
                features = [] if spec.label_names else recorded["features"]
                payload = {"features": features, "num_rows_total": len(recorded["rows"]),
                           "rows": recorded["rows"][offset:offset + length]}
                return json.dumps(payload).encode("utf-8")
        raise AssertionError(f"unexpected url {url}")


def load(name, cache, http=None, **kw):
    path = fetch_one(name, cache, http or FakeHttp(), **kw)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_nine_specs():
    assert sorted(OOD_SPECS) == sorted(["scicite", "finer139", "amazon5", "fpb", "goemotions",
                                        "banking77", "agnews", "scitail", "tweet_hate"])
    assert all(isinstance(s, OodSpec) and s.limit == 1000 for s in OOD_SPECS.values())


@pytest.mark.parametrize("name", sorted(OOD_SPECS))
def test_every_spec_loads_its_fixture_into_tasks(name, tmp_path):
    tasks = load(name, tmp_path)
    assert tasks
    for task in tasks:
        assert set(task) == {"input", "label", "labels"}
        assert task["input"].strip() and task["label"] in task["labels"]


def test_agnews_integer_labels_become_names(tmp_path):
    tasks = load("agnews", tmp_path)
    assert tasks[0]["label"] == "Business"          # served label 2
    assert tasks[0]["labels"] == ["World", "Sports", "Business", "Sci/Tech"]
    assert {t["label"] for t in tasks} <= set(tasks[0]["labels"])


def test_tweet_hate_names_and_scitail_join(tmp_path):
    hate = load("tweet_hate", tmp_path)
    assert hate[0]["label"] == "non-hate" and hate[0]["labels"] == ["non-hate", "hate"]
    row = fixture("scitail")["rows"][0]["row"]
    first = load("scitail", tmp_path)[0]
    assert row["premise"] in first["input"] and row["hypothesis"] in first["input"]
    assert first["input"].index(row["premise"]) < first["input"].index(row["hypothesis"])
    assert first["label"] == row["label"]


def test_goemotions_keeps_only_single_label_rows(tmp_path):
    recorded = fixture("goemotions")["rows"]
    assert sorted(len(r["row"]["labels"]) for r in recorded) == [1, 1, 1, 2, 2]
    tasks = load("goemotions", tmp_path)
    assert len(tasks) == 3 and len(tasks[0]["labels"]) == 28
    assert tasks[0]["label"] == "sadness"           # served id 25


def test_finer_sentence_framing(tmp_path):
    tasks = load("finer139", tmp_path)
    assert len(tasks) == 4                         # the all-O sentence is dropped
    assert len(tasks[0]["labels"]) == 139 and not tasks[0]["label"].startswith(("B-", "I-"))
    assert tasks[0]["label"] == "EffectiveIncomeTaxRateContinuingOperations"
    spec = OOD_SPECS["finer139"]
    names = ("O", "B-A", "I-A", "B-B", "I-B")
    two_types = {"tokens": ["x", "y", "z"], "ner_tags": [1, 3, 0]}
    assert convert_hf_row(spec, two_types, names) is None  # no single sentence-level label
    assert convert_hf_row(spec, {"tokens": ["x", "y"], "ner_tags": [1, 2]}, names)["label"] == "A"


def test_raw_sources(tmp_path):
    sci = load("scicite", tmp_path)
    assert len(sci) == 4 and {t["label"] for t in sci} <= {"background", "method", "result"}
    fpb = load("fpb", tmp_path)
    assert {t["label"] for t in fpb} == {"neutral", "positive", "negative"}
    assert not any("@" + t["label"] in t["input"] for t in fpb)
    bank = load("banking77", tmp_path)
    assert len(bank) == 4 and len(set(t["label"] for t in bank)) == 4


def test_amazon_uses_served_label_text(tmp_path):
    assert load("amazon5", tmp_path)[0]["label"] == fixture("amazon5")["rows"][0]["row"]["label_text"]


def test_limit_subsampling_is_seeded_and_stable(tmp_path):
    rows = [{"input": str(i), "label": "a"} for i in range(50)]
    assert subsample(rows, 7, 3) == subsample(rows, 7, 3)
    assert subsample(rows, 7, 3) != subsample(rows, 7, 4)
    assert len(subsample(rows, 7, 3)) == 7 and subsample(rows, 99, 3) == rows
    first = load("agnews", tmp_path / "a", limit=2, seed=1)
    again = load("agnews", tmp_path / "b", limit=2, seed=1)
    assert first == again and len(first) == 2
    full = load("agnews", tmp_path / "c")
    assert all(t in full for t in first)


def test_second_fetch_makes_no_requests(tmp_path):
    http = FakeHttp()
    fetch_one("agnews", tmp_path, http)
    calls = len(http.calls)
    assert calls > 0
    fetch_one("agnews", tmp_path, http)
    assert len(http.calls) == calls


def test_unknown_dataset_names_known_ones(tmp_path):
    with pytest.raises(KeyError, match="agnews"):
        fetch_one("nope", tmp_path, FakeHttp())


def _script(tmp_path):
    script = tmp_path / "script.jsonl"
    script.write_text(json.dumps({"default": "World"}) + "\n", encoding="utf-8")
    return script


def test_ood_command_writes_one_row_per_dataset_per_harness(tmp_path, capsys):
    cache = tmp_path / "cache"
    http = FakeHttp()
    for name in ("agnews", "scitail"):
        fetch_one(name, cache / "ood", http)
    second = tmp_path / "second.py"
    second.write_text("class Harness:\n    def run(self, task, model):\n        return 'neutral'\n",
                      encoding="utf-8")
    out = tmp_path / "out"
    code = cli.main(["ood", "--harness", str(BASELINE), "--harness", str(second),
                     "--datasets", "agnews,scitail", "--provider", "scripted",
                     "--model", str(_script(tmp_path)), "--cache-dir", str(cache), "--out", str(out)])
    assert code == 0
    rows = json.loads((out / "ood.json").read_text(encoding="utf-8"))["rows"]
    assert [(r["harness"], r["dataset"]) for r in rows] == [
        ("zero_shot", "agnews"), ("zero_shot", "scitail"), ("second", "agnews"), ("second", "scitail")]
    by = {(r["harness"], r["dataset"]): r for r in rows}
    agnews = fixture("agnews")["rows"]
    expected = sum(r["row"]["label"] == 0 for r in agnews) / len(agnews)
    assert by[("zero_shot", "agnews")]["accuracy"] == pytest.approx(expected)
    assert by[("zero_shot", "agnews")]["n"] == len(agnews)
    assert by[("zero_shot", "agnews")]["context_tokens"] > 0
    assert by[("second", "agnews")]["accuracy"] == 0.0
    assert by[("second", "scitail")]["accuracy"] > 0
    table = capsys.readouterr().out
    assert "accuracy" in table and "agnews" in table and "scitail" in table


def test_ood_command_rejects_unknown_dataset(tmp_path, capsys):
    code = cli.main(["ood", "--harness", str(BASELINE), "--datasets", "nope", "--provider", "scripted",
                     "--model", str(_script(tmp_path)), "--cache-dir", str(tmp_path),
                     "--out", str(tmp_path / "o")])
    assert code == 2 and "agnews" in capsys.readouterr().err
