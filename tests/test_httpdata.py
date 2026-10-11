import json
from urllib.parse import parse_qs, urlparse

from meta_harness.httpdata import carve, hf_rows


def test_hf_rows_pages_and_respects_limit():
    seen = []
    def http(url):
        q = {k: v[0] for k, v in parse_qs(urlparse(url).query).items()}
        seen.append(int(q["offset"]))
        start, length = int(q["offset"]), int(q["length"])
        rows = [{"row": {"i": i}} for i in range(start, min(start + length, 5))]
        return json.dumps({"rows": rows, "num_rows_total": 5}).encode("utf-8")
    assert [r["i"] for r in hf_rows("d", "c", "train", http, page=2)] == [0, 1, 2, 3, 4]
    assert seen == [0, 2, 4]
    assert len(hf_rows("d", "c", "train", http, page=2, limit=3)) == 3


def test_carve_is_seeded_and_disjoint():
    rows = [{"i": i} for i in range(10)]
    a1, b1 = carve(rows, 0.2, 0)
    a2, b2 = carve(rows, 0.2, 0)
    assert (a1, b1) == (a2, b2) and len(b1) == 2
    assert not {r["i"] for r in a1} & {r["i"] for r in b1}
