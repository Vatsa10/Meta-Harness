import hashlib
import json
import threading

import pytest

from meta_harness import rollout
from meta_harness.__main__ import main
from meta_harness.rollout import Router, analyze, fisher_exact, read_log, z_test


def rows(c_succ, c_n, t_succ, t_n, c_cost=1.0, t_cost=1.0):
    out = []
    for arm, s, n, cost in (("control", c_succ, c_n, c_cost), ("treatment", t_succ, t_n, t_cost)):
        out += [{"arm": arm, "success": i < s, "cost": cost, "latency_ms": 10.0 + i % 5}
                for i in range(n)]
    return out


def test_bucket_stable_across_instances_and_known_value():
    a, b = Router("c", "t", 50, "salt"), Router("c", "t", 50, "salt")
    assert [a.bucket(f"k{i}") for i in range(50)] == [b.bucket(f"k{i}") for i in range(50)]
    import hashlib
    expect = int.from_bytes(hashlib.sha256(b"saltuser-1").digest()[:8], "big") % 10000
    assert a.bucket("user-1") == expect


def test_proportion_within_tolerance_and_salt_matters():
    for percent in (10, 50, 90):
        r = Router("c", "t", percent, "s")
        share = sum(r.route(f"key-{i}") == "treatment" for i in range(10000)) / 10000
        assert abs(share - percent / 100) < 0.02
    a, b = Router("c", "t", 50, "s1"), Router("c", "t", 50, "s2")
    assert sum(a.route(f"k{i}") != b.route(f"k{i}") for i in range(1000)) > 300


def test_extremes_and_validation():
    assert all(Router("c", "t", 0).route(f"k{i}") == "control" for i in range(500))
    assert all(Router("c", "t", 100).route(f"k{i}") == "treatment" for i in range(500))
    for bad in (-1, 100.5):
        with pytest.raises(ValueError):
            Router("c", "t", bad)


def test_sticky_when_percent_grows():
    low, high = Router("c", "t", 20, "s"), Router("c", "t", 60, "s")
    keys = [f"u{i}" for i in range(2000)]
    assert all(low.route(k) == low.route(k) for k in keys)
    assert all(high.route(k) == "treatment" for k in keys if low.route(k) == "treatment")
    assert low.target("u1") in ("c", "t")


def test_fisher_hand_computed():
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(34 / 70)
    assert fisher_exact(8, 2, 1, 5) == pytest.approx(400 / 11440)
    assert fisher_exact(5, 5, 5, 5) == pytest.approx(1.0)
    assert fisher_exact(0, 0, 0, 0) == 1.0


def test_z_hand_computed():
    z, p = z_test(30, 100, 45, 100)
    assert z == pytest.approx(0.15 / (0.375 * 0.625 * 0.02) ** 0.5)
    assert z == pytest.approx(2.1909, abs=1e-3)
    assert p == pytest.approx(0.02846, abs=1e-4)
    assert z_test(10, 20, 10, 20) == (0.0, 1.0)


def test_verdicts():
    assert analyze(rows(30, 100, 45, 100), min_samples=100)["verdict"] == "promote"
    assert analyze(rows(45, 100, 30, 100), min_samples=100)["verdict"] == "rollback"
    assert analyze(rows(40, 100, 44, 100), min_samples=100)["verdict"] == "hold"
    small = analyze(rows(30, 100, 45, 100), min_samples=200)
    assert small["verdict"] == "hold" and "insufficient" in small["reason"]
    blown = analyze(rows(30, 100, 45, 100, t_cost=1.5), min_samples=100)
    assert blown["verdict"] == "rollback" and "cost" in blown["reason"]
    ok = analyze(rows(30, 100, 45, 100, t_cost=1.05), min_samples=100)
    assert ok["verdict"] == "promote"
    assert analyze(rows(30, 100, 45, 100, t_cost=1.5), min_samples=100,
                   max_cost_increase=0.6)["verdict"] == "promote"
    strict = analyze(rows(30, 100, 45, 100), min_samples=100, alpha=0.01)
    assert strict["verdict"] == "hold"


def test_summary_latency_and_unjudged():
    data = rows(1, 4, 2, 4) + [{"arm": "control", "success": None, "cost": 0, "latency_ms": 1000}]
    s = rollout.summarize(data)
    assert s["control"]["n"] == 5 and s["control"]["n_judged"] == 4
    assert s["control"]["success_rate"] == 0.25
    assert s["control"]["latency_p95_ms"] > s["control"]["latency_p50_ms"]


def test_record_redacts_and_roundtrips(tmp_path):
    log = tmp_path / "r.jsonl"
    r = Router("c", "t", 50, "s", log=log)
    r.record("user sk-abcdefghijklmnop", "control", True, 0.5, 12.0,
             {"note": "api_key=hunter2hunter2", "n": 1})
    text = log.read_text(encoding="utf-8")
    assert "sk-abcdefghij" not in text and "hunter2" not in text
    row = read_log(log)[0]
    assert row["arm"] == "control" and row["success"] is True and row["meta"]["n"] == 1
    with pytest.raises(ValueError):
        r.record("k", "other")


def test_threaded_appends_all_intact(tmp_path):
    log = tmp_path / "r.jsonl"
    r = Router("c", "t", 50, log=log)

    def work(n):
        for i in range(100):
            r.record(f"{n}-{i}", "control", True, 1.0, 1.0)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len(read_log(log)) == 800
    assert len(log.read_text(encoding="utf-8").splitlines()) == 800


def test_torn_line_tolerated_and_not_glued(tmp_path):
    log = tmp_path / "r.jsonl"
    r = Router("c", "t", 50, log=log)
    r.record("a", "control", True)
    with log.open("a", encoding="utf-8") as h:
        h.write('{"arm": "treat')  # crash mid-write, no newline
    assert len(read_log(log)) == 1
    r.record("b", "treatment", False)
    got = read_log(log)
    assert [x["key_hash"] for x in got] == [r.key_hash("a"), r.key_hash("b")]


def test_cli_report_json_and_table(tmp_path, capsys):
    log = tmp_path / "r.jsonl"
    log.write_text("".join(json.dumps(x) + "\n" for x in rows(30, 100, 45, 100)) + "{torn",
                   encoding="utf-8")
    assert main(["rollout", "report", "--log", str(log), "--min-samples", "100", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["verdict"] == "promote" and data["arms"]["treatment"]["n"] == 100
    assert main(["rollout", "report", "--log", str(log), "--min-samples", "100"]) == 0
    out = capsys.readouterr().out
    assert "verdict: promote" in out and "control" in out


def test_both_tests_must_agree():
    # z-test alone says p=0.025, Fisher says p=0.087: not significant
    assert analyze(rows(0, 10, 4, 10), min_samples=10)["verdict"] == "hold"
    assert analyze(rows(4, 10, 0, 10), min_samples=10)["verdict"] == "hold"


def test_read_log_skips_non_rows(tmp_path):
    log = tmp_path / "r.jsonl"
    log.write_text('[1, 2]\n"x"\n{"arm": "bogus"}\n\n{"arm": "control", "success": true}\n',
                   encoding="utf-8")
    assert len(read_log(log)) == 1


def test_no_raw_key_or_pii_in_log(tmp_path):
    log = tmp_path / "r.jsonl"
    r = Router("c", "t", 50, holdout_salt="s", log=log)
    r.record("user@example.com", "control", True, meta={
        "email": "mail bob@example.com now", "phone": "call +1 (415) 555-0132",
        "card": "4111 1111 1111 1111", "nested": [{"n": "id 8675309123"}], "ok": "fine", "n": 3})
    r.record("8675309123", "treatment", False)
    text = log.read_text(encoding="utf-8")
    for leak in ("user@example.com", "bob@example.com", "8675309123", "555-0132", "4111"):
        assert leak not in text
    row = read_log(log)[0]
    assert row["key_hash"] == hashlib.sha256(b"suser@example.com").hexdigest()[:16]
    assert "key" not in row and row["meta"]["ok"] == "fine" and row["meta"]["n"] == 3


def test_holdout_salt_name_and_alias():
    a = Router("c", "t", 50, holdout_salt="x")
    b = Router("c", "t", 50, salt="x")
    assert a.holdout_salt == b.salt == "x"
    assert [a.route(f"k{i}") for i in range(100)] == [b.route(f"k{i}") for i in range(100)]
