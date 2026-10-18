"""Offline smoke tests for the real-world LangGraph use cases in examples/usecases.

A scripted fake chat model plays every node; graders must give the gold (oracle) policy 1.0 and a
broken policy less; every write must land in the per-task sandbox and nowhere else.
"""
import importlib
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from usecases._common import file_digest, read_jsonl, split_tasks  # noqa: E402

USECASES = ["tau_retail", "sql_analyst", "research_ops"]
EXPECTED = {"tau_retail": (50, 20, 15, 15), "sql_analyst": (40, 15, 10, 15), "research_ops": (36, 11, 10, 15)}


def mods(name):
    return tuple(importlib.import_module(f"usecases.{name}.{m}") for m in ("app", "grader", "oracle"))


def tasks(name):
    return read_jsonl(EXAMPLES / "usecases" / name / "tasks.jsonl")


def sample(name, n=2):
    ts = tasks(name)
    return [t for split in ("train", "search", "test") for t in [x for x in ts if x["split"] == split][:n]]


# ------------------------------------------------------------------ task files

@pytest.mark.parametrize("name", USECASES)
def test_task_files_have_seeded_splits(name):
    ts = tasks(name)
    total, train, search, test = EXPECTED[name]
    assert len(ts) == total and len({t["id"] for t in ts}) == total
    counts = {s: sum(t["split"] == s for t in ts) for s in ("train", "search", "test")}
    assert counts == {"train": train, "search": search, "test": test}
    stripped = [{k: v for k, v in t.items() if k != "split"} for t in ts]
    assert [t["split"] for t in split_tasks(stripped, 20261018, (train, search, test))] == [t["split"] for t in ts]
    assert [t["split"] for t in split_tasks(stripped, 1, (train, search, test))] != [t["split"] for t in ts]


# ------------------------------------------------------------------ graph runs + graders

@pytest.mark.parametrize("name", USECASES)
def test_oracle_policy_scores_one(name):
    app, grader, _ = mods(name)
    for task in sample(name):
        result = app.run(task)
        assert not result["error"], result["error"]
        assert result["tool_calls"], "the graph must call tools"
        assert grader.grade(task, result)["score"] == 1.0, (task["id"], grader.grade(task, result))


@pytest.mark.parametrize("name", USECASES)
def test_broken_policy_scores_below_one(name):
    app, grader, oracle = mods(name)
    train = [t for t in tasks(name) if t["split"] == "train"][:6]
    scores = [grader.grade(t, app.run(t, oracle.broken_model(t)))["score"] for t in train]
    assert sum(scores) / len(scores) < 1.0
    if name != "research_ops":  # ops' broken policy is only wrong when the gold ticket differs
        assert scores == [0.0] * len(scores)


@pytest.mark.parametrize("name", USECASES)
def test_writes_stay_in_the_per_task_sandbox(name, tmp_path):
    app, _, _ = mods(name)
    data = EXAMPLES / "usecases" / name / "data"
    before = {p.name: file_digest(p) for p in data.iterdir()}
    task = next(t for t in tasks(name) if name != "research_ops" or t["ticket"])
    boxes = []
    for i in range(2):  # two runs of the same task: each gets a fresh backend
        result = app.run(task, root=tmp_path, keep=True)
        box = Path(result["sandbox"])
        assert box.parent == tmp_path
        boxes.append(box)
        writes = [c for c in result["tool_calls"] if c["write"]]
        assert writes and not any(str(c["output"]).startswith("Error") for c in writes)
    assert boxes[0] != boxes[1]
    assert {p.name: file_digest(p) for p in data.iterdir()} == before
    if name == "tau_retail":
        assert file_digest(boxes[0] / "orders.json") == file_digest(boxes[1] / "orders.json")
        assert file_digest(boxes[0] / "orders.json") != file_digest(data / "orders.json")
    if name == "sql_analyst":
        assert file_digest(boxes[0] / "chinook.sqlite") == file_digest(data / "chinook.sqlite")
        assert (boxes[0] / "scratch.sqlite").stat().st_size > 0
    if name == "research_ops":
        assert len(json.loads((boxes[1] / "tickets.json").read_text())) == 1  # not 2: state was reset


# ------------------------------------------------------------------ backend policy constraints

def test_sql_writes_outside_scratch_are_refused(tmp_path):
    from usecases._common import Recorder, sandbox
    from usecases.sql_analyst.tools import DB_FILE, AnalystDB

    box = sandbox([EXAMPLES / "usecases" / "sql_analyst" / "data" / DB_FILE], tmp_path)
    db = AnalystDB(box)
    original = file_digest(box / DB_FILE)
    for sql in ("DELETE FROM Track", "UPDATE Track SET Name = 'x'", "CREATE TABLE main.evil(a)", "DROP TABLE Album",
                "ATTACH DATABASE 'x.db' AS x", "CREATE TABLE scratch.t(a)"):
        assert str(db.run_sql(sql)).startswith("Error"), sql
    assert str(db.save_result("ok", "SELECT 1 AS a")).startswith("{'saved'")
    assert str(db.insert_rows("ok", "SELECT 2")).startswith("{'inserted_into'")
    assert db.scratch_rows("ok") == [(1,), (2,)]
    assert str(db.save_result("x; DROP TABLE Track", "SELECT 1")).startswith("Error")
    assert str(db.save_result("evil", "SELECT 1; DELETE FROM Track")).startswith("Error")
    assert str(db._scratch_write("DELETE FROM main.Track")).startswith("Error")  # even inside a write tool
    assert str(db.insert_rows("ok", "SELECT 3; DELETE FROM Track")).startswith("Error")
    db.close()
    assert file_digest(box / DB_FILE) == original
    assert Recorder  # imported for parity with the app's tool factory


def test_tau_policy_constraints(tmp_path):
    from usecases._common import sandbox
    from usecases.tau_retail.tools import DB_FILES, RetailDB

    data = EXAMPLES / "usecases" / "tau_retail" / "data"
    db = RetailDB(sandbox([data / f for f in DB_FILES], tmp_path))
    delivered = next(o for o, v in db.orders.items() if v["status"] == "delivered")
    pending = next(o for o, v in db.orders.items() if v["status"] == "pending")
    assert db.cancel_order(delivered, "no longer needed").startswith("Error: non-pending")
    assert db.cancel_order(pending, "too expensive") == "Error: invalid reason"
    assert db.return_items(pending, [], "x").startswith("Error: non-delivered")
    delivered, other = next((o, m) for o, v in db.orders.items() if v["status"] == "delivered"
                            for m in db.users[v["user_id"]]["payment_methods"]
                            if m != v["payment_history"][0]["payment_method_id"] and "gift_card" not in m)
    order = db.orders[delivered]
    item = order["items"][0]["item_id"]
    assert "original payment method" in db.return_items(delivered, [item], other)
    digest = db.state_digest()
    assert db.cancel_order(pending, "ordered by mistake")["status"] == "cancelled"
    assert db.state_digest() != digest


def test_write_tools_classify_as_writes_with_the_shipped_kinds():
    from meta_harness.integrations.langgraph.tools import classify
    from usecases._common import Recorder
    from usecases.tau_retail.tools import TOOL_KINDS, make_tools

    tools = make_tools(EXAMPLES / "usecases" / "tau_retail" / "data", Recorder())
    kinds = {t.name: classify(t, TOOL_KINDS) for t in tools}
    assert kinds["return_items"] == kinds["exchange_items"] == kinds["cancel_order"] == "write"
    assert kinds["get_order"] == kinds["list_products"] == "read"


# ------------------------------------------------------------------ grader mutations

def test_tau_grader_needs_state_and_outputs():
    from usecases.tau_retail.grader import grade

    task = {"gold_state_digest": "abc", "required_outputs": ["1,234.50"]}
    assert grade(task, {"state_digest": "abc", "agent_messages": ["Total is $1234.50"]})["score"] == 1.0
    assert grade(task, {"state_digest": "abd", "agent_messages": ["Total is 1234.50"]})["score"] == 0.0
    assert grade(task, {"state_digest": "abc", "agent_messages": ["Total is 1234"]})["score"] == 0.0
    assert grade({**task, "required_outputs": []}, {"state_digest": "abc", "agent_messages": []})["score"] == 1.0


def test_sql_grader_is_set_equality():
    from usecases.sql_analyst.grader import grade

    task = {"gold_rows": [["Rock", 826.65], ["Latin", 382.14]]}
    assert grade(task, {"result_rows": [(382.14, "latin"), (826.649, "Rock")]})["score"] == 1.0
    assert grade(task, {"result_rows": [("Rock", 826.65)]})["score"] == 0.0
    assert grade(task, {"result_rows": [("Rock", 826.65), ("Latin", 382.14), ("Rock", 826.65)]})["score"] == 0.0
    assert grade(task, {"result_rows": [("Rock", 826.65, 1), ("Latin", 382.14, 2)]})["score"] == 0.0
    assert grade(task, {"result_rows": [("Rock", 826.75), ("Latin", 382.14)]})["score"] == 0.0
    assert grade(task, {"result_rows": None})["score"] == 0.0
    assert grade({"gold_rows": []}, {"result_rows": None})["score"] == 0.0  # nothing saved is never a match


def test_ops_grader_checks_answer_and_ticket_fields():
    from usecases.research_ops.grader import grade

    gold = {"queue": "eligibility", "doc_id": "art1-sec2", "priority": "high"}
    task = {"accept": ["25", "twenty five"], "ticket": gold}
    good = {**gold, "ticket_id": "T-0001", "summary": "s"}
    assert grade(task, {"answer": "25 years", "tickets": [good]})["score"] == 1.0
    assert grade(task, {"answer": "Twenty-five", "tickets": [good]})["score"] == 1.0
    assert grade(task, {"answer": "250", "tickets": [good]})["score"] == 0.0
    assert grade(task, {"answer": "25", "tickets": [{**good, "priority": "normal"}]})["score"] == 0.0
    assert grade(task, {"answer": "25", "tickets": [{**good, "doc_id": "art1-sec3"}]})["score"] == 0.0
    assert grade(task, {"answer": "25", "tickets": [good, good]})["score"] == 0.0
    assert grade(task, {"answer": "25", "tickets": []})["score"] == 0.0
    no_ticket = {"accept": ["6"], "ticket": None}
    assert grade(no_ticket, {"answer": "6", "tickets": []})["score"] == 1.0
    assert grade(no_ticket, {"answer": "6", "tickets": [good]})["score"] == 0.0


# ------------------------------------------------------------------ traces + labels

@pytest.mark.parametrize("name", USECASES)
def test_traces_are_replayable_and_labelled(name):
    from meta_harness.integrations.langgraph.tools import Recordings

    base = EXAMPLES / "usecases" / name
    traces = read_jsonl(base / "traces.jsonl")
    labels = read_jsonl(base / "labels.jsonl")
    assert {t["id"] for t in traces} == {x["id"] for x in labels}
    assert {t["task_id"] for t in traces if t["policy"] == "oracle"} == {t["id"] for t in tasks(name)}
    assert all(t["score"] == 1.0 for t in traces if t["policy"] == "oracle")
    assert {x["label"] for x in labels} == {"good", "bad"}
    rec = Recordings.from_traces(base / "traces.jsonl")
    call = next(c for t in traces for c in t["tool_calls"] if not c["write"])
    found, output = rec.lookup(call["name"], call["args"])
    assert found
    assert output == (json.loads(call["output"]) if call["output"][:1] in "[{" else call["output"])
