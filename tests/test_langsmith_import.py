import io
import json
import urllib.error
from pathlib import Path

import pytest

from meta_harness.integrations import langsmith as ls

FIX = Path(__file__).parent / "fixtures" / "langsmith"


def _by_trace(tasks):
    return {t["source_trace_id"]: t for t in tasks}


def test_export_becomes_failure_first_tasks_with_tool_calls():
    tasks = ls.import_file(FIX / "runs.json")
    assert [t["source_trace_id"] for t in tasks] == ["t3", "t1", "t2"]   # error, negative, ok; t4 dup of t2
    t1, t2, t3 = (_by_trace(tasks)[k] for k in ("t1", "t2", "t3"))
    assert set(t1) >= {"id", "input", "reference_output", "criteria", "source_trace_id", "failure",
                       "tool_calls", "node_outputs", "project", "metadata"}
    assert t1["input"] == "Refund order 8812 please, email jane.doe@example.com"   # system msg dropped
    assert t1["reference_output"] == "I refunded order 8812 in full."
    assert t1["failure"] == "negative" and "wrong amount refunded" in t1["criteria"]
    assert t1["tool_calls"] == [
        {"name": "lookup_order", "args": {"order_id": "8812"}, "output": {"total": 60.0}},
        {"name": "issue_refund", "args": {"order_id": "8812", "amount": 120.0}, "output": "refund ok"},
    ]
    assert t1["node_outputs"] == {"planner": '{"plan": "lookup then refund"}', "ChatOpenAI": "lookup then refund"}
    assert t1["metadata"] == {"langgraph_step": 3, "app": "support"} and t1["project"] == "support-prod"
    assert t2["failure"] == "ok" and t2["tool_calls"][0]["name"] == "search_kb"
    assert t3["failure"] == "error"
    assert "cancel_subscription: billing API 503" in t3["criteria"]
    assert t3["tool_calls"] == [{"name": "cancel_subscription", "args": {"user": "u1"}, "output": None,
                                 "error": "billing API 503"}]


def test_jsonl_export_flat_children_and_pii_and_limit():
    rows = json.loads((FIX / "runs.json").read_text(encoding="utf-8"))
    flat = ls.parse(rows)
    text = "\n".join(json.dumps(r) for r in flat)
    tasks = ls.to_tasks(ls.parse_text(text), pii=True, limit=2)
    assert [t["source_trace_id"] for t in tasks] == ["t3", "t1"]
    assert "[email]" in tasks[1]["input"] and "jane.doe" not in json.dumps(tasks)
    assert [c["name"] for c in tasks[1]["tool_calls"]] == ["lookup_order", "issue_refund"]


def test_secrets_redacted_in_tool_args(monkeypatch):
    monkeypatch.setenv("SHOP_API_KEY", "supersecretvalue123")
    runs = [{"id": "a", "inputs": {"input": "hi"}, "outputs": {"output": "yo"}},
            {"id": "b", "parent_run_id": "a", "run_type": "tool", "name": "t",
             "inputs": {"auth": "supersecretvalue123"}, "outputs": {"output": "ok"}}]
    (task,) = ls.to_tasks(runs)
    assert task["tool_calls"][0]["args"] == {"auth": "[redacted]"}


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_server(pages_root, children):
    calls = []

    def opener(request, timeout=None):
        body = json.loads(request.data) if request.data else None
        calls.append((request.get_method(), request.full_url, body, dict(request.header_items())))
        if "/sessions" in request.full_url:
            return _Resp(json.dumps([{"id": "sess-1", "name": "support-prod"}]).encode())
        if body.get("is_root"):
            page = pages_root[body.get("cursor") or "start"]
            return _Resp(json.dumps(page).encode())
        return _Resp(json.dumps({"runs": children[body["trace"]], "cursors": {"next": None}}).encode())
    return opener, calls


def test_live_pull_paginates_and_attaches_children():
    pages = {"start": {"runs": [{"id": "r1", "trace_id": "t1", "inputs": {"input": "q1"},
                                 "outputs": {"output": "a1"}}], "cursors": {"next": "c2"}},
             "c2": {"runs": [{"id": "r2", "trace_id": "t2", "inputs": {"input": "q2"},
                              "outputs": {"output": "a2"}}], "cursors": {"next": None}}}
    children = {"t1": [{"id": "r1", "trace_id": "t1"},
                       {"id": "c1", "parent_run_id": "r1", "run_type": "tool", "name": "send_email",
                        "inputs": {"to": "x"}, "outputs": {"output": "sent"}}],
                "t2": []}
    opener, calls = _fake_server(pages, children)
    runs = ls.fetch_runs("support-prod", limit=5, api_key="k-123456789", page_size=1,
                         endpoint="https://ls.example/", opener=opener)
    root_queries = [c for c in calls if c[2] and c[2].get("is_root")]
    assert [q[2].get("cursor") for q in root_queries] == [None, "c2"]
    assert root_queries[0][2]["session"] == ["sess-1"] and root_queries[0][2]["limit"] == 1
    assert all(c[1].startswith("https://ls.example/api/v1/") for c in calls)
    headers = calls[0][3]
    assert headers["X-api-key"] == "k-123456789" and headers["User-agent"].startswith("meta-harness")
    tasks = ls.to_tasks(runs)
    assert {t["input"] for t in tasks} == {"q1", "q2"}
    assert _by_trace(tasks)["t1"]["tool_calls"][0]["name"] == "send_email"
    assert all(t["project"] == "support-prod" for t in tasks)


def test_live_pull_stops_at_limit():
    pages = {"start": {"runs": [{"id": "r1", "inputs": {"input": "q"}}], "cursors": {"next": "c2"}},
             "c2": {"runs": [{"id": "r2", "inputs": {"input": "q2"}}], "cursors": {"next": "c3"}}}
    opener, calls = _fake_server(pages, {"r1": []})
    runs = ls.fetch_runs("support-prod", limit=1, api_key="k-123456789", page_size=1, opener=opener)
    assert [r["id"] for r in runs] == ["r1"]
    assert sum(1 for c in calls if c[2] and c[2].get("is_root")) == 1


def test_live_pull_errors_are_redacted_and_key_required(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    with pytest.raises(ls.LangSmithError, match="LANGSMITH_API_KEY"):
        ls.fetch_runs("p", opener=lambda *a, **k: None)
    key = "lsv2_secret_abcdef123"

    def opener(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "no", {}, io.BytesIO(f"bad key {key}".encode()))
    with pytest.raises(ls.LangSmithError) as info:
        ls.fetch_runs("p", api_key=key, opener=opener)
    assert "401" in str(info.value) and key not in str(info.value)


def test_comment_only_feedback_marks_negative():
    runs = [{"id": "a", "inputs": {"input": "hi"}, "outputs": {"output": "yo"},
             "feedback": [{"key": "note", "comment": "this answer is wrong"}]},
            {"id": "b", "inputs": {"input": "other"}, "outputs": {"output": "fine"},
             "feedback": [{"key": "note", "comment": "great"}]}]
    kinds = {t["source_trace_id"]: t["failure"] for t in ls.to_tasks(runs)}
    assert kinds == {"a": "negative", "b": "ok"}
