"""Replay/capture/dev tool wrappers inside a real (scripted, offline) LangGraph graph."""
import json

import pytest

from meta_harness.integrations.langgraph.tools import (Recordings, ReplayMiss, ToolSession, canonical_args,
                                                       classifier_from_settings, classify, heuristic_kind)

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, MessagesState, StateGraph  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from meta_harness.integrations.langgraph import ProductionEnvironmentError, wrap_tools  # noqa: E402


class Calls:
    def __init__(self):
        self.lookup = []
        self.refund = []


@pytest.fixture
def real():
    calls = Calls()

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order's status by id."""
        calls.lookup.append(order_id)
        return f"LIVE status of {order_id}"

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Refund money to the customer."""
        calls.refund.append((order_id, amount))
        return f"LIVE refund {order_id}"

    return calls, [lookup_order, issue_refund]


CLASSES = {"lookup_order": "read", "issue_refund": "write"}


def run_graph(tools, tool_calls):
    """Planner node scripted to request `tool_calls`, then a ToolNode executes them."""
    def planner(state):
        return {"messages": [AIMessage(content="", tool_calls=[
            {"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(tool_calls)])]}

    graph = StateGraph(MessagesState)
    graph.add_node("planner", planner)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "tools")
    graph.add_edge("tools", END)
    out = graph.compile().invoke({"messages": [HumanMessage(content="refund order 7")]})
    return [m for m in out["messages"] if isinstance(m, ToolMessage)]


def recordings():
    return Recordings.from_traces([{"input": "x", "tool_calls": [
        {"name": "lookup_order", "args": {"order_id": "7"}, "output": "shipped (recorded)"},
        {"name": "issue_refund", "args": {"order_id": "1", "amount": 2}, "output": "refund queued #991"}]}])


class FakeTrace:
    def __init__(self):
        self.events = []

    def event(self, name, payload=None):
        self.events.append((name, payload))


def test_wrapper_keeps_name_description_schema(real):
    _, tools = real
    wrapped = wrap_tools(tools, "replay", recordings(), CLASSES)
    for orig, new in zip(tools, wrapped):
        assert (new.name, new.description, new.args_schema) == (orig.name, orig.description, orig.args_schema)
    assert [w.kind for w in wrapped] == ["read", "write"]


def test_replay_answers_reads_and_captures_writes_without_executing(real):
    calls, tools = real
    trace = FakeTrace()
    wrapped = wrap_tools(tools, "replay", recordings(), CLASSES, trace=trace)
    msgs = run_graph(wrapped, [("lookup_order", {"order_id": "7"}),
                               ("issue_refund", {"order_id": "7", "amount": 30.0})])
    assert [m.content for m in msgs] == ["shipped (recorded)", "refund queued #991"]
    assert calls.lookup == [] and calls.refund == []
    session = wrapped[0].session
    assert [(c["name"], c["args"]) for c in session.captured] == [("issue_refund", {"order_id": "7", "amount": 30.0})]
    assert session.captured[0]["timestamp"]
    assert not session.replay_miss
    events = [p for n, p in trace.events if n == "tool_call"]
    assert [(e["tool"], e["mode"], e["kind"], e["outcome"]) for e in events] == [
        ("lookup_order", "replay", "read", "hit"), ("issue_refund", "replay", "write", "captured")]


def test_write_without_recorded_ack_returns_configured_ack(real):
    calls, tools = real
    wrapped = wrap_tools(tools, "replay", Recordings(), CLASSES, write_ack="done", on_missing="empty")
    msgs = run_graph(wrapped, [("issue_refund", {"order_id": "7", "amount": 1.0})])
    assert msgs[0].content == "done" and calls.refund == []


def test_replay_miss_flags_run_and_errors(real):
    calls, tools = real
    trace = FakeTrace()
    wrapped = wrap_tools(tools, "replay", recordings(), CLASSES, trace=trace)
    with pytest.raises(ReplayMiss, match="no recorded output"):  # surfaces through the graph
        run_graph(wrapped, [("lookup_order", {"order_id": "8"})])
    assert calls.lookup == []
    assert wrapped[0].session.replay_miss
    assert ("replay_miss", {"flag": "replay_miss"}) in trace.events
    with pytest.raises(ReplayMiss):
        wrapped[0].invoke({"order_id": "9"})


def test_replay_miss_empty_mode(real):
    _, tools = real
    wrapped = wrap_tools(tools, "replay", recordings(), CLASSES, on_missing="empty")
    assert wrapped[0].invoke({"order_id": "8"}) == ""
    assert wrapped[0].session.replay_miss
    assert wrapped[0].session.events[-1]["outcome"] == "miss"


def test_dev_mode_executes_for_real_when_env_is_safe(real):
    calls, tools = real
    wrapped = wrap_tools(tools, "dev", recordings(), CLASSES, env={"APP_ENV": "staging"}, production_markers=[])
    msgs = run_graph(wrapped, [("lookup_order", {"order_id": "7"}),
                               ("issue_refund", {"order_id": "7", "amount": 3.0})])
    assert [m.content for m in msgs] == ["LIVE status of 7", "LIVE refund 7"]
    assert calls.lookup == ["7"] and calls.refund == [("7", 3.0)]
    assert [e["outcome"] for e in wrapped[0].session.events] == ["executed", "executed"]


def test_dev_mode_refuses_production(real):
    calls, tools = real
    with pytest.raises(ProductionEnvironmentError):
        wrap_tools(tools, "dev", None, CLASSES, env={"NODE_ENV": "production"}, production_markers=[])
    env = {"APP_ENV": "dev"}
    wrapped = wrap_tools(tools, "dev", None, CLASSES, env=env, production_markers=["db.prod.acme"])
    env["DATABASE_URL"] = "postgres://u:p@db.prod.acme:5432/x"  # environment flips after wrapping
    with pytest.raises(ProductionEnvironmentError):
        wrapped[1].invoke({"order_id": "7", "amount": 1.0})
    assert calls.refund == []


def _drive(coro):
    """Run a coroutine that never suspends (the conftest forbids the sockets an event loop opens)."""
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError("coroutine suspended")


def test_async_replay(real):
    calls, tools = real
    wrapped = wrap_tools(tools, "replay", recordings(), CLASSES)
    assert _drive(wrapped[0].coroutine(order_id="7")) == "shipped (recorded)"
    assert _drive(wrapped[1].coroutine(order_id="7", amount=1.0)) == "refund queued #991"
    assert calls.lookup == [] and calls.refund == []


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        ToolSession("live")


# ------------------------------------------------------------------ classification

class T:
    def __init__(self, name, description=""):
        self.name, self.description = name, description


@pytest.mark.parametrize("name,description,kind", [
    ("search_docs", "Search the knowledge base", "read"),
    ("get_weather", "Current weather for a city", "read"),
    ("listInvoices", "List invoices", "read"),
    ("send_email", "", "write"),
    ("get_and_update_ticket", "", "write"),
    ("lookup_user", "Looks up a user then emails them", "write"),
    ("createIssue", "", "write"),
    ("frobnicate", "Does a thing", "write"),
    ("get_settings", "Reads the settings", "read"),
    ("pay_invoice", "", "write"),
])
def test_heuristic(name, description, kind):
    assert heuristic_kind(name, description) == kind


def test_explicit_classifier_wins_and_validates():
    assert classify(T("send_digest"), {"send_digest": "read"}) == "read"
    assert classify(T("search"), {"other": "read"}) == "read"
    assert classify(T("search"), lambda t: "write") == "write"
    assert classify(T("search"), lambda t: None) == "read"
    with pytest.raises(ValueError):
        classify(T("x"), {"x": "maybe"})


def test_classifier_from_settings(tmp_path):
    path = tmp_path / "meta-harness.toml"
    path.write_text('[tools]\nsend_digest = "read"\nfoo = "write"\nbar = "maybe"\n', encoding="utf-8")
    assert classifier_from_settings(path) == {"send_digest": "read", "foo": "write"}
    assert classifier_from_settings(tmp_path / "missing.toml") == {}


# ------------------------------------------------------------------ recordings

def test_canonical_args_ignores_order_and_json_strings():
    assert canonical_args({"b": 1, "a": 2}) == canonical_args('{"a": 2, "b": 1}') == '{"a":2,"b":1}'
    assert canonical_args({"a": 1}) != canonical_args({"a": "1"})


def test_recordings_from_langsmith_langfuse_otel_and_files(tmp_path):
    langsmith = {"run_type": "chain", "name": "agent", "child_runs": [
        {"run_type": "tool", "name": "lookup_order", "inputs": {"input": json.dumps({"order_id": "1"})},
         "outputs": {"output": {"type": "tool", "content": "LS shipped"}}},
        {"run_type": "llm", "name": "lookup_order", "inputs": {"order_id": "2"}, "outputs": {"output": "no"}}]}
    langfuse = {"data": [{"id": "t", "observations": [
        {"type": "TOOL", "name": "lookup_order", "input": {"order_id": "3"}, "output": "LF shipped"},
        {"type": "GENERATION", "name": "lookup_order", "input": {"order_id": "4"}, "output": "no"}]}]}
    otel = {"resourceSpans": [{"scopeSpans": [{"spans": [{"spanId": "s", "attributes": [
        {"key": "gen_ai.tool.name", "value": {"stringValue": "lookup_order"}},
        {"key": "gen_ai.tool.call.arguments", "value": {"stringValue": '{"order_id":"5"}'}},
        {"key": "gen_ai.tool.call.result", "value": {"stringValue": "OT shipped"}}]}]}]}]}
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"input": "q", "tool_calls": [
        {"name": "lookup_order", "arguments": '{"order_id": "6"}', "result": "JL shipped"},
        {"name": "lookup_order", "args": {"order_id": "99"}}]}) + "\n", encoding="utf-8")
    lf_file = tmp_path / "langfuse.json"
    lf_file.write_text(json.dumps(langfuse), encoding="utf-8")
    rec = Recordings.from_traces([langsmith, otel, str(tasks), lf_file])
    expected = {"1": "LS shipped", "3": "LF shipped", "5": "OT shipped", "6": "JL shipped"}
    for order_id, output in expected.items():
        assert rec.lookup("lookup_order", {"order_id": order_id}) == (True, output)
    for order_id in ("2", "4", "99"):
        assert rec.lookup("lookup_order", {"order_id": order_id}) == (False, None)
    assert len(rec) == 4
    assert Recordings.from_traces(langsmith).lookup("lookup_order", {"order_id": "1"})[0]


def test_async_dev_awaits_the_real_tool():
    from langchain_core.tools import StructuredTool
    ran = []

    class LiveTool(StructuredTool):
        async def ainvoke(self, input, config=None, **kwargs):  # no event loop: conftest bans its sockets
            ran.append(input)
            return "LIVE page"

    def fetch_page(url: str) -> str:
        """Fetch a page."""
        raise AssertionError("sync path must not be used")

    live = LiveTool.from_function(fetch_page)
    wrapped = wrap_tools([live], "dev", None, None, env={}, production_markers=[])
    assert _drive(wrapped[0].coroutine(url="u")) == "LIVE page" and ran == [{"url": "u"}]
