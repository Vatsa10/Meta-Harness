"""Research desk graph: retriever -> researcher -> operations.

* retriever (deterministic): BM25 `search_docs` on the request, top RETRIEVE_K into the context.
* researcher (LLM + search_docs/read_doc): answers with "ANSWER:" and "SOURCE:" lines.
* operations (LLM + list_tickets/file_ticket, policy in settings + inline instructions): files a
  ticket when the request needs follow-up, otherwise replies NO_TICKET.

Tickets are written to a JSON file in a fresh sandbox per task.
"""
from __future__ import annotations

import json
import operator
import re
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from usecases._common import Recorder, read_jsonl, sandbox, text_of, tool_loop  # noqa: E402
from usecases.research_ops import settings  # noqa: E402
from usecases.research_ops.tools import TOOL_KINDS, Desk, make_tools  # noqa: E402

DATA = HERE / "data"
OPS_PROMPT = ("You are the operations agent of a civic research desk. You receive a request and the "
              "analyst's answer. Decide whether a follow-up ticket is required by the policy below; if so, "
              "call file_ticket once, then reply DONE. If not, reply NO_TICKET without calling tools.\n\n")
_ANSWER = re.compile(r"ANSWER:\s*(.+)", re.I)
_SOURCE = re.compile(r"SOURCE:\s*([\w-]+)", re.I)


class State(TypedDict, total=False):
    request: str
    context: str
    answer: str
    source: str
    research: str
    ops: str
    log: Annotated[list, operator.add]


def corpus() -> list[dict]:
    return read_jsonl(DATA / "corpus.jsonl")


def build_graph(llm: Any, tools: list, cfg: Any = settings):
    by_name = {t.name: t for t in tools}

    def retriever(state: State) -> dict:
        hits = json.loads(by_name["search_docs"].invoke({"query": state["request"], "top_k": cfg.RETRIEVE_K}))
        ctx = "\n".join(f"[{h['doc_id']}] {h['title']}: {h['snippet']}" for h in hits)
        return {"context": ctx, "log": [f"retrieved: {[h['doc_id'] for h in hits]}"]}

    def researcher(state: State) -> dict:
        msg = f"Request: {state['request']}\n\nRetrieved documents:\n{state.get('context', '')}"
        tools_r = [by_name["search_docs"], by_name["read_doc"]]
        new = tool_loop(llm, tools_r, [SystemMessage(cfg.RESEARCHER_PROMPT), HumanMessage(msg)], "researcher",
                        cfg.MAX_RESEARCH_STEPS)
        text = text_of(new[-1])
        a, s = _ANSWER.search(text), _SOURCE.search(text)
        return {"research": text, "answer": a.group(1).strip() if a else text.strip()[-200:],
                "source": s.group(1) if s else "", "log": [f"answer: {text[-200:]}"]}

    def operations(state: State) -> dict:
        msg = (f"Request: {state['request']}\nAnalyst answer: {state.get('answer', '')}\n"
               f"Governing provision: {state.get('source', '')}")
        new = tool_loop(llm, [by_name["list_tickets"], by_name["file_ticket"]],
                        [SystemMessage(OPS_PROMPT + cfg.TICKET_POLICY), HumanMessage(msg)], "operations",
                        cfg.MAX_ACTION_STEPS)
        return {"ops": text_of(new[-1]), "log": [f"ops: {text_of(new[-1])[:200]}"]}

    g = StateGraph(State)
    g.add_node("retriever", retriever)
    g.add_node("researcher", researcher)
    g.add_node("operations", operations)
    g.add_edge(START, "retriever")
    g.add_edge("retriever", "researcher")
    g.add_edge("researcher", "operations")
    g.add_edge("operations", END)
    return g.compile()


def run(task: dict, llm: Any = None, root: str | Path | None = None, keep: bool = False, callbacks=None) -> dict:
    if llm is None:
        from usecases.research_ops.oracle import oracle_model
        llm = oracle_model(task)
    box = sandbox([], root, prefix="ops-")
    desk, recorder = Desk(corpus(), box), Recorder()
    config = {"recursion_limit": 30, **({"callbacks": callbacks} if callbacks else {})}
    error = ""
    try:
        state = build_graph(llm, make_tools(desk, recorder, settings.RETRIEVE_K)).invoke({"request": task["input"]}, config)
    except Exception as exc:
        state, error = {}, f"{type(exc).__name__}: {exc}"
    row = {"id": task["id"], "input": task["input"], "output": state.get("research", ""), "answer": state.get("answer", ""),
           "source": state.get("source", ""), "tickets": desk.tickets(), "log": state.get("log", []), "error": error,
           "tool_calls": [{**c, "write": TOOL_KINDS.get(c["name"]) == "write"} for c in recorder.calls],
           "sandbox": str(box)}
    if not keep:
        shutil.rmtree(box, ignore_errors=True)
    return row
