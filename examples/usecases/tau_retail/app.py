"""Retail customer-service graph in the style of tau-bench retail.

    START -> triage -> agent <-> customer -> END

* triage (LLM, inline prompt): labels the request so the agent gets a short plan hint.
* agent (LLM + tools, system prompt from `settings` + store policy): reads and writes the order DB.
* customer (scripted, deterministic): confirms consequential actions and answers follow-ups with
  the facts the task gives it (payment method, cancel reason); ends when the agent stops asking.

The backend is three JSON files copied into a fresh sandbox per task. `run(task, llm)` is the entry;
`build_graph(llm, tools)` is the seam an integration swaps models/tools through.
"""
from __future__ import annotations

import operator
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from usecases._common import Recorder, sandbox, text_of, tool_loop  # noqa: E402
from usecases.tau_retail import settings  # noqa: E402
from usecases.tau_retail.tools import DB_FILES, TOOL_KINDS, RetailDB, make_tools  # noqa: E402

DATA = HERE / "data"
TRIAGE_PROMPT = ("You route retail support requests. Read the customer's message and reply with a "
                 "comma-separated list of the actions it needs, chosen from: cancel, modify, address, "
                 "return, exchange, info. Reply with the labels only.")
CONFIRM_WORDS = ("confirm", "proceed", "yes", "shall i", "would you like", "do you want", "?")


class State(TypedDict, total=False):
    messages: Annotated[list, operator.add]
    intent: str
    user_turns: int
    user_facts: str
    done: bool


def policy_text() -> str:
    return (DATA / "policy.md").read_text(encoding="utf-8")


def build_graph(llm: Any, tools: list, cfg: Any = settings):
    system = cfg.AGENT_SYSTEM_PROMPT.format(policy=policy_text()) + "\n" + "\n".join(f"- {r}" for r in cfg.AGENT_RULES)

    def triage(state: State) -> dict:
        first = next(m for m in state["messages"] if isinstance(m, HumanMessage))
        reply = llm.with_config(tags=["node:triage"]).invoke([SystemMessage(TRIAGE_PROMPT), first])
        return {"intent": text_of(reply).strip()[:200]}

    def agent(state: State) -> dict:
        hint = f"\nTriage labels for this conversation: {state['intent']}" if state.get("intent") else ""
        msgs = [SystemMessage(system + hint)] + state["messages"]
        return {"messages": tool_loop(llm, tools, msgs, "agent", cfg.MAX_TOOL_STEPS)}

    def customer(state: State) -> dict:
        last = text_of(state["messages"][-1]).lower()
        turns = state.get("user_turns", 0)
        if turns >= cfg.MAX_USER_TURNS or not any(w in last for w in CONFIRM_WORDS):
            return {"done": True}
        facts = state.get("user_facts", "")
        reply = "Yes, I confirm, please go ahead. " + (facts or "That is everything I need.")
        return {"messages": [HumanMessage(reply)], "user_turns": turns + 1}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("customer", customer)
    if cfg.TRIAGE_ENABLED:
        g.add_node("triage", triage)
        g.add_edge(START, "triage")
        g.add_edge("triage", "agent")
    else:
        g.add_edge(START, "agent")
    g.add_edge("agent", "customer")
    g.add_conditional_edges("customer", lambda s: END if s.get("done") else "agent")
    return g.compile()


def run(task: dict, llm: Any = None, root: str | Path | None = None, keep: bool = False, callbacks=None) -> dict:
    """Run one task in a fresh sandbox. Returns the trace row (input, output, tool_calls, final state)."""
    if llm is None:
        from usecases.tau_retail.oracle import oracle_model
        llm = oracle_model(task)
    box = sandbox([DATA / f for f in DB_FILES], root, prefix="tau-")
    recorder = Recorder()
    graph = build_graph(llm, make_tools(box, recorder))
    config = {"recursion_limit": 60}
    if callbacks:
        config["callbacks"] = callbacks
    error = ""
    try:
        state = graph.invoke({"messages": [HumanMessage(task["input"])], "user_facts": task.get("user_facts", ""),
                              "user_turns": 0}, config)
    except Exception as exc:  # a crashed run is a failed run, not a crashed eval
        state, error = {"messages": []}, f"{type(exc).__name__}: {exc}"
    agent_text = [text_of(m) for m in state["messages"] if isinstance(m, AIMessage) and text_of(m)]
    db, base = RetailDB(box), RetailDB(DATA)
    changed = {k: v for k, v in db.orders.items() if base.orders.get(k) != v}
    changed.update({k: v for k, v in db.users.items() if base.users.get(k) != v})
    row = {"id": task["id"], "input": task["input"], "output": agent_text[-1] if agent_text else "",
           "agent_messages": agent_text, "intent": state.get("intent", ""),
           "tool_calls": [{**c, "write": TOOL_KINDS.get(c["name"]) == "write"} for c in recorder.calls],
           "state_digest": db.state_digest(), "changed": changed, "error": error, "sandbox": str(box)}
    if not keep:
        shutil.rmtree(box, ignore_errors=True)
    return row
