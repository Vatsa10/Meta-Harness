"""SQL analyst graph over Chinook: planner -> writer -> executor -> verifier (-> writer once more).

* planner (LLM, prompt in settings): a plan, no SQL.
* writer (LLM + read tools + scratch write tools): explores the schema, replies with one sql block.
* executor (deterministic): saves the writer's query as scratch.<RESULT_TABLE> through the
  `save_result` write tool (recorded like any tool call).
* verifier (LLM, inline prompt): sees the question, query and preview; APPROVE or REVISE: <why>.

The database and scratch schema are copied into a fresh sandbox per task; writes outside `scratch`
are refused by SQLite's authorizer.
"""
from __future__ import annotations

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

from usecases._common import Recorder, sandbox, text_of, tool_loop  # noqa: E402
from usecases.sql_analyst import settings  # noqa: E402
from usecases.sql_analyst.tools import DB_FILE, TOOL_KINDS, AnalystDB, make_tools  # noqa: E402

DATA = HERE / "data"
VERIFIER_PROMPT = ("You review a SQL answer. Check that the query answers the question, returns exactly the "
                   "requested columns, and that the preview looks plausible. Reply 'APPROVE' or "
                   "'REVISE: <what to fix>'.")
_SQL = re.compile(r"```(?:sql)?\s*(.*?)```", re.S | re.I)


class State(TypedDict, total=False):
    question: str
    plan: str
    sql: str
    saved: str
    verdict: str
    revisions: int
    log: Annotated[list, operator.add]


def extract_sql(text: str) -> str:
    m = _SQL.search(text or "")
    return (m.group(1) if m else text or "").strip().rstrip(";")


def build_graph(llm: Any, tools: list, cfg: Any = settings):
    by_name = {t.name: t for t in tools}

    def planner(state: State) -> dict:
        reply = llm.with_config(tags=["node:planner"]).invoke(
            [SystemMessage(cfg.PLANNER_PROMPT), HumanMessage(state["question"])])
        return {"plan": text_of(reply), "log": [f"plan: {text_of(reply)[:300]}"]}

    def writer(state: State) -> dict:
        parts = [f"Question: {state['question']}", f"Plan:\n{state.get('plan', '')}"]
        if cfg.SCHEMA_HINT:
            parts.append(f"Tables: {by_name['list_tables'].invoke({})}")
        if state.get("verdict", "").upper().startswith("REVISE"):
            parts.append(f"Your previous query:\n{state.get('sql', '')}\nReviewer feedback: {state['verdict']}")
        new = tool_loop(llm, tools, [SystemMessage(cfg.WRITER_PROMPT), HumanMessage("\n\n".join(parts))],
                        "writer", cfg.MAX_WRITER_STEPS)
        sql = extract_sql(text_of(new[-1]))
        return {"sql": sql, "log": [f"sql: {sql[:300]}"]}

    def executor(state: State) -> dict:
        out = by_name["save_result"].invoke({"table": cfg.RESULT_TABLE, "sql": state.get("sql") or "SELECT NULL WHERE 0"})
        return {"saved": str(out), "log": [f"saved: {str(out)[:300]}"]}

    def verifier(state: State) -> dict:
        msg = (f"Question: {state['question']}\n\nQuery:\n{state.get('sql', '')}\n\n"
               f"Execution result: {state.get('saved', '')[:1500]}")
        reply = llm.with_config(tags=["node:verifier"]).invoke([SystemMessage(VERIFIER_PROMPT), HumanMessage(msg)])
        return {"verdict": text_of(reply).strip(), "revisions": state.get("revisions", 0) + 1,
                "log": [f"verdict: {text_of(reply)[:200]}"]}

    def after_verify(state: State) -> str:
        if state["verdict"].upper().startswith("REVISE") and state["revisions"] <= cfg.MAX_REVISIONS:
            return "writer"
        return END

    g = StateGraph(State)
    for name, fn in (("planner", planner), ("writer", writer), ("executor", executor), ("verifier", verifier)):
        g.add_node(name, fn)
    g.add_edge(START, "planner")
    g.add_edge("planner", "writer")
    g.add_edge("writer", "executor")
    g.add_edge("executor", "verifier")
    g.add_conditional_edges("verifier", after_verify)
    return g.compile()


def run(task: dict, llm: Any = None, root: str | Path | None = None, keep: bool = False, callbacks=None) -> dict:
    """Run one question in a fresh sandbox; the trace row carries the saved scratch result."""
    if llm is None:
        from usecases.sql_analyst.oracle import oracle_model
        llm = oracle_model(task)
    box = sandbox([DATA / DB_FILE], root, prefix="sql-")
    db, recorder = AnalystDB(box), Recorder()
    config = {"recursion_limit": 40, **({"callbacks": callbacks} if callbacks else {})}
    error = ""
    try:
        state = build_graph(llm, make_tools(db, recorder)).invoke({"question": task["input"], "revisions": 0}, config)
    except Exception as exc:  # a crashed run is a failed run
        state, error = {}, f"{type(exc).__name__}: {exc}"
    rows = db.scratch_rows(settings.RESULT_TABLE)
    db.close()
    row = {"id": task["id"], "input": task["input"], "output": state.get("sql", ""), "verdict": state.get("verdict", ""),
           "result_rows": rows, "log": state.get("log", []), "error": error,
           "tool_calls": [{**c, "write": TOOL_KINDS.get(c["name"]) == "write"} for c in recorder.calls],
           "sandbox": str(box)}
    if not keep:
        shutil.rmtree(box, ignore_errors=True)
    return row
