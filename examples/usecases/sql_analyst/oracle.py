"""Scripted policies: the oracle writes the gold SQL after one schema look-up; the broken policy
writes a plausible wrong query (drops the LIMIT, else the WHERE filter, else the HAVING)."""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, ToolMessage

from usecases._common import ScriptedChatModel, call


def _policy(sql: str):
    def policy(node, messages, tools):
        if node == "planner":
            return AIMessage(content="1. Find the relevant tables. 2. Join, filter, aggregate. 3. Return the asked columns.")
        if node == "writer":
            if not any(isinstance(m, ToolMessage) for m in messages):
                return call("describe_table", {"table": "Track"})
            return AIMessage(content=f"```sql\n{sql}\n```")
        return AIMessage(content="APPROVE")
    return policy


def break_sql(sql: str) -> str:
    for pat in (r"\s+LIMIT\s+\d+\s*$", r"\s+WHERE\s+.*?(?=\s+GROUP BY|\s*$)", r"\s+HAVING\s+.*?(?=\s+ORDER BY|\s*$)"):
        broken = re.sub(pat, "", sql, count=1, flags=re.I | re.S)
        if broken != sql:
            return broken
    return sql + " UNION ALL " + sql


def oracle_model(task: dict) -> ScriptedChatModel:
    return ScriptedChatModel(policy=_policy(task["gold_sql"]))


def broken_model(task: dict) -> ScriptedChatModel:
    return ScriptedChatModel(policy=_policy(break_sql(task["gold_sql"])))
