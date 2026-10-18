"""Deterministic scripted policies for offline runs.

`oracle_model(task)` replays the task's gold actions (asking for confirmation before the first write,
then stating the required outputs), so its run is a gold trace. `broken_model(task)` is the same but
skips the last write - a realistic "forgot one item" failure the grader must catch.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from usecases._common import ScriptedChatModel, call
from usecases.tau_retail.tools import WRITE_TOOLS


def _policy(task: dict, drop_last_write: bool = False):
    gold = list(task["gold_actions"])
    if drop_last_write:
        last = max(i for i, g in enumerate(gold) if g["name"] in WRITE_TOOLS)
        gold = gold[:last] + gold[last + 1:]

    def policy(node, messages, tools):
        if node == "triage":
            return AIMessage(content=",".join(sorted({g["name"].split("_")[0] for g in gold if g["name"] in WRITE_TOOLS})) or "info")
        done = sum(isinstance(m, ToolMessage) for m in messages)
        humans = sum(isinstance(m, HumanMessage) for m in messages)
        if done < len(gold):
            nxt = gold[done]
            if nxt["name"] in WRITE_TOOLS and humans == 1:
                return AIMessage(content=f"I am ready to run {nxt['name']} with {nxt['args']}. Do you confirm (yes)?")
            return call(nxt["name"], nxt["args"], done)
        outputs = ", ".join(task.get("required_outputs") or [])
        return AIMessage(content="All done. " + (f"The answer: {outputs}." if outputs else "Thank you for contacting us."))

    return policy


def oracle_model(task: dict) -> ScriptedChatModel:
    return ScriptedChatModel(policy=_policy(task))


def broken_model(task: dict) -> ScriptedChatModel:
    return ScriptedChatModel(policy=_policy(task, drop_last_write=True))
