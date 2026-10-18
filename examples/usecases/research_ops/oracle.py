"""Scripted policies: the oracle reads the gold source, answers, and files the gold ticket (if any).
The broken policy answers correctly but files a ticket on every request with priority "normal" -
the over-eager / under-prioritising operator the grader must catch."""
from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage

from usecases._common import ScriptedChatModel, call


def _policy(task: dict, broken: bool = False):
    ticket = task["ticket"]
    if broken:
        ticket = {"queue": (ticket or {}).get("queue", "compliance"), "doc_id": task["source"], "priority": "normal"}

    def policy(node, messages, tools):
        called = any(isinstance(m, ToolMessage) for m in messages)
        if node == "researcher":
            if not called:
                return call("read_doc", {"doc_id": task["source"]})
            return AIMessage(content=f"The provision says so.\nANSWER: {task['answer']}\nSOURCE: {task['source']}")
        if node == "operations":
            if ticket and not called:
                return call("file_ticket", {**ticket, "summary": task["input"][:120]})
            return AIMessage(content="DONE" if ticket else "NO_TICKET")
        return AIMessage(content="")
    return policy


def oracle_model(task: dict) -> ScriptedChatModel:
    return ScriptedChatModel(policy=_policy(task))


def broken_model(task: dict) -> ScriptedChatModel:
    return ScriptedChatModel(policy=_policy(task, broken=True))
