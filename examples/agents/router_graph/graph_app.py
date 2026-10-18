"""A LangGraph-style app written in plain Python (no framework import).

The graph: router -> (math | chat) -> END. The router scores how "mathy" the question is and
sends it to the math node when the score reaches `router.math_threshold`. Every node's system
prompt, the few-shot examples, the retrieval k and the memory policy come from the config, which
is what Meta-Harness searches. `run_graph(task, config, model)` is the adapter entrypoint.
"""

from __future__ import annotations

import re
from typing import Any, Callable

KNOWLEDGE = [
    "Paris is the capital of France.",
    "Hamlet was written by William Shakespeare.",
    "Water boils at 100 degrees Celsius at sea level.",
]
_MATH_TOKEN = re.compile(r"^(\d+(\.\d+)?|[+\-*/=])$")


def math_score(question: str) -> float:
    tokens = question.replace("?", " ").split()
    return sum(1 for t in tokens if _MATH_TOKEN.match(t)) / len(tokens) if tokens else 0.0


def retrieve(question: str, k: int) -> list[str]:
    words = set(question.lower().split())
    ranked = sorted(KNOWLEDGE, key=lambda doc: -len(words & set(doc.lower().rstrip(".").split())))
    return ranked[:k]


def router(state: dict, config: dict) -> str:
    state["math_score"] = math_score(state["question"])
    return "math" if state["math_score"] >= config["router"]["math_threshold"] else "chat"


def math_node(state: dict, config: dict, model: Callable[[str], str]) -> dict:
    node = config["nodes"]["math"]
    shots = "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in config["few_shot"])
    reply = model(f"{node['system_prompt']}\n\n{shots}Question: {state['question']}")
    return {**state, "answer": str(reply).strip()}


def chat_node(state: dict, config: dict, model: Callable[[str], str]) -> dict:
    node = config["nodes"]["chat"]
    context = "\n".join(retrieve(state["question"], config["retrieval_k"]))
    history = state.get("history", "") if config["memory"]["policy"] == "last_turn" else ""
    reply = model(f"{node['system_prompt']}\n\nContext:\n{context}\n{history}\nQuestion: {state['question']}")
    return {**state, "answer": str(reply).strip()}


NODES = {"math": math_node, "chat": chat_node}


def run_graph(task: dict, config: dict, model: Any) -> dict:
    state = {"question": str(task.get("input", ""))}
    route = router(state, config)
    state = NODES[route](state, config, model)
    return {"output": state["answer"], "route": route, "math_score": round(state["math_score"], 3)}
