"""An OpenAI-Agents-SDK-style app written in plain Python (no framework import).

One agent with instructions and two function tools. Each turn the model sees the instructions,
the tool list (name: description) and the transcript, and replies `CALL <tool> <arg>` or
`FINAL <answer>`; the loop stops at `max_turns`. The instructions, the tool descriptions,
`max_turns` and `tool_choice` come from the config, which is what Meta-Harness searches.
`run_agent(task, config, model)` is the adapter entrypoint.
"""

from __future__ import annotations

from typing import Any, Callable

ORDERS = {"17": "shipped", "23": "delivered", "42": "processing"}


def lookup_order(order_id: str) -> str:
    return ORDERS.get(order_id.strip(" #?"), "unknown order")


def refund_order(order_id: str) -> str:
    return "refund issued" if order_id.strip(" #?") in ORDERS else "unknown order"


FUNCTIONS: dict[str, Callable[[str], str]] = {"lookup_order": lookup_order, "refund_order": refund_order}


def run_agent(task: dict, config: dict, model: Any) -> dict:
    tools = [t for t in config["tools"] if t["name"] in FUNCTIONS] if config["tool_choice"] != "none" else []
    listing = "\n".join(f"- {t['name']}: {t['description']}" for t in tools) or "(no tools)"
    transcript = f"User: {task.get('input', '')}"
    calls = []
    for _ in range(int(config["max_turns"])):
        reply = str(model(f"{config['instructions']}\n\nTools:\n{listing}\n\n{transcript}")).strip()
        if reply.startswith("CALL "):
            _, _, rest = reply.partition(" ")
            name, _, argument = rest.partition(" ")
            if name not in {t["name"] for t in tools}:
                transcript += f"\nAssistant: {reply}\nObservation: no such tool {name}"
                continue
            calls.append(name)
            transcript += f"\nAssistant: {reply}\nObservation: {FUNCTIONS[name](argument)}"
            continue
        answer = reply[len("FINAL "):] if reply.startswith("FINAL ") else reply
        return {"output": answer.strip(), "tool_calls": calls}
    return {"output": "", "tool_calls": calls, "stop_reason": "max_turns"}
