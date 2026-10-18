"""Shared plumbing for the real-world LangGraph use cases.

Everything here is small on purpose: an owner's app would have its own versions of these.

* `ScriptedChatModel` - an offline LangChain chat model driven by a Python policy
  ``policy(node, messages, tool_names) -> AIMessage``. Each graph node tags its model call with
  ``node:<name>`` (``llm.with_config(tags=[...])``), which a real model ignores and the scripted
  model reads, so one fake model can play every node of a multi-agent graph.
* `Recorder` + `recorded` - every tool call is captured with its arguments and output, which is the
  ``tool_calls: [{name, args, output}]`` shape `Recordings.from_traces` replays.
* `tool_loop` - the usual "call model, run requested tools, repeat" loop inside one node.
* `split_tasks` - a seeded train/search/test split.
* `live_llm` - DeepSeek / OpenAI chat models for live baselines (needs langchain-openai and keys).
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

Policy = Callable[[str, list, list], AIMessage]


# ------------------------------------------------------------------ offline chat model

class ScriptedChatModel(BaseChatModel):
    """Chat model whose replies come from `policy(node, messages, tool_names)`."""

    policy: Any

    @property
    def _llm_type(self) -> str:
        return "scripted-usecase"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tags = list(getattr(run_manager, "tags", None) or [])
        node = next((t.split(":", 1)[1] for t in tags if t.startswith("node:")), "")
        names = [t["function"]["name"] for t in kwargs.get("tools", [])]
        reply = self.policy(node, list(messages), names)
        if not isinstance(reply, AIMessage):
            reply = AIMessage(content=str(reply))
        return ChatResult(generations=[ChatGeneration(message=reply)])


def call(name: str, args: dict, idx: int = 0) -> AIMessage:
    """An AIMessage requesting one tool call (for scripted policies)."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}_{idx}"}])


def tool_results(messages: Iterable[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


# ------------------------------------------------------------------ tool recording

class Recorder:
    """Collects every tool call with its output (the recordings-ready trace)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def add(self, name: str, args: dict, output: Any) -> None:
        self.calls.append({"name": name, "args": dict(args), "output": output})


def recorded(fn: Callable, recorder: Recorder, name: str | None = None, description: str | None = None) -> StructuredTool:
    """Wrap `fn` as a LangChain tool whose every call (args + string output) lands in `recorder`."""
    tool = StructuredTool.from_function(fn, name=name or fn.__name__, description=description or (fn.__doc__ or "").strip())
    inner = tool.func

    def run(**kwargs):
        try:
            out = inner(**kwargs)
        except Exception as exc:  # tools report errors to the model, like a real API would
            out = f"Error: {exc}"
        text = out if isinstance(out, str) else json.dumps(out, sort_keys=True, default=str)
        recorder.add(tool.name, kwargs, text)
        return text

    tool.func = run
    return tool


def tool_loop(llm: Any, tools: Sequence[Any], messages: list, node: str, max_steps: int = 12) -> list:
    """Run model <-> tools inside one node until the model answers in text. Returns new messages."""
    by_name = {t.name: t for t in tools}
    bound = (llm.bind_tools(list(tools)) if tools else llm).with_config(tags=[f"node:{node}"])
    new: list = []
    for _ in range(max_steps):
        reply = bound.invoke(messages + new)
        new.append(reply)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            return new
        for tc in calls:
            tool = by_name.get(tc["name"])
            out = tool.invoke(tc.get("args") or {}) if tool else f"Error: unknown tool {tc['name']}"
            new.append(ToolMessage(content=str(out), tool_call_id=tc.get("id") or tc["name"], name=tc["name"]))
    new.append(AIMessage(content="(stopped: step limit reached)"))
    return new


def text_of(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


# ------------------------------------------------------------------ sandbox + data

def sandbox(files: Iterable[Path], root: str | Path | None = None, prefix: str = "usecase-") -> Path:
    """A fresh per-task directory holding copies of `files` (the stateful backend is reset per task)."""
    base = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root) if root else None))
    for f in files:
        shutil.copy2(f, base / Path(f).name)
    return base


def file_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def split_tasks(tasks: list[dict], seed: int, counts: tuple[int, int, int]) -> list[dict]:
    """Assign train/search/test with a seeded shuffle; `counts` must sum to len(tasks)."""
    if sum(counts) != len(tasks):
        raise ValueError(f"split counts {counts} do not sum to {len(tasks)}")
    order = list(range(len(tasks)))
    random.Random(seed).shuffle(order)
    names = ["train"] * counts[0] + ["search"] * counts[1] + ["test"] * counts[2]
    out = [dict(t) for t in tasks]
    for slot, idx in enumerate(order):
        out[idx]["split"] = names[slot]
    return out


# ------------------------------------------------------------------ live models

LIVE_MODELS = {
    "deepseek-flash": {"base_url": "https://api.deepseek.com", "key": "DEEPSEEK_API_KEY",
                       "price": (0.07, 0.28)},  # USD / M tokens (input, output) - flash list price
    "gpt-4o-mini": {"base_url": None, "key": "OPENAI_API_KEY", "price": (0.15, 0.60)},
}


def live_llm(model: str, temperature: float = 0.0):
    import os

    from langchain_openai import ChatOpenAI

    spec = LIVE_MODELS[model]
    kwargs = {"model": model, "temperature": temperature, "api_key": os.environ[spec["key"]], "max_retries": 3,
              "timeout": 120}
    if spec["base_url"]:
        kwargs["base_url"] = spec["base_url"]
    return ChatOpenAI(**kwargs)


class UsageMeter:
    """LangChain callback summing token usage across a run (for the cost line in docs)."""

    def __init__(self) -> None:
        from langchain_core.callbacks import BaseCallbackHandler

        meter = self

        class _Handler(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):
                for gens in response.generations:
                    for g in gens:
                        usage = getattr(getattr(g, "message", None), "usage_metadata", None) or {}
                        meter.input += int(usage.get("input_tokens", 0))
                        meter.output += int(usage.get("output_tokens", 0))

        self.input = 0
        self.output = 0
        self.handler = _Handler()

    def cost(self, model: str) -> float:
        pin, pout = LIVE_MODELS[model]["price"]
        return (self.input * pin + self.output * pout) / 1_000_000
