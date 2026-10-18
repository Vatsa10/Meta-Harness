"""Safe tool execution for LangGraph apps under search.

`wrap_tools(tools, mode, recordings, classifier)` returns LangChain tools with the same name,
description and args schema as the originals, so a graph (ToolNode, bind_tools) works unchanged:

* ``replay`` (default): read tools are answered from `Recordings` built from imported traces; a call
  that was never recorded is a *miss* (``on_missing="error"`` raises, ``"empty"`` returns "") and
  flags the run ``replay_miss``. Write tools are never executed: the call is captured (name, args,
  timestamp) and a plausible ack is returned (the recorded output of that tool when traces have one,
  else ``write_ack``). Replay is meant to run with no network inside isolation; it needs no envguard
  because nothing real is touched.
* ``dev``: tools execute for real, but only after `envguard.check` passes.

Every call emits a ``tool_call`` trace event (mode, read/write, hit/miss/captured/executed) so the
judge and proposer see what the agent tried to do.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping

from . import envguard

ToolMode = Literal["replay", "dev"]
MODES = ("replay", "dev")
WRITE_WORDS = ("send", "post", "create", "update", "delete", "write", "pay", "insert", "commit", "email",
               "remove", "put", "patch", "upload", "publish", "submit", "transfer", "charge", "refund",
               "cancel", "book", "drop", "set", "save", "modify", "execute", "deploy", "notify")
READ_WORDS = ("get", "read", "search", "list", "fetch", "lookup", "look", "query", "find", "retrieve",
              "show", "describe", "view", "check", "count", "browse", "inspect", "load")


class ReplayMiss(LookupError):
    """A read tool was called with arguments no imported trace recorded."""


# ------------------------------------------------------------------ recordings

def canonical_args(args: Any) -> str:
    """Stable JSON for a tool call's arguments (key order and whitespace never matter)."""
    if hasattr(args, "model_dump"):
        args = args.model_dump()
    if isinstance(args, str):
        parsed = _maybe_json(args)
        args = parsed if isinstance(parsed, (dict, list)) else args
    return json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _content(value: Any) -> Any:
    """Tool output as the tool returned it: unwrap LangSmith {"output": ...} and ToolMessage dicts."""
    value = _maybe_json(value)
    for _ in range(3):
        if isinstance(value, dict) and len(value) == 1 and next(iter(value)) in ("output", "result", "content"):
            value = next(iter(value.values()))
        elif isinstance(value, dict) and "content" in value and value.get("type") in ("tool", "ToolMessage"):
            value = value["content"]
        elif isinstance(value, dict) and isinstance(value.get("kwargs"), dict) and "content" in value["kwargs"]:
            value = value["kwargs"]["content"]
        else:
            break
    return value


def _args(value: Any) -> Any:
    """Tool inputs: unwrap LangSmith's {"input": "<json>"} wrapper when it hides the real arguments."""
    value = _maybe_json(value)
    if isinstance(value, dict) and set(value) == {"input"}:
        inner = _maybe_json(value["input"])
        if isinstance(inner, dict):
            return inner
    return value


class Recordings:
    """Tool outputs observed in real traces, keyed by tool name + canonical JSON args."""

    def __init__(self) -> None:
        self._outputs: dict[tuple[str, str], Any] = {}
        self._last: dict[str, Any] = {}

    def add(self, name: str, args: Any, output: Any) -> None:
        self._outputs[(str(name), canonical_args(args))] = output
        self._last[str(name)] = output

    def lookup(self, name: str, args: Any) -> tuple[bool, Any]:
        key = (str(name), canonical_args(args))
        if key in self._outputs:
            return True, self._outputs[key]
        return False, None

    def ack(self, name: str) -> tuple[bool, Any]:
        """The most recent recorded output of `name` (used as a write tool's plausible ack)."""
        if name in self._last:
            return True, self._last[name]
        return False, None

    def __len__(self) -> int:
        return len(self._outputs)

    @classmethod
    def from_calls(cls, calls: Iterable[Mapping]) -> "Recordings":
        rec = cls()
        for call in calls:
            rec.add(call["name"], call.get("args", {}), call.get("output"))
        return rec

    @classmethod
    def from_traces(cls, tasks_or_trace_files: Iterable[Any] | Any) -> "Recordings":
        """Read tool calls from tasks (dicts with ``tool_calls``) or trace files.

        Accepted: task dicts / JSONL rows carrying ``tool_calls: [{name, args, output}]``; LangSmith
        run exports (``run_type == "tool"``, nested ``child_runs``); Langfuse exports (observations of
        ``type == "TOOL"``); OTel spans with ``gen_ai.tool.name`` / ``gen_ai.tool.call.arguments`` /
        ``gen_ai.tool.call.result``. Files may be .json or .jsonl.
        """
        if isinstance(tasks_or_trace_files, (str, Path, dict)):
            tasks_or_trace_files = [tasks_or_trace_files]
        calls: list[dict] = []
        for item in tasks_or_trace_files:
            if isinstance(item, (str, Path)):
                calls.extend(_calls_in(_read_file(Path(item))))
            else:
                calls.extend(_calls_in(item))
        return cls.from_calls(calls)


def _read_file(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _otel_attrs(span: Mapping) -> dict:
    raw = span.get("attributes") or {}
    if isinstance(raw, dict):
        return dict(raw)
    out = {}
    for item in raw:
        if isinstance(item, dict):
            value = item.get("value")
            if isinstance(value, dict) and len(value) == 1:
                value = next(iter(value.values()))
            out[item.get("key")] = value
    return out


def _calls_in(data: Any) -> list[dict]:
    calls: list[dict] = []
    if isinstance(data, list):
        for item in data:
            calls.extend(_calls_in(item))
        return calls
    if not isinstance(data, dict):
        return calls
    for call in data.get("tool_calls") or []:
        if isinstance(call, dict) and call.get("name") and any(k in call for k in ("output", "result")):
            calls.append({"name": call["name"],
                          "args": _args(call.get("args", call.get("arguments", call.get("input", {})))),
                          "output": _content(call.get("output", call.get("result")))})
    if data.get("run_type") == "tool" and data.get("name"):  # LangSmith run
        calls.append({"name": data["name"], "args": _args(data.get("inputs") or {}),
                      "output": _content(data.get("outputs"))})
    if str(data.get("type") or "").upper() == "TOOL" and data.get("name"):  # Langfuse observation
        calls.append({"name": data["name"], "args": _args(data.get("input") or {}),
                      "output": _content(data.get("output"))})
    if "spanId" in data or "span_id" in data:  # OTel span
        attrs = _otel_attrs(data)
        if attrs.get("gen_ai.tool.name"):
            calls.append({"name": attrs["gen_ai.tool.name"],
                          "args": _args(attrs.get("gen_ai.tool.call.arguments") or {}),
                          "output": _content(attrs.get("gen_ai.tool.call.result"))})
    for key in ("child_runs", "observations", "runs", "data", "traces", "resourceSpans", "resource_spans",
                "scopeSpans", "scope_spans", "spans"):
        if isinstance(data.get(key), (list, dict)):
            calls.extend(_calls_in(data[key]))
    return calls


# ------------------------------------------------------------------ classification

def _words(text: str) -> list[str]:
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    return [w for w in re.split(r"[^a-zA-Z0-9]+", text.lower()) if w]


def _inflections(verb: str) -> set[str]:
    stem = verb[:-1] if verb.endswith("e") else verb
    return {verb, verb + "s", verb + "es", stem + "ed", stem + "ing", verb + "d", verb + verb[-1] + "ed",
            verb + verb[-1] + "ing"}


def heuristic_kind(name: str, description: str = "") -> str:
    """'write' if name/description mention a side effect, 'read' if the name says read, else 'write'."""
    words = _words(name) + _words(description)
    if any(w in _inflections(verb) for w in words for verb in WRITE_WORDS):
        return "write"
    if any(w in READ_WORDS for w in _words(name)):
        return "read"
    return "write"


def classifier_from_settings(path: str | Path | None = None) -> dict[str, str]:
    """The `[tools]` section of meta-harness.toml: ``tool_name = "read" | "write"``."""
    section = envguard.load_settings(path).get("tools") or {}
    return {str(k): str(v).lower() for k, v in section.items() if str(v).lower() in ("read", "write")}


def classify(tool: Any, classifier: Mapping[str, str] | Callable[[Any], str | None] | None = None) -> str:
    name = getattr(tool, "name", "")
    explicit = None
    if callable(classifier):
        explicit = classifier(tool)
    elif classifier:
        explicit = classifier.get(name)
    if explicit is not None:
        explicit = str(explicit).lower()
        if explicit not in ("read", "write"):
            raise ValueError(f"tool {name!r}: classification must be 'read' or 'write', got {explicit!r}")
        return explicit
    return heuristic_kind(name, getattr(tool, "description", "") or "")


# ------------------------------------------------------------------ session + wrappers

class ToolSession:
    """Shared state of one run's wrapped tools: captured writes, events and flags."""

    def __init__(self, mode: ToolMode = "replay", recordings: Recordings | None = None, *,
                 on_missing: Literal["error", "empty"] = "error", write_ack: Any = "ok", trace: Any = None,
                 env: Mapping[str, str] | None = None, production_markers: Iterable[str] | None = None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if on_missing not in ("error", "empty"):
            raise ValueError("on_missing must be 'error' or 'empty'")
        self.mode = mode
        self.recordings = recordings if recordings is not None else Recordings()
        self.on_missing = on_missing
        self.write_ack = write_ack
        self.trace = trace
        self.env = env
        self.production_markers = production_markers
        self.captured: list[dict] = []
        self.events: list[dict] = []
        self.flags: set[str] = set()
        if mode == "dev":
            self.guard()

    @property
    def replay_miss(self) -> bool:
        return "replay_miss" in self.flags

    def guard(self) -> None:
        envguard.check(self.env, self.production_markers)

    def log(self, payload: dict) -> None:
        self.events.append(payload)
        if self.trace is None:
            return
        if hasattr(self.trace, "event"):
            self.trace.event("tool_call", payload)
        elif callable(self.trace):
            self.trace("tool_call", payload)

    def _flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.add(flag)
            if self.trace is not None and hasattr(self.trace, "event"):
                self.trace.event(flag, {"flag": flag})

    def before(self, name: str, kind: str, args: Any) -> tuple[bool, Any]:
        """(handled, result): replay answers here; dev returns handled=False after the guard."""
        base = {"tool": name, "mode": self.mode, "kind": kind, "args": _jsonable(args)}
        if self.mode == "dev":
            self.guard()
            self.log({**base, "outcome": "executed"})
            return False, None
        if kind == "write":
            stamp = datetime.now(timezone.utc).isoformat()
            self.captured.append({"name": name, "args": _jsonable(args), "timestamp": stamp, "time": time.time()})
            recorded, ack = self.recordings.ack(name)
            self.log({**base, "outcome": "captured", "timestamp": stamp})
            return True, ack if recorded else self.write_ack
        hit, output = self.recordings.lookup(name, args)
        if hit:
            self.log({**base, "outcome": "hit"})
            return True, output
        self.log({**base, "outcome": "miss"})
        self._flag("replay_miss")
        if self.on_missing == "error":
            raise ReplayMiss(f"no recorded output for {name}({canonical_args(args)})")
        return True, ""


def _jsonable(value: Any) -> Any:
    return json.loads(canonical_args(value))


def wrap_tools(tools: Iterable[Any], mode: ToolMode = "replay", recordings: Recordings | None = None,
               classifier: Mapping[str, str] | Callable[[Any], str | None] | None = None, *,
               session: ToolSession | None = None, **session_options: Any) -> list[Any]:
    """Wrap LangChain tools for safe execution. The shared ToolSession is on each wrapper's `.session`.

    `classifier` maps tool name -> "read"/"write" (or is a callable); unmapped tools fall back to
    the name/description heuristic, and anything unrecognised is treated as a write.
    `session_options` go to ToolSession (on_missing, write_ack, trace, env, production_markers).
    """
    from langchain_core.tools import BaseTool, StructuredTool  # lazy: optional dependency

    if session is None:
        session = ToolSession(mode, recordings, **session_options)
    wrapped = []
    for tool in tools:
        if not isinstance(tool, BaseTool):
            from langchain_core.tools import tool as as_tool
            tool = as_tool(tool)
        kind = classify(tool, classifier)
        wrapped.append(_wrap_one(tool, kind, session, StructuredTool))
    return wrapped


def _wrap_one(tool: Any, kind: str, session: ToolSession, structured: Any) -> Any:
    name = tool.name

    def run(**kwargs: Any) -> Any:
        handled, result = session.before(name, kind, kwargs)
        if handled:
            return result
        return tool.invoke(kwargs)

    async def arun(**kwargs: Any) -> Any:
        handled, result = session.before(name, kind, kwargs)
        if handled:
            return result
        return await tool.ainvoke(kwargs)

    wrapper = structured(name=name, description=tool.description, args_schema=tool.args_schema,
                         func=run, coroutine=arun, return_direct=getattr(tool, "return_direct", False))
    object.__setattr__(wrapper, "metadata", {**(tool.metadata or {}), "meta_harness_kind": kind})
    wrapper.__dict__["session"] = session
    wrapper.__dict__["kind"] = kind
    wrapper.__dict__["wrapped"] = tool
    return wrapper
