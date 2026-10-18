"""Import LangSmith runs (exported JSON / JSONL, or pulled live over the REST API) as eval tasks.

Each root run becomes one task with the same fields `meta_harness.traceimport` produces (id, input,
reference_output, criteria, source_trace_id, failure) plus `tool_calls` ([{name, args, output}] from
tool descendant runs, in start order), `node_outputs` (output of each named chain/llm descendant),
`project` and `metadata`. Selection, criteria and PII scrubbing go through traceimport's public
helpers; every field is redacted. Stdlib only; the live pull uses urllib.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from .. import traceimport as ti
from ..redact import redact, redact_obj

ENDPOINT = "https://api.smith.langchain.com"
API_KEY_ENV = "LANGSMITH_API_KEY"
ENDPOINT_ENV = "LANGSMITH_ENDPOINT"
USER_AGENT = "meta-harness-langsmith/1"
PAGE_SIZE = 100
TIMEOUT = 30
NODE_TYPES = ("chain", "llm", "prompt", "parser", "retriever")
_NEGATIVE = {"bad", "wrong", "incorrect", "negative", "thumbs_down", "thumbs-down", "thumbsdown", "no"}
_NEGATIVE_HINTS = ("wrong", "incorrect", "unhelpful", "not helpful", "hallucinat", "useless", "broken")


class LangSmithError(RuntimeError):
    """A failed LangSmith call; the message never carries the API key."""


# ---------------------------------------------------------------- parsing

def parse(data: Any) -> list[dict]:
    """Flatten an export (list, {"runs": [...]}, or one run) into run dicts, including nested
    `child_runs`, with `parent_run_id` filled in for nested children."""
    if isinstance(data, dict):
        data = data["runs"] if isinstance(data.get("runs"), list) else [data]
    if not isinstance(data, list):
        raise ValueError("expected a list of LangSmith runs")
    flat: list[dict] = []

    def walk(run: Any, parent: str | None) -> None:
        if not isinstance(run, dict):
            return
        run = dict(run)
        children = run.pop("child_runs", None) or []
        if parent is not None and not run.get("parent_run_id"):
            run["parent_run_id"] = parent
        flat.append(run)
        for child in children:
            walk(child, str(run.get("id", "")))

    for item in data:
        walk(item, None)
    return flat


def parse_text(text: str) -> list[dict]:
    """JSON (list or {"runs": ...}) or JSONL, one run per line."""
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return parse(json.loads(stripped))
    except ValueError:
        pass
    runs: list[dict] = []
    for number, line in enumerate(stripped.splitlines(), start=1):
        if line.strip():
            try:
                runs += parse(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"line {number}: {exc}") from None
    return runs


def load(path: str | Path) -> list[dict]:
    return parse_text(Path(path).read_text(encoding="utf-8"))


def _is_root(run: dict) -> bool:
    if "is_root" in run:
        return bool(run["is_root"])
    return not run.get("parent_run_id")


def _order(run: dict) -> str:
    return str(run.get("dotted_order") or run.get("start_time") or "")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _negative(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in _NEGATIVE or any(hint in text for hint in _NEGATIVE_HINTS)


def _feedback(run: dict) -> tuple[list[str], list[tuple[str, float, str]], bool]:
    """Comments, numeric scores and a negative flag from `feedback` rows and `feedback_stats`."""
    comments: list[str] = []
    scores: list[tuple[str, float, str]] = []
    negative = False
    rows = run.get("feedback") or []
    if isinstance(rows, dict):
        rows = [rows]
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = str(row.get("key", "feedback"))
        comment = str(row.get("comment") or "")
        value = row.get("score")
        if value is None:
            value = row.get("value")
        if comment:
            comments.append(comment)
            negative = negative or _negative(comment)
        number = _number(value)
        if number is not None:
            scores.append((key, number, comment))
        elif value is not None:
            comments.append(f"{key}: {value}")
            negative = negative or _negative(value)
    stats = run.get("feedback_stats") or {}
    if isinstance(stats, dict):
        for key, stat in stats.items():
            avg = _number(stat.get("avg")) if isinstance(stat, dict) else None
            if avg is not None and not any(name == key for name, _, _ in scores):
                scores.append((str(key), avg, ""))
    return comments, scores, negative


def _descendants(root: dict, by_parent: dict[str, list[dict]]) -> list[dict]:
    out, stack, seen = [], list(by_parent.get(str(root.get("id")), [])), set()
    while stack:
        run = stack.pop()
        if id(run) in seen:
            continue
        seen.add(id(run))
        out.append(run)
        stack += by_parent.get(str(run.get("id")), [])
    return sorted(out, key=_order)


def _clean(value: Any, pii: bool) -> Any:
    value = redact_obj(value)
    if not pii:
        return value
    if isinstance(value, str):
        return ti.scrub_pii(value)
    if isinstance(value, dict):
        return {key: _clean(item, pii) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item, pii) for item in value]
    return value


def _unwrap(outputs: Any) -> Any:
    if isinstance(outputs, dict) and set(outputs) == {"output"}:
        return outputs["output"]
    return outputs


def to_tasks(runs: list[dict], *, pii: bool = False, limit: int | None = None) -> list[dict]:
    """Root runs -> eval tasks, failure-first and deduplicated (traceimport.select)."""
    by_parent: dict[str, list[dict]] = {}
    for run in runs:
        if run.get("parent_run_id"):
            by_parent.setdefault(str(run["parent_run_id"]), []).append(run)
    extras: dict[int, dict] = {}
    traces: list[ti.Trace] = []
    for run in runs:
        if not _is_root(run):
            continue
        comments, scores, negative = _feedback(run)
        descendants = _descendants(run, by_parent)
        if run.get("error"):
            error = str(run["error"])
        else:  # a failed tool or node inside an otherwise "successful" graph run still counts
            error = "; ".join(f"{child.get('name') or child.get('run_type')}: {child['error']}"
                              for child in descendants if child.get("error")) or None
        trace = ti.Trace(id=str(run.get("trace_id") or run.get("id") or ""),
                         input=ti.text_of(run.get("inputs")), output=ti.text_of(run.get("outputs")),
                         error=error, feedback=comments, scores=scores, negative_feedback=negative)
        if not trace.input.strip():
            continue
        tool_calls = []
        for child in descendants:
            if child.get("run_type") == "tool":
                call = {"name": str(child.get("name", "")), "args": child.get("inputs") or {},
                        "output": _unwrap(child.get("outputs"))}
                if child.get("error"):
                    call["error"] = str(child["error"])
                tool_calls.append(call)
        nodes = {str(child["name"]): ti.text_of(child.get("outputs"))
                 for child in descendants if child.get("run_type") in NODE_TYPES and child.get("name")}
        extra = run.get("extra") if isinstance(run.get("extra"), dict) else {}
        extras[id(trace)] = {
            "tool_calls": tool_calls,
            "node_outputs": nodes,
            "project": str(run.get("session_name") or run.get("project_name") or run.get("session_id") or ""),
            "metadata": extra.get("metadata") or {},
        }
        traces.append(trace)
    tasks = []
    for trace in ti.select(traces, limit):
        task = ti.to_task(trace, pii)
        task.update(_clean(extras[id(trace)], pii))
        tasks.append(task)
    return tasks


def import_file(path: str | Path, *, pii: bool = False, limit: int | None = None) -> list[dict]:
    return to_tasks(load(path), pii=pii, limit=limit)


# ---------------------------------------------------------------- live pull

def _scrub(text: str, api_key: str) -> str:
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return redact(text)


def _call(method: str, url: str, api_key: str, body: dict | None, opener: Callable) -> Any:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={
        "x-api-key": api_key, "Content-Type": "application/json", "Accept": "application/json",
        "User-Agent": USER_AGENT})
    where = f"{method} {urllib.parse.urlsplit(url).path}"
    try:
        with opener(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise LangSmithError(_scrub(f"LangSmith HTTP {exc.code} on {where}: {detail}", api_key)) from None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LangSmithError(_scrub(f"LangSmith request {where} failed: {exc}", api_key)) from None


def _query(base: str, api_key: str, opener: Callable, body: dict, limit: int | None, page_size: int) -> list[dict]:
    out: list[dict] = []
    cursor = None
    while limit is None or len(out) < limit:
        page = dict(body, limit=page_size if limit is None else min(page_size, limit - len(out)))
        if cursor:
            page["cursor"] = cursor
        reply = _call("POST", f"{base}/api/v1/runs/query", api_key, page, opener)
        if not isinstance(reply, dict):
            break
        runs = [run for run in reply.get("runs") or [] if isinstance(run, dict)]
        out += runs
        cursor = (reply.get("cursors") or {}).get("next")
        if not cursor or not runs:
            break
    return out if limit is None else out[:limit]


def fetch_runs(project: str, *, limit: int = 100, api_key: str | None = None, endpoint: str | None = None,
               page_size: int = PAGE_SIZE, opener: Callable | None = None) -> list[dict]:
    """Up to `limit` root runs of the named project plus all their descendant runs."""
    api_key = api_key or os.environ.get(API_KEY_ENV, "")
    if not api_key:
        raise LangSmithError(f"set {API_KEY_ENV} to pull runs from LangSmith")
    base = (endpoint or os.environ.get(ENDPOINT_ENV) or ENDPOINT).rstrip("/")
    opener = opener or urllib.request.urlopen
    sessions = _call("GET", f"{base}/api/v1/sessions?name={urllib.parse.quote(project)}", api_key, None, opener)
    if isinstance(sessions, dict):
        sessions = sessions.get("sessions") or [sessions]
    match = [s for s in sessions or [] if isinstance(s, dict) and s.get("name") == project and s.get("id")]
    if not match:
        raise LangSmithError(f"no LangSmith project named {project!r}")
    session_id = match[0]["id"]
    roots = _query(base, api_key, opener, {"session": [session_id], "is_root": True}, limit, page_size)
    runs = list(roots)
    for root in roots:
        trace_id = root.get("trace_id") or root.get("id")
        children = _query(base, api_key, opener, {"session": [session_id], "trace": trace_id, "is_root": False},
                          None, page_size)
        runs += [child for child in children if child.get("id") != root.get("id")]
    for run in runs:
        run.setdefault("session_name", project)
    return runs
