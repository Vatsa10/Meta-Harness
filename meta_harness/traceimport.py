"""Import production traces (OpenTelemetry GenAI, Langfuse, generic JSONL) as eval sets.

Each trace becomes one task: `input`, the original `output` as `reference_output`, `criteria`
derived from feedback, scores and errors (so `meta_harness.judge:verdict` can grade it), and
`source_trace_id`. Selection is failure-first (errors, then negative feedback / low scores),
deduplicated by normalised input, and split train/search/test with a seeded, failure-stratified
shuffle. Redaction runs over every text field; `pii=True` also scrubs emails, phones and cards.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .redact import redact_obj, safe_id

FORMATS = ("otel", "langfuse", "jsonl")
LOW_SCORE = 0.5
SPLITS = ("train", "search", "test")

SEVERITY = {"error": 0, "negative": 1, "ok": 2}

_NEGATIVE_WORDS = {"bad", "wrong", "incorrect", "negative", "thumbs_down", "thumbsdown", "thumbs-down",
                   "down", "no", "unhelpful", "not helpful", "fail", "failed", "false", "-1", "0", "dislike"}
_NEGATIVE_HINT = re.compile(r"(?i)\b(wrong|incorrect|unhelpful|not helpful|hallucinat\w*|useless|bad|broken|"
                            r"didn'?t|doesn'?t|failed|missing)\b")

EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
CARD = re.compile(r"(?<![\w])\d(?:[ -]?\d){12,18}(?![\w])")
PHONE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)\s?|\d{2,4}[\s.-])\d{3,4}[\s.-]\d{3,4}(?![\w])"
                   r"|\+\d{10,14}(?![\w])")


@dataclass
class Trace:
    id: str
    input: str
    output: str
    error: str | None = None
    feedback: list[str] = field(default_factory=list)
    scores: list[tuple[str, float, str]] = field(default_factory=list)
    negative_feedback: bool = False

    @property
    def min_score(self) -> float | None:
        return min((value for _, value, _ in self.scores), default=None)

    @property
    def kind(self) -> str:
        if self.error:
            return "error"
        score = self.min_score
        if self.negative_feedback or (score is not None and score < LOW_SCORE):
            return "negative"
        return "ok"


# ---------------------------------------------------------------- text helpers

def text_of(value: Any) -> str:
    """Flatten str / chat-message lists / {content|text|parts} dicts into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in "[{":
            try:
                return text_of(json.loads(stripped))
            except ValueError:
                pass
        return value
    if isinstance(value, (int, float, bool)):
        return json.dumps(value)
    if isinstance(value, list):
        messages = [item for item in value if not (isinstance(item, dict) and item.get("role") == "system")]
        return "\n".join(part for part in (text_of(item) for item in messages) if part)
    if isinstance(value, dict):
        for key in ("content", "text", "parts", "messages", "input", "output", "query", "question",
                    "completion", "message", "choices"):
            if key in value:
                return text_of(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _is_negative(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value <= 0
    text = str(value).strip().lower()
    return text in _NEGATIVE_WORDS or bool(_NEGATIVE_HINT.search(text)) or text.startswith("\U0001f44e")


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- parsers

def _otel_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        return [_otel_value(item) for item in value["arrayValue"].get("values", [])]
    return value


def _otel_attrs(span: dict) -> dict:
    raw = span.get("attributes") or {}
    if isinstance(raw, dict):
        return dict(raw)
    return {item.get("key"): _otel_value(item.get("value")) for item in raw if isinstance(item, dict)}


def _otel_spans(data: Any) -> Iterable[dict]:
    if isinstance(data, list):
        for item in data:
            yield from _otel_spans(item)
        return
    if not isinstance(data, dict):
        return
    if "spanId" in data or "span_id" in data:
        yield data
        return
    for key in ("resourceSpans", "resource_spans", "scopeSpans", "scope_spans",
                "instrumentationLibrarySpans", "spans", "data"):
        if key in data:
            yield from _otel_spans(data[key])


def _otel_error(span: dict) -> str | None:
    status = span.get("status") or {}
    code = status.get("code")
    if code in (2, "2", "STATUS_CODE_ERROR", "ERROR"):
        return status.get("message") or "span status ERROR"
    attrs = _otel_attrs(span)
    if attrs.get("error.type"):
        return str(attrs.get("error.message") or attrs["error.type"])
    for event in span.get("events") or []:
        if event.get("name") == "exception":
            ev = _otel_attrs(event)
            return str(ev.get("exception.message") or ev.get("exception.type") or "exception")
    return None


def parse_otel(data: Any) -> list[Trace]:
    """OTel JSON export (OTLP/JSON resourceSpans or a flat span list) with GenAI semconv attributes."""
    by_trace: dict[str, list[dict]] = {}
    for span in _otel_spans(data):
        by_trace.setdefault(str(span.get("traceId") or span.get("trace_id") or ""), []).append(span)
    traces = []
    for trace_id, spans in by_trace.items():
        spans.sort(key=lambda s: int(s.get("startTimeUnixNano") or 0))
        prompt = completion = ""
        error = None
        feedback: list[str] = []
        scores: list[tuple[str, float, str]] = []
        negative = False
        for span in spans:
            attrs = _otel_attrs(span)
            got_in = attrs.get("gen_ai.input.messages", attrs.get("gen_ai.prompt"))
            got_out = attrs.get("gen_ai.output.messages", attrs.get("gen_ai.completion"))
            if got_in is not None and not prompt:
                prompt = text_of(got_in)
            if got_out is not None:
                completion = text_of(got_out)
            error = error or _otel_error(span)
            for key, value in attrs.items():
                lowered = str(key).lower()
                if lowered.endswith("feedback") or lowered.endswith("feedback.value"):
                    feedback.append(text_of(value))
                    negative = negative or _is_negative(value)
                elif lowered.endswith("score") or lowered.endswith("score.value"):
                    number = _num(value)
                    if number is not None:
                        scores.append((str(key), number, ""))
        if prompt:
            traces.append(Trace(trace_id, prompt, completion, error, feedback, scores, negative))
    return traces


def parse_langfuse(data: Any) -> list[Trace]:
    """Langfuse trace export: a list (or {"data": [...]}) of traces with observations and scores."""
    if isinstance(data, dict):
        data = data.get("data", data.get("traces", [data]))
    traces = []
    for trace in data or []:
        observations = sorted(trace.get("observations") or [], key=lambda o: str(o.get("startTime") or ""))
        prompt = text_of(trace.get("input"))
        completion = text_of(trace.get("output"))
        if not prompt:
            prompt = next((text_of(o.get("input")) for o in observations if o.get("input") is not None), "")
        if not completion:
            completion = next((text_of(o.get("output")) for o in reversed(observations)
                               if o.get("output") is not None), "")
        error = None
        for obs in observations:
            if str(obs.get("level") or "").upper() == "ERROR":
                error = obs.get("statusMessage") or f"{obs.get('name') or 'observation'} failed"
                break
        feedback, scores, negative = [], [], False
        for score in trace.get("scores") or []:
            name = str(score.get("name") or "score")
            comment = str(score.get("comment") or "")
            value = score.get("value", score.get("stringValue"))
            number = _num(value)
            data_type = str(score.get("dataType") or "").upper()
            if number is not None and data_type != "CATEGORICAL":
                scores.append((name, number, comment))
            else:
                feedback.append(" ".join(part for part in (str(value), comment) if part))
                negative = negative or _is_negative(value)
            if comment and _NEGATIVE_HINT.search(comment):
                negative = True
        if prompt:
            traces.append(Trace(str(trace.get("id") or ""), prompt, completion, error, feedback, scores, negative))
    return traces


def parse_jsonl(text: str) -> list[Trace]:
    """Generic JSONL: {input, output, feedback?, error?, score?, id?} per line."""
    traces = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        prompt = text_of(row.get("input"))
        if not prompt:
            continue
        feedback, negative = [], False
        if row.get("feedback") is not None:
            feedback.append(text_of(row["feedback"]))
            negative = _is_negative(row["feedback"])
        scores = []
        number_score = _num(row.get("score"))
        if number_score is not None:
            scores.append(("score", number_score, ""))
        error = row.get("error")
        traces.append(Trace(str(row.get("id") or row.get("trace_id") or f"line-{number}"), prompt,
                            text_of(row.get("output")), text_of(error) if error else None,
                            feedback, scores, negative))
    return traces


def load(fmt: str, path: str | Path) -> list[Trace]:
    raw = Path(path).read_text(encoding="utf-8")
    if fmt == "jsonl":
        return parse_jsonl(raw)
    data = json.loads(raw)
    if fmt == "otel":
        return parse_otel(data)
    if fmt == "langfuse":
        return parse_langfuse(data)
    raise ValueError(f"unknown trace format {fmt!r}; known: {', '.join(FORMATS)}")


# ---------------------------------------------------------------- privacy

def scrub_pii(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = CARD.sub("[card]", text)
    return PHONE.sub("[phone]", text)


def _clean(value: Any, pii: bool) -> Any:
    value = redact_obj(value)
    if not pii:
        return value
    if isinstance(value, str):
        return scrub_pii(value)
    if isinstance(value, dict):
        return {key: _clean(item, pii) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item, pii) for item in value]
    return value


# ---------------------------------------------------------------- tasks

def criteria_for(trace: Trace) -> str:
    lines = ["Respond to the user's request correctly, completely and helpfully."]
    if trace.error:
        lines.append(f"The original run failed with this error: {trace.error}. "
                     "A passing answer completes the request without that failure.")
    if trace.feedback:
        lines.append("User feedback on the original answer: " + " | ".join(trace.feedback))
    for name, value, comment in trace.scores:
        lines.append(f"The original answer scored {name}={value:g}" + (f" ({comment})." if comment else "."))
    if trace.output:
        if trace.kind == "ok":
            lines.append("The original answer below was accepted; pass answers that are equivalent in "
                         "substance:\n" + trace.output)
        else:
            lines.append("The original answer below was unsatisfactory; fail answers that repeat its "
                         "mistakes:\n" + trace.output)
    return "\n".join(lines)


def to_task(trace: Trace, pii: bool = False) -> dict:
    task = {
        "id": safe_id(trace.id or trace.input[:40]),
        "input": trace.input,
        "reference_output": trace.output,
        "criteria": criteria_for(trace),
        "source_trace_id": trace.id,
        "failure": trace.kind,
    }
    return _clean(task, pii)


def normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _rank(trace: Trace) -> tuple:
    score = trace.min_score
    return (SEVERITY[trace.kind], score if score is not None else float("inf"))


def select(traces: list[Trace], limit: int | None = None) -> list[Trace]:
    """Failure-first (stable within a class), deduplicated by normalised input (the most severe
    occurrence wins), optionally capped at `limit`."""
    ordered = sorted(enumerate(traces), key=lambda pair: (_rank(pair[1]), pair[0]))
    seen, kept = set(), []
    for _, trace in ordered:
        key = normalise(trace.input)
        if key in seen:
            continue
        seen.add(key)
        kept.append(trace)
    return kept[:limit] if limit is not None else kept


def split(tasks: list[dict], seed: int = 0, train: float = 0.2, test: float = 0.3) -> dict[str, list[dict]]:
    """Seeded split stratified by failure kind, so failures land in every split. Each split keeps
    failure-first order."""
    if not 0 <= train < 1 or not 0 < test < 1 or train + test >= 1:
        raise ValueError("need 0 <= train, 0 < test and train + test < 1")
    rng = random.Random(seed)
    groups: dict[str, list[tuple[int, dict]]] = {}
    for index, task in enumerate(tasks):
        groups.setdefault(task.get("failure", "ok"), []).append((index, task))
    dealt: list[tuple[int, dict]] = []
    for kind in sorted(groups, key=lambda name: SEVERITY.get(name, len(SEVERITY))):
        group = groups[kind]
        rng.shuffle(group)
        dealt += group
    targets = {"test": test, "search": 1 - train - test, "train": train}
    out: dict[str, list[tuple[int, dict]]] = {name: [] for name in SPLITS}
    for position, item in enumerate(dealt, start=1):
        # largest deficit against its target share wins: spreads every failure kind over the splits
        name = max(targets, key=lambda n: targets[n] * position - len(out[n]))
        out[name].append(item)
    return {name: [task for _, task in sorted(items, key=lambda pair: pair[0])] for name, items in out.items()}


def write_jsonl(path: str | Path, rows: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return path


def import_traces(fmt: str, path: str | Path, *, pii: bool = False, limit: int | None = None) -> list[dict]:
    return [to_task(trace, pii) for trace in select(load(fmt, path), limit)]


def stats(traces: list[Trace]) -> dict:
    kinds = {kind: 0 for kind in SEVERITY}
    for trace in traces:
        kinds[trace.kind] += 1
    unique = len({normalise(trace.input) for trace in traces})
    return {
        "traces": len(traces),
        "unique_inputs": unique,
        "duplicates": len(traces) - unique,
        "with_output": sum(1 for t in traces if t.output),
        "errors": kinds["error"],
        "negative": kinds["negative"],
        "ok": kinds["ok"],
        "with_feedback": sum(1 for t in traces if t.feedback),
        "negative_feedback": sum(1 for t in traces if t.negative_feedback),
        "scored": sum(1 for t in traces if t.scores),
        "low_score": sum(1 for t in traces if t.min_score is not None and t.min_score < LOW_SCORE),
    }
