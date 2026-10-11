"""Provider-agnostic tool-using proposer: the core loop of the Meta-Harness paper.

For each candidate the agent gets a rendered prompt and four confined tools (list, read, grep,
write_candidate) over a view of the experience filesystem. It loops model turn -> tool calls ->
tool results until it calls write_candidate or exhausts its budget. Every message and every tool
call is appended, redacted, to proposals/iteration-NNNN/candidate-KK.proposer_log.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from .chat_types import ToolCall
from .metrics import estimate_tokens
from .proposer_tools import TOOL_SPECS, ToolRefused, ToolSandbox, dispatch
from .redact import bounded, redact, redact_obj, safe_id
from .timeouts import Aborted, Watchdog

PROMPT_PATH = Path(__file__).with_name("proposer_prompt.md")
SYSTEM = ("You are the proposer in a harness-search loop. Investigate the experience directory "
          "with the tools, then submit exactly one candidate with write_candidate.")
NUDGE = "Use the tools; finish by calling write_candidate."
READ_ONLY = ("list", "read", "grep")
MAX_CALLS_PER_TURN = 8
TURN_RESULT_TOKENS = 10_000
MAX_LENGTH_STOPS = 3
STUCK_REMIND = 5
STUCK_STOP = 8
REMIND_REPEATING = ("Reminder: you have made the same tool call with the same arguments 5 times. "
                    "Change your approach or call write_candidate.")
REMIND_FAILING = ("Reminder: the same tool call keeps failing with the same arguments. "
                  "Change the arguments or your approach.")
NOT_EXECUTED_WRITE = "error: not executed - write_candidate ended the turn"
NOT_EXECUTED_BUDGET = "error: not executed - the tool call budget is exhausted"
NOT_EXECUTED_CAP = f"error: not executed - at most {MAX_CALLS_PER_TURN} tool calls per turn"
TRUNCATED = "error: your output was truncated - call one tool with smaller arguments"
BATCH_POINTER = (f"error: the results of this turn exceeded {TURN_RESULT_TOKENS} tokens; this result "
                 f"was dropped - narrow the request (smaller limit, narrower path or pattern)")
SPLIT_DENIED = "error: the held-out test split is not available to the proposer"
_DEFAULTS = {"domain_notes": ""}


class _Strict(dict):
    def __missing__(self, key: str) -> Any:
        if key in _DEFAULTS:
            return _DEFAULTS[key]
        raise KeyError(key)


def render_prompt(template: str, **values: Any) -> str:
    """Fill {name} placeholders. Unknown or leftover placeholders raise KeyError - never shipped raw.

    Only `domain_notes` has a default (empty): a domain without notes is legitimate.
    """
    return template.format_map(_Strict(values))


class ProposerBudgetExceeded(RuntimeError):
    stop_reason = "step_limit"


class ProposerTruncated(RuntimeError):
    stop_reason = "error"


class ProposerStuck(RuntimeError):
    stop_reason = "stuck"


def _redact_args(name: str, arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return bounded(str(arguments), 500)
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        if name == "write_candidate" and key == "source" and isinstance(value, str):
            out[key] = len(value)  # the source is saved as harness.py anyway
        elif isinstance(value, str):
            out[key] = bounded(value, 500)
        else:
            out[key] = redact_obj(value)
    return out


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def fingerprint(root: Path) -> str:
    """v1: sha256 of sorted (candidate_id, scores.json sha256) pairs plus frontier.json sha256."""
    candidates = Path(root) / "candidates"
    pairs = sorted((p.parent.name, _sha(p)) for p in candidates.glob("*/scores.json")) \
        if candidates.is_dir() else []
    digest = hashlib.sha256()
    for cid, sha in pairs:
        digest.update(f"{cid}\x00{sha}\n".encode("utf-8"))
    digest.update(_sha(Path(root) / "frontier.json").encode("utf-8"))
    return "v1:" + digest.hexdigest()


class _Log:
    """Append-only JSONL; each record chains to the previous one through `parent`."""

    def __init__(self, path: Path):
        self.path = path
        self.last: int | None = None
        self.seq = 0

    def write(self, record: dict[str, Any]) -> None:
        self.seq += 1
        clean = redact_obj({"id": self.seq, "parent": self.last, **record})
        for key in ("text", "result", "reason", "message"):
            if isinstance(clean.get(key), str):
                clean[key] = bounded(clean[key])
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(clean, default=str) + "\n")
        self.last = self.seq


class AgentProposer:
    def __init__(self, model: Any, *, domain: str = "classification", domain_notes: str = "",
                 view: str = "full", max_tool_calls: int = 40,
                 summarizer: Callable[[str], str] | None = None, forbidden: Sequence[Path] = (),
                 objective: str = "maximize score, then minimize context tokens",
                 max_tokens: int = 4096, test_root: Path | str | None = None,
                 idle: float = 600.0, deadline: float = 3600.0,
                 clock: Callable[[], float] = time.monotonic):
        if not callable(getattr(model, "chat", None)):
            raise TypeError("proposer provider must support tool calling")
        self.model = model
        self.domain = domain
        self.domain_notes = domain_notes or ""
        self.view = view
        self.max_tool_calls = int(max_tool_calls)
        self.summarizer = summarizer
        self.extra_forbidden = [Path(p) for p in forbidden]
        self.objective = objective
        self.max_tokens = max_tokens
        self.test_root = Path(test_root) if test_root else None
        self.idle, self.deadline, self.clock = idle, deadline, clock
        self.usage: dict[str, int] = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                                      "cache_read": 0, "cache_write": 0, "reasoning": 0}

    # -- helpers -----------------------------------------------------------------------------

    def _count(self, usage: Any) -> dict[str, int]:
        self.usage["calls"] += 1
        record = {k: int(getattr(usage, k, 0) or 0)
                  for k in ("input", "output", "cache_read", "cache_write", "reasoning")}
        self.usage["input_tokens"] += record["input"]
        self.usage["output_tokens"] += record["output"]
        for key in ("cache_read", "cache_write", "reasoning"):
            self.usage[key] += record[key]
        return record

    def _split_roots(self, root: Path) -> list[Path]:
        roots = [self.test_root or root.with_name(root.name + "-test")]
        return [Path(os.path.abspath(r)) for r in roots]

    def _in_split(self, target: Path, splits: Sequence[Path]) -> bool:
        for candidate in {Path(os.path.abspath(target)), target.resolve()}:
            for split in splits:
                for s in {split, split.resolve()}:
                    if candidate == s or s in candidate.parents:
                        return True
        return False

    def before_tool(self, sandbox: ToolSandbox, call: ToolCall,
                    splits: Sequence[Path]) -> tuple[str, bool] | None:
        """Deny any path outside the root or forbidden; (error text, is_split) or None to allow."""
        args = call.arguments if isinstance(call.arguments, dict) else {}
        if call.name not in READ_ONLY:
            return None
        path = args.get("path", ".")
        if not isinstance(path, str):
            return None
        lexical = Path(os.path.normpath(str(sandbox.root / path.replace("\\", "/"))))
        if self._in_split(lexical, splits):
            return SPLIT_DENIED, True
        try:
            target = sandbox.resolve(path)
        except ToolRefused as exc:
            return f"error: {redact(str(exc))}", False
        if self._in_split(target, splits):
            return SPLIT_DENIED, True
        return None

    # -- main loop ---------------------------------------------------------------------------

    def __call__(self, experience: Any, iteration: int, count: int) -> list[Path]:
        from .views import build_view

        root = Path(experience.root)
        output = root / "proposals" / f"iteration-{iteration:04d}"
        output.mkdir(parents=True, exist_ok=True)
        view_root = build_view(experience, self.view, root / "views" / f"iteration-{iteration:04d}",
                               self.summarizer)
        splits = self._split_roots(root)
        forbidden = [*splits, *self.extra_forbidden, root / "proposals"]
        if Path(view_root).resolve() == root.resolve():
            forbidden.append(root / "views")
        template = PROMPT_PATH.read_text(encoding="utf-8")
        paths: list[Path] = []
        for index in range(count):
            name = safe_id(f"candidate-{index:02d}")
            sandbox = ToolSandbox(Path(view_root), output, forbidden)
            prompt = render_prompt(template, iteration=iteration, count=count, index=index,
                                   domain=self.domain, domain_notes=self.domain_notes,
                                   max_tool_calls=self.max_tool_calls, objective=self.objective)
            paths.append(self._propose(sandbox, output, name, prompt, splits, root))
        return paths

    def _propose(self, sandbox: ToolSandbox, output: Path, name: str, prompt: str,
                 splits: Sequence[Path], root: Path) -> Path:
        log = _Log(output / f"{name}.proposer_log.jsonl")
        outcomes = {"dispatched": 0, "denied": 0, "denied_split": 0, "failed": 0, "aborted": 0}
        proposal = {"candidate": name, "fingerprint": fingerprint(root), "parents_read": []}
        parents: list[str] = []
        stop_reason = "error"
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        log.write({"role": "user", "chars": len(prompt)})
        watchdog = Watchdog(self.idle, self.deadline, clock=self.clock)
        used = 0
        length_stops = 0
        streak: dict[str, Any] = {"key": None, "n": 0, "fail": 0, "reminded": False}
        pending: list[ToolCall] = []

        def close(calls: Sequence[ToolCall], text: str, step: int) -> None:
            for call in calls:
                outcomes["aborted"] += 1
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                 "content": text, "is_error": True})
                log.write({"role": "tool", "step": step, "tool": call.name,
                           "arguments": _redact_args(call.name, call.arguments), "ok": False,
                           "executed": False, "result": text, "result_tokens": estimate_tokens(text),
                           "result_chars": len(text), "path": _path_of(call)})

        step = 0
        try:
            while used < self.max_tool_calls:
                step += 1
                watchdog.check()
                try:
                    turn = self.model.chat(SYSTEM, messages, TOOL_SPECS, max_tokens=self.max_tokens)
                except Exception as exc:
                    log.write({"event": "error", "role": "event", "step": step,
                               "message": f"{type(exc).__name__}: {exc}"})
                    raise
                watchdog.kick()
                usage = self._count(turn.usage)
                calls = list(turn.tool_calls or [])
                messages.append({"role": "assistant", "text": turn.text, "tool_calls": calls})
                log.write({"role": "assistant", "step": step, "tool": None, "text": turn.text or "",
                           "tool_calls": [c.name for c in calls], "usage": usage,
                           "stop_reason": turn.stop_reason})
                if turn.stop_reason == "length":
                    length_stops += 1
                    used += 1
                    log.write({"event": "length", "role": "event", "step": step})
                    close(calls, TRUNCATED, step)
                    if not calls:
                        messages.append({"role": "user", "content": TRUNCATED[len("error: "):]})
                    if length_stops >= MAX_LENGTH_STOPS:
                        raise ProposerTruncated(f"{MAX_LENGTH_STOPS} consecutive truncated turns")
                    continue
                length_stops = 0
                if not calls:
                    used += 1
                    messages.append({"role": "user", "content": NUDGE})
                    continue
                written: Path | None = None
                turn_tokens = 0
                reminder: str | None = None
                pending = list(calls)
                while pending:
                    call = pending[0]
                    position = len(calls) - len(pending)
                    if position >= MAX_CALLS_PER_TURN:
                        close(pending, NOT_EXECUTED_CAP, step)
                        pending = []
                        break
                    if used >= self.max_tool_calls:
                        close(pending, NOT_EXECUTED_BUDGET, step)
                        pending = []
                        break
                    watchdog.check()
                    pending.pop(0)
                    used += 1
                    denial = self.before_tool(sandbox, call, splits)
                    executed = denial is None
                    if denial is not None:
                        result, is_split = denial
                        outcomes["denied"] += 1
                        if is_split:
                            outcomes["denied_split"] += 1
                        got = None
                    else:
                        outcomes["dispatched"] += 1
                        try:
                            result, got = dispatch(sandbox, call, name)
                        except Exception as exc:  # a tool bug never ends the loop
                            log.write({"event": "error", "role": "event", "step": step,
                                       "message": f"{type(exc).__name__}: {exc}"})
                            result, got = f"error: {type(exc).__name__}: {redact(str(exc))}", None
                        if result.startswith("error:"):
                            outcomes["failed"] += 1
                        tokens = estimate_tokens(result)
                        if turn_tokens + tokens > TURN_RESULT_TOKENS and got is None:
                            result = BATCH_POINTER
                            tokens = estimate_tokens(result)
                        turn_tokens += tokens
                        if got is not None:
                            written = got
                    ok = not result.startswith("error:")
                    if ok and call.name in READ_ONLY:
                        cid = _candidate_of(_path_of(call))
                        if cid and cid not in parents:
                            parents.append(cid)
                    messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                                     "content": result, **({} if ok else {"is_error": True})})
                    log.write({"role": "tool", "step": step, "tool": call.name,
                               "arguments": _redact_args(call.name, call.arguments), "ok": ok,
                               "executed": executed, "denied": not executed,
                               "result_tokens": estimate_tokens(result),
                               "result_chars": len(result), "path": _path_of(call)})
                    if written is not None:
                        close(pending, NOT_EXECUTED_WRITE, step)
                        pending = []
                        break
                    if executed:
                        reminder = self._stuck(streak, call, ok, log, step) or reminder
                if written is not None:
                    stop_reason = "done"
                    return written
                if reminder:
                    messages.append({"role": "user", "content": reminder})
            stop_reason = "step_limit"
            raise ProposerBudgetExceeded(f"no candidate after {self.max_tool_calls} tool turns")
        except Aborted as exc:
            close(pending, f"error: not executed - aborted ({exc.reason})", step)
            log.write({"event": "aborted", "role": "event", "step": step, "reason": exc.reason})
            stop_reason = "timeout"
            raise
        except ProposerStuck:
            stop_reason = "stuck"
            raise
        finally:
            proposal["parents_read"] = parents
            proposal["stop_reason"] = stop_reason
            (output / f"{name}.outcomes.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")
            (output / f"{name}.proposal.json").write_text(json.dumps(redact_obj(proposal), indent=2),
                                                         encoding="utf-8")

    @staticmethod
    def _stuck(streak: dict[str, Any], call: ToolCall, ok: bool, log: _Log, step: int) -> str | None:
        """Track byte-identical (tool, args) streaks; remind once at 5, stop at 8 identical failures."""
        key = call.name + "\x00" + json.dumps(call.arguments, sort_keys=True, default=str)
        if key != streak["key"]:
            streak.update(key=key, n=0, fail=0, reminded=False)
        streak["n"] += 1
        streak["fail"] = streak["fail"] + 1 if not ok else 0
        if streak["fail"] >= STUCK_STOP:
            log.write({"event": "stop", "role": "event", "step": step, "stop_reason": "stuck"})
            raise ProposerStuck(f"{STUCK_STOP} identical failing calls to {call.name}")
        if not streak["reminded"] and (streak["n"] >= STUCK_REMIND or streak["fail"] >= STUCK_REMIND):
            streak["reminded"] = True
            text = REMIND_FAILING if streak["fail"] >= STUCK_REMIND else REMIND_REPEATING
            log.write({"event": "reminder", "role": "event", "step": step, "text": text})
            return text
        return None


def _path_of(call: ToolCall) -> Any:
    args = call.arguments if isinstance(call.arguments, dict) else {}
    value = args.get("path")
    return bounded(value, 500) if isinstance(value, str) else value


def _candidate_of(path: Any) -> str | None:
    if not isinstance(path, str):
        return None
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    if len(parts) >= 2 and parts[0] == "candidates":
        return safe_id(parts[1])
    return None


__all__ = ["AgentProposer", "PROMPT_PATH", "ProposerBudgetExceeded", "ProposerStuck",
           "ProposerTruncated", "SYSTEM", "fingerprint", "render_prompt"]
