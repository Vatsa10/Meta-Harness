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
from typing import Any, Callable, Iterable, Sequence

from .cache import DiskCache, default_proposer_cache_dir, proposer_cache_enabled, proposer_call_key
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
REMIND_BUDGET = ("Reminder: {left} tool calls remain for this candidate. Stop investigating and call "
                 "write_candidate now with your best hypothesis.")
VIEW_NOTES = {
    "full": "Every file in the layout below is present.",
    "scores": "This view holds only run.json, frontier.json and, per candidate, harness.py and "
              "scores.json. traces.jsonl, proposer_reasoning.md and summary.md are absent: do not "
              "list, read or grep for them. Form your hypothesis from the source and the scores.",
    "summary": "This view holds only run.json, frontier.json and, per candidate, harness.py, "
               "scores.json and summary.md. traces.jsonl and proposer_reasoning.md are absent: do "
               "not list, read or grep for them.",
    "seed": "This view holds only run.json, frontier.json and the seed candidates (initial-*), each "
            "with harness.py and scores.json. traces.jsonl, proposer_reasoning.md and summary.md "
            "are absent: do not list, read or grep for them. Read the seeds' source and scores, "
            "then write.",
}
_RUN = ("run.json                       search config and the search-split tasks\n"
        "frontier.json                  Pareto frontier of the last completed run, if any\n")
_LAYOUT_LINES = {
    "harness.py": "candidates/ID/harness.py       the candidate's full source",
    "scores.json": "candidates/ID/scores.json      score, context_cost, repeats, score_std, valid, error",
    "traces.jsonl": "candidates/ID/traces.jsonl     one JSON event per line (model_call, task_end, reward)",
    "proposer_reasoning.md": "candidates/ID/proposer_reasoning.md  why the previous proposer wrote it",
    "summary.md": "candidates/ID/summary.md       a written summary of its evaluation",
}
_VIEW_LAYOUT_FILES = {"full": ("harness.py", "scores.json", "traces.jsonl", "proposer_reasoning.md"),
                      "scores": ("harness.py", "scores.json"),
                      "summary": ("harness.py", "scores.json", "summary.md"),
                      "seed": ("harness.py", "scores.json")}
LAYOUTS = {view: _RUN + "\n".join(_LAYOUT_LINES[f] for f in files)
           for view, files in _VIEW_LAYOUT_FILES.items()}
LAYOUTS["seed"] = LAYOUTS["seed"].replace("candidates/ID/", "candidates/initial-ID/")
_DEFAULTS = {"domain_notes": "", "view_note": VIEW_NOTES["full"], "layout": LAYOUTS["full"]}


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


# -- leak tripwire -----------------------------------------------------------------------------
# Adapted from google-research/harness2 (Apache-2.0), harness2/core/context.py
# (assert_no_leak_paths, assert_no_grader_verdicts): scan the proposer's whole context tree
# before any model call and refuse it on a held-out path or a grader marker.

LEAK_REASON = "leak_tripwire"
GRADER_MARKERS: tuple[str, ...] = ("VERDICT: PASS", "VERDICT: FAIL", "test_results")
LEAK_FIELDS = ("input", "question", "prompt", "problem")
TARGET_FIELDS = ("expected", "label", "answer", "target")
# A short target ("fruit", "42") is shared with search tasks by construction; only targets this
# long are distinctive enough to hash on their own.
MIN_TARGET_CHARS = 24
_SCAN_SKIP_DIRS = ("proposals", "views")
_SCAN_MAX_BYTES = 8_000_000


class ProposerLeak(RuntimeError):
    """The view the proposer would see references held-out data; the proposal is aborted."""

    stop_reason = "error"
    reason = LEAK_REASON

    def __init__(self, detail: str):
        super().__init__(f"{LEAK_REASON}: {detail}")
        self.detail = detail


def leak_digest(value: Any) -> str:
    """sha256 of one held-out value, whitespace-trimmed; the caller passes digests, never data."""
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def heldout_digests(tasks: Sequence[Any], fields: Sequence[str] = LEAK_FIELDS,
                    targets: Sequence[str] = TARGET_FIELDS) -> frozenset[str]:
    """Digests of every held-out task (canonical JSON), its input values and its long targets."""
    out: set[str] = set()
    for task in tasks:
        out.add(leak_digest(task))
        if not isinstance(task, dict):
            continue
        for key in (*fields, *targets):
            value = task.get(key)
            if value in (None, ""):
                continue
            text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
            if key in fields or len(text.strip()) >= MIN_TARGET_CHARS:
                out.add(leak_digest(value))
    return frozenset(out)


def _strings(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def scan_for_leaks(view_root: Path | str, test_root: Path | str | None = None,
                   digests: Iterable[str] = (), markers: Sequence[str] = GRADER_MARKERS,
                   skip: Sequence[Path] = ()) -> None:
    """Raise ProposerLeak when any file visible in the view references the held-out split.

    (a) a path or a file body naming the held-out root (absolute path, basename) or test_results;
    (b) a grader marker in a body, or a JSON value / text line whose sha256 is in `digests`.
    """
    root = Path(view_root)
    if not root.is_dir():
        return
    needles: list[str] = list(markers)
    if test_root is not None:
        test = Path(os.path.abspath(test_root))
        needles += [str(test), test.as_posix(), test.name]
    needles = [n for n in dict.fromkeys(needles) if n]
    wanted = frozenset(digests)
    skipped = [Path(os.path.abspath(p)) for p in skip]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel.split("/")[0] in _SCAN_SKIP_DIRS:  # forbidden to the proposer anyway
            continue
        absolute = Path(os.path.abspath(path))
        if any(s == absolute or s in absolute.parents for s in skipped):
            continue
        for needle in needles:
            if needle in rel:
                raise ProposerLeak(f"{rel!r} names held-out marker {needle!r} in its path")
        try:
            if path.stat().st_size > _SCAN_MAX_BYTES:
                continue
            body = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for needle in needles:
            if needle in body:
                raise ProposerLeak(f"{rel!r} contains held-out marker {needle!r}")
        if not wanted:
            continue
        values: list[str] = [line for line in body.splitlines() if line.strip()]
        for line in values[:]:
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            for item in _strings(parsed):
                if isinstance(item, dict):
                    values.append(json.dumps(item, sort_keys=True, default=str))
                else:
                    values.append(item)
        try:
            for item in _strings(json.loads(body)):
                values.append(json.dumps(item, sort_keys=True, default=str)
                              if isinstance(item, dict) else item)
        except ValueError:
            pass
        for value in values:
            if value.strip() and leak_digest(value) in wanted:
                raise ProposerLeak(f"{rel!r} contains a held-out task value "
                                   f"(sha256 {leak_digest(value)[:12]})")


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
                 clock: Callable[[], float] = time.monotonic,
                 heldout_hashes: Iterable[str] = (), leak_markers: Sequence[str] = GRADER_MARKERS,
                 cache: Path | str | None = None):
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
        self.heldout_hashes = frozenset(heldout_hashes)
        self.leak_markers = tuple(leak_markers)
        if cache is None and proposer_cache_enabled():
            cache = os.environ.get("META_HARNESS_PROPOSER_CACHE_DIR") or default_proposer_cache_dir()
        self.cache = DiskCache(cache) if cache else None
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
        self._tripwire(Path(view_root), output, splits)
        template = PROMPT_PATH.read_text(encoding="utf-8")
        paths: list[Path] = []
        for index in range(count):
            name = safe_id(f"candidate-{index:02d}")
            sandbox = ToolSandbox(Path(view_root), output, forbidden)
            prompt = render_prompt(template, iteration=iteration, count=count, index=index,
                                   domain=self.domain, domain_notes=self.domain_notes,
                                   max_tool_calls=self.max_tool_calls, objective=self.objective,
                                   view_note=VIEW_NOTES.get(self.view, VIEW_NOTES["full"]),
                                   layout=LAYOUTS.get(self.view, LAYOUTS["full"]))
            key = self._cache_key(Path(view_root), root, prompt) if self.cache is not None else None
            try:
                hit = self._from_cache(key, sandbox, output, name, root) if key else None
                if hit is not None:
                    paths.append(hit)
                    continue
                paths.append(self._propose(sandbox, output, name, prompt, splits, root))
                if key:
                    self._to_cache(key, paths[-1])
            except (ProposerBudgetExceeded, ProposerStuck, ProposerTruncated) as exc:
                # One proposal giving up must not discard its siblings' written candidates.
                (output / f"{name}.error").write_text(repr(exc), encoding="utf-8")
                if index == count - 1 and not paths:
                    raise
        return paths

    # -- proposer call cache (opt-in: META_HARNESS_PROPOSER_CACHE=1) -----------------------------

    def _cache_key(self, view_root: Path, root: Path, prompt: str) -> str:
        model_id = str(getattr(self.model, "model", None) or type(self.model).__name__)
        return proposer_call_key(view_root, root, model_id, SYSTEM + "\x00" + prompt,
                                 skip=_SCAN_SKIP_DIRS)

    def _from_cache(self, key: str, sandbox: ToolSandbox, output: Path, name: str,
                    root: Path) -> Path | None:
        stored = self.cache.get(key) if self.cache is not None else None
        if not stored or not isinstance(stored.get("source"), str):
            return None
        written = sandbox.write_candidate(stored["source"], str(stored.get("rationale", "")), name)
        log = _Log(output / f"{name}.proposer_log.jsonl")
        log.write({"event": "cache_hit", "role": "event", "cache_hit": True, "key": key})
        (output / f"{name}.outcomes.json").write_text(json.dumps(
            {"dispatched": 0, "denied": 0, "denied_split": 0, "failed": 0, "aborted": 0,
             "cache_hit": 1}, indent=2), encoding="utf-8")
        proposal = {"candidate": name, "fingerprint": fingerprint(root),
                    "parents_read": list(stored.get("parents_read") or []), "stop_reason": "done",
                    "cache_hit": True}
        (output / f"{name}.proposal.json").write_text(json.dumps(redact_obj(proposal), indent=2),
                                                     encoding="utf-8")
        return written

    def _to_cache(self, key: str, written: Path) -> None:
        if self.cache is None or not written.is_file():
            return
        reasoning = written.with_name(written.stem + ".reasoning.md")
        proposal = written.with_name(written.stem + ".proposal.json")
        parents = json.loads(proposal.read_text(encoding="utf-8")).get("parents_read", []) \
            if proposal.is_file() else []
        self.cache.put(key, {"source": written.read_text(encoding="utf-8"),
                             "rationale": reasoning.read_text(encoding="utf-8")
                             if reasoning.is_file() else "", "parents_read": parents})

    def _tripwire(self, view_root: Path, output: Path, splits: Sequence[Path]) -> None:
        """Abort the whole proposal before any model call if the view leaks held-out data."""
        try:
            for split in splits:
                scan_for_leaks(view_root, split, self.heldout_hashes, self.leak_markers,
                               skip=[output, *self.extra_forbidden])
        except ProposerLeak as exc:
            (output / "proposer.leak.json").write_text(
                json.dumps({"reason": LEAK_REASON, "detail": redact(exc.detail)}, indent=2),
                encoding="utf-8")
            raise

    def _propose(self, sandbox: ToolSandbox, output: Path, name: str, prompt: str,
                 splits: Sequence[Path], root: Path) -> Path:
        log = _Log(output / f"{name}.proposer_log.jsonl")
        outcomes = {"dispatched": 0, "denied": 0, "denied_split": 0, "failed": 0, "aborted": 0}
        proposal = {"candidate": name, "fingerprint": fingerprint(root), "parents_read": []}
        parents: list[str] = []
        stop_reason = "error"
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        log.write({"role": "user", "chars": len(prompt), "cache_hit": False})
        watchdog = Watchdog(self.idle, self.deadline, clock=self.clock)
        used = 0
        length_stops = 0
        streak: dict[str, Any] = {"key": None, "n": 0, "fail": 0, "reminded": False}
        budget_reminded = False
        warn_at = max(3, self.max_tool_calls // 5)  # calls left when the proposer is told to write
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
                left = self.max_tool_calls - used
                if not budget_reminded and 0 < left <= warn_at:
                    budget_reminded = True
                    log.write({"event": "budget_reminder", "role": "event", "step": step, "left": left})
                    reminder = "\n".join(filter(None, [reminder, REMIND_BUDGET.format(left=left)]))
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


__all__ = ["AgentProposer", "GRADER_MARKERS", "LAYOUTS", "LEAK_REASON", "PROMPT_PATH", "ProposerLeak",
           "heldout_digests", "leak_digest", "scan_for_leaks", "ProposerBudgetExceeded", "ProposerStuck",
           "ProposerTruncated", "SYSTEM", "fingerprint", "render_prompt"]
