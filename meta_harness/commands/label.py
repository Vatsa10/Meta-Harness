"""`meta-harness label`: mark imported runs good / bad in the terminal, for judge calibration."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, TextIO

NAME = "label"
HELP = "label imported runs good/bad (resumable) so the judge can be calibrated to your marks"

WRITE_HINT = re.compile(r"(?i)(write|create|update|delete|remove|send|post|put|patch|insert|upsert|set|"
                        r"refund|cancel|charge|pay|transfer|book|submit|publish|commit|deploy|email)")
CHOICES = {"g": "good", "b": "bad", "s": "skip"}
SHOW = 600


def add_arguments(parser) -> None:
    parser.add_argument("--in", dest="inp", required=True, metavar="PATH", help="tasks JSONL (traces/langsmith import)")
    parser.add_argument("--out", required=True, metavar="PATH", help="labels JSONL; existing labels are kept and skipped")
    parser.add_argument("--n", type=int, default=30, help="label at most N runs this session (default 30)")


def is_write(call: dict) -> bool:
    """A tool call that changes the world: flagged by the capture layer, or named like a write."""
    if call.get("captured") or call.get("write") or call.get("side_effect"):
        return True
    return bool(WRITE_HINT.search(str(call.get("name", ""))))


def _short(value: Any, limit: int = SHOW) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f"...[+{len(text) - limit}]"


def final_output(task: dict) -> str:
    for key in ("output", "final_output", "reference_output"):
        if task.get(key) not in (None, ""):
            return task[key] if isinstance(task[key], str) else _short(task[key])
    return ""


def render(task: dict, index: int, total: int) -> str:
    lines = [f"--- run {index}/{total}  id={task.get('id')}  ({task.get('failure', '?')})",
             "INPUT:", "  " + _short(task.get("input", "")), "FINAL OUTPUT:", "  " + _short(final_output(task))]
    calls = task.get("tool_calls") or []
    if calls:
        lines.append("TOOL ACTIONS:")
        for call in calls:
            mark = "  >>> WRITE " if is_write(call) else "  -         "
            line = f"{mark}{call.get('name')}({_short(call.get('args', {}), 200)})"
            if call.get("error"):
                line += f" !! error: {_short(call['error'], 200)}"
            else:
                line += f" -> {_short(call.get('output'), 200)}"
            lines.append(line)
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_loop(tasks: list[dict], out: Path, n: int, stdin: TextIO, stdout: TextIO) -> dict:
    done = {row.get("id") for row in _read_jsonl(out)}
    todo = [task for task in tasks if task.get("id") not in done]
    counts = {"good": 0, "bad": 0, "skip": 0}
    out.parent.mkdir(parents=True, exist_ok=True)
    quit_ = False
    for index, task in enumerate(todo[:max(n, 0)], start=1):
        stdout.write(render(task, index, min(len(todo), n)) + "\n")
        choice = None
        while choice is None:
            stdout.write("[g]ood / [b]ad / [s]kip / [q]uit: ")
            stdout.flush()
            line = stdin.readline()
            if not line:  # EOF behaves like quit; everything answered so far is saved
                quit_ = True
                break
            key = line.strip().lower()[:1]
            if key == "q":
                quit_ = True
                break
            choice = CHOICES.get(key)
            if choice is None:
                stdout.write("please answer g, b, s or q\n")
        if quit_:
            break
        reason = ""
        if choice != "skip":
            stdout.write("reason (optional, one line): ")
            stdout.flush()
            reason = stdin.readline().strip()
        row = {"id": task.get("id"), "label": choice, "reason": reason, "input": task.get("input", ""),
               "output": final_output(task), "criteria": task.get("criteria", ""),
               "tool_calls": task.get("tool_calls") or []}
        with out.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts[choice] += 1
    remaining = len(todo) - sum(counts.values())
    return {**counts, "remaining": remaining, "out": str(out)}


def run(args, stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    try:
        tasks = _read_jsonl(Path(args.inp))
    except (OSError, ValueError) as exc:
        print(f"cannot read {args.inp}: {exc}", file=sys.stderr)
        return 2
    try:
        summary = label_loop(tasks, Path(args.out), args.n, stdin, stdout)
    except ValueError as exc:
        print(f"cannot read {args.out}: {exc}", file=sys.stderr)
        return 2
    stdout.write("\n" + json.dumps(summary, ensure_ascii=False) + "\n")
    return 0
