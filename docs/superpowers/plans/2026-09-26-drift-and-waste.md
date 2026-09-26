# Drift and Waste Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the tool calls a developer loses to work that was going the wrong way, by measuring that cost from their own transcripts and warning them mid-stretch instead of leaving them to notice.

**Architecture:** Three phases over one shared experience store. Phase 1 makes failure causes legible and ships `meta-harness waste`, the offline report that is both the shopfront and the tuning set. Phase 2 builds three drift judges scored offline against 98 real corrections, behind a precision gate that can refuse to ship them. Phase 3 adds live control — rejection memory, session-scoped rules, commands — and corrects docs that misattribute claims to the paper.

**Tech Stack:** Python 3.10+ stdlib only. TypeScript function hooks (`hooks/*.ts`), Node 22 `--experimental-strip-types` for hook tests. pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-drift-and-waste-design.md`

## Global Constraints

- Python stdlib only. `dependencies = []` in `pyproject.toml` stays empty. No new dependencies, no build step.
- Every `subprocess.run(..., text=True)` must also pass `encoding="utf-8", errors="replace"`. A repo-wide AST test enforces this.
- Every file read and write passes an explicit `encoding="utf-8"`. Windows default is cp1252 and raises on the first non-ASCII byte.
- Windows-compatible. No hardcoded POSIX paths. Use `pathlib`.
- Every hook path fails open: a throwing judge, a corrupt store, a missing config all leave the turn proceeding. The one deliberate exception is a `tool.check` denial, which is the feature working.
- Transcripts never leave the machine. Stored records keep signatures and counts, not message text, unless the user passed `--include-text`.
- Sibling TypeScript imports use `.js` specifiers (the runtime resolves them to `.ts`). The test-only resolver at `hooks/loaders/` stays as it is.
- Tests must discriminate: every new test is verified to FAIL against a deliberately broken implementation before it is committed, and the mutation plus its failure message is recorded in the task report.
- No AI attribution anywhere: commits, authors, co-authors, comments, docs, contributor lists. No `Co-Authored-By` trailers.
- One commit per task minimum; more is fine. Conventional-commit subjects.
- Full suite (`python -m pytest -q`) passes before every commit. 247 tests pass at plan start.

## Review Focus

These are the conditions the spec implies but does not enumerate. Each has a test in the task that owns the code.

1. **Empty or absent transcript store** — a brand-new user has no `~/.claude/projects`. Every report, judge and bootstrap must produce an honest empty result, never a crash or a divide-by-zero. (Tasks 3, 4, 9, 12)
2. **A single enormous session** — this machine has a 23,671-turn session. Analysis must stream per session and never hold all sessions' turns at once. (Task 2)
3. **Malformed or truncated JSONL** — a session being written while we read it ends mid-line. One bad line must skip, not abort the file. (Task 2)
4. **Non-ASCII in error text under cp1252** — error text contains box-drawing and emoji. Reading and reporting must not raise. (Tasks 1, 4)
5. **Missing or skewed timestamps** — sessions carry `started`/`ended` that may be absent or unparseable. Temporal weighting must degrade to "no decay", never to a negative or infinite weight. (Task 5)

---

# Phase 1 — Experience and measurement

### Task 1: Cause coverage to 85%

**Files:**
- Modify: `meta_harness/replay.py` (`ERROR_PATTERNS`)
- Test: `tests/test_replay_signature.py`

**Interfaces:**
- Consumes: `_cause(text: str) -> str` (exists)
- Produces: `ERROR_PATTERNS` extended. New slugs: `needs-approval`, `user-rejected`, `blocked-policy`, `shell-quoting`, `compound-shell`, `tab-target`, `stale-read`, `is-directory`.

**Ordering note:** `ERROR_PATTERNS` is ordered and the order is load-bearing — the first match wins. `UnicodeEncodeError` must stay ahead of the decode pattern. New patterns go **after** the existing nine so no existing classification changes.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_replay_signature.py`:

```python
import pytest
from meta_harness.replay import _cause


@pytest.mark.parametrize("text,expected", [
    ("This command requires approval", "needs-approval"),
    ("Permission for this action was denied by the Claude Code auto mode classifier", "needs-approval"),
    ("The user doesn't want to proceed with this tool use. The tool use was rejected", "user-rejected"),
    ("<tool_use_error>Blocked: sleep 45 followed by: echo waited", "blocked-policy"),
    ("bash: -c: line 1: unexpected EOF while looking for matching `'", "shell-quoting"),
    ("Contains simple_expansion", "shell-quoting"),
    ("Compound command changes working directory (Set-Location)", "compound-shell"),
    ("This PowerShell command contains multiple operations. The following part requires approval", "compound-shell"),
    ("Tab 3 is not in Claude's tab group for this session", "tab-target"),
    ("Couldn't determine which page this action targets", "tab-target"),
    ("File has been modified since read, either by the user or by a linter", "stale-read"),
    ("EISDIR: illegal operation on a directory, read", "is-directory"),
])
def test_dominant_failure_classes_get_their_own_cause(text, expected):
    assert _cause(text) == expected


def test_existing_causes_are_unchanged_by_the_new_patterns():
    # The compound-shell pattern mentions approval; it must not steal plain permission errors,
    # and the encode/decode ordering stays load-bearing.
    assert _cause("Permission denied") == "permission"
    assert _cause("UnicodeEncodeError: 'charmap' codec") == "unicode-encode"
    assert _cause("No such file or directory") == "missing-path"


def test_cause_survives_non_ascii_error_text():
    assert _cause("bash: ─────: command not found \U0001f600") == "missing-command"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_replay_signature.py -q`
Expected: FAIL — the parametrized cases return `other`.

- [ ] **Step 3: Write minimal implementation**

In `meta_harness/replay.py`, append to `ERROR_PATTERNS` **after** the existing nine entries:

```python
    # Classes that dominate real Claude Code history. Appended after the originals so the
    # first-match rule cannot change any existing classification.
    (r"contains multiple operations|Compound command changes working directory", "compound-shell"),
    (r"requires approval|denied by the Claude Code auto mode classifier", "needs-approval"),
    (r"doesn't want to proceed|tool use was rejected", "user-rejected"),
    (r"Blocked:|blocked by a deny rule", "blocked-policy"),
    (r"unexpected EOF while looking|simple_expansion|expansion obfuscation", "shell-quoting"),
    (r"not in Claude's tab group|determine which page this action targets", "tab-target"),
    (r"modified since read", "stale-read"),
    (r"EISDIR|illegal operation on a directory", "is-directory"),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_replay_signature.py -q`
Expected: PASS

- [ ] **Step 5: Measure coverage against the acceptance target**

Run:

```bash
python -c "
import json, collections
from meta_harness.replay import _cause
eps=[json.loads(l) for l in open('.meta-harness/history-text/episodes.jsonl',encoding='utf-8')]
te=[e for e in eps if e['kind']=='tool_error']
named=sum(1 for e in te if _cause(e.get('assistant_text','')) != 'other')
print(f'coverage {named}/{len(te)} = {named/len(te):.0%}')
"
```

Expected: at least 85%. If below, inspect the remaining `other` texts and add patterns for the largest classes only — do not add a catch-all. Record the final number in the task report.

If `.meta-harness/history-text/` is absent, regenerate it first with
`python -m meta_harness mine --limit 400 --include-text --out .meta-harness/history-text`.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/replay.py tests/test_replay_signature.py
git commit -m "feat(replay): classify the failure classes that actually dominate"
```

---

### Task 2: Stretch extraction

**Files:**
- Create: `meta_harness/waste.py`
- Test: `tests/test_waste_stretches.py`

**Interfaces:**
- Produces:
  - `@dataclass Stretch: session_id: str, project: str, start_turn: int, calls: list[str], paths: list[str], ended_by: str, correction_text: str`
    — `ended_by` is one of `"correction"`, `"user"`, `"end"`.
  - `iter_stretches(path: Path, min_calls: int = 3) -> Iterator[Stretch]` — streams one transcript file.
  - `CORRECTION_RE: re.Pattern[str]`

A **stretch** is a run of assistant tool calls with no intervening user text. `ended_by="correction"`
when the user text that ends it matches `CORRECTION_RE`. Stretches shorter than `min_calls` are
not yielded.

This task streams one file at a time and never accumulates turns across sessions — Review Focus
item 2. A `json.JSONDecodeError` on one line skips that line only — Review Focus item 3.

- [ ] **Step 1: Write the failing test**

Create `tests/test_waste_stretches.py`:

```python
import json
from pathlib import Path

from meta_harness.waste import Stretch, iter_stretches


def write_transcript(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "session.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def assistant(*tools):
    content = [{"type": "tool_use", "id": f"t{i}", "name": n, "input": inp}
               for i, (n, inp) in enumerate(tools)]
    return {"type": "assistant", "message": {"role": "assistant", "content": content}}


def user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def test_a_run_of_tool_calls_between_user_messages_is_one_stretch(tmp_path):
    path = write_transcript(tmp_path, [
        user("add a flag"),
        assistant(("Read", {"file_path": "a.py"}), ("Edit", {"file_path": "a.py"})),
        assistant(("Bash", {"command": "pytest"})),
        user("thanks"),
    ])
    stretches = list(iter_stretches(path, min_calls=3))
    assert len(stretches) == 1
    assert stretches[0].calls == ["Read", "Edit", "Bash"]
    assert stretches[0].ended_by == "user"


def test_a_correction_is_labelled_and_carries_its_text(tmp_path):
    path = write_transcript(tmp_path, [
        user("add a flag"),
        assistant(("Read", {"file_path": "a.py"}), ("Edit", {"file_path": "a.py"}),
                  ("Bash", {"command": "pytest"})),
        user("no, that's not what I asked for"),
    ])
    stretch = next(iter_stretches(path, min_calls=3))
    assert stretch.ended_by == "correction"
    assert "not what I asked" in stretch.correction_text


def test_short_stretches_are_not_yielded(tmp_path):
    path = write_transcript(tmp_path, [
        user("hi"), assistant(("Read", {"file_path": "a.py"})), user("no, wrong"),
    ])
    assert list(iter_stretches(path, min_calls=3)) == []


def test_paths_are_collected_from_tool_input(tmp_path):
    path = write_transcript(tmp_path, [
        user("go"),
        assistant(("Read", {"file_path": "src/a.py"}), ("Edit", {"file_path": "src/b.py"}),
                  ("Bash", {"command": "ls src"})),
        user("ok"),
    ])
    stretch = next(iter_stretches(path, min_calls=3))
    assert "src/a.py" in stretch.paths and "src/b.py" in stretch.paths


def test_one_malformed_line_does_not_abort_the_file(tmp_path):
    path = tmp_path / "session.jsonl"
    good = [user("go"),
            assistant(("Read", {"file_path": "a"}), ("Edit", {"file_path": "b"}),
                      ("Bash", {"command": "c"})),
            user("no, wrong")]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(good[0]) + "\n")
        handle.write('{"type": "assistant", "message": {"content": [trunc\n')  # truncated write
        handle.write(json.dumps(good[1]) + "\n")
        handle.write(json.dumps(good[2]) + "\n")
    stretch = next(iter_stretches(path, min_calls=3))
    assert stretch.ended_by == "correction"


def test_a_missing_file_yields_nothing_rather_than_raising(tmp_path):
    assert list(iter_stretches(tmp_path / "absent.jsonl")) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_waste_stretches.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.waste'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/waste.py`:

```python
"""What a developer loses to work that was going the wrong way.

Reads Claude Code transcripts directly rather than through cc_history, because the unit here
is the stretch - a run of tool calls with no human input - which cc_history does not model.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

# A human saying the work went the wrong way. A keyword heuristic: it will both over- and
# under-count, which every report built on it has to say out loud.
CORRECTION_RE = re.compile(
    r"\b(no,|not what|wrong|don'?t do|stop|revert|undo|why did you|i said|actually,"
    r"|that'?s not|instead of|you broke|doesn'?t work|still (broken|failing))",
    re.I,
)

PATH_KEYS = ("file_path", "path", "notebook_path")


@dataclass
class Stretch:
    """A run of assistant tool calls with no human input, and how it ended."""

    session_id: str
    project: str
    start_turn: int
    calls: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)
    ended_by: str = "end"          # correction | user | end
    correction_text: str = ""


def _user_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(block.get("text", "") for block in content
                        if isinstance(block, dict) and block.get("type") == "text")
    return ""


def _paths_in(tool_input: object) -> list[str]:
    if not isinstance(tool_input, dict):
        return []
    found = [str(tool_input[key]) for key in PATH_KEYS if tool_input.get(key)]
    command = tool_input.get("command")
    if isinstance(command, str):
        found.extend(re.findall(r"[\w./\\-]+\.[A-Za-z]{1,4}\b", command)[:4])
    return found


def iter_stretches(path: Path, min_calls: int = 3) -> Iterator[Stretch]:
    """Stream one transcript, yielding each stretch of at least `min_calls` tool calls.

    Streams line by line and holds only the current stretch, because a single session on this
    machine reaches 23,000 turns. A malformed line is skipped, not fatal: a session may be
    written while we read it.
    """
    path = Path(path)
    session_id = path.stem
    project = path.parent.name
    current = Stretch(session_id, project, 0)
    turn = 0
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            message = record.get("message")
            if not isinstance(message, dict):
                continue
            turn += 1
            role = message.get("role")
            content = message.get("content")

            if role == "user":
                text = _user_text(message).strip()
                if text:
                    if len(current.calls) >= min_calls:
                        current.ended_by = "correction" if CORRECTION_RE.search(text[:400]) else "user"
                        current.correction_text = text[:400] if current.ended_by == "correction" else ""
                        yield current
                    current = Stretch(session_id, project, turn)
                    continue

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if not current.calls:
                            current.start_turn = turn
                        current.calls.append(str(block.get("name", "?")))
                        current.paths.extend(_paths_in(block.get("input")))

    if len(current.calls) >= min_calls:
        yield current
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_waste_stretches.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Verify the tests discriminate**

Break the implementation one way at a time, confirm the named test fails, then restore with
`git checkout -- meta_harness/waste.py` and confirm `git status --short` is clean:

1. Delete the `except ValueError: continue` — `test_one_malformed_line_does_not_abort_the_file` must fail.
2. Change `min_calls` handling to yield every stretch — `test_short_stretches_are_not_yielded` must fail.
3. Make `ended_by` always `"user"` — `test_a_correction_is_labelled_and_carries_its_text` must fail.

Record each mutation and its failure message in the task report.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/waste.py tests/test_waste_stretches.py
git commit -m "feat(waste): model the stretch, the unit of wrong-direction work"
```

---

### Task 3: Waste aggregation

**Files:**
- Modify: `meta_harness/waste.py`
- Test: `tests/test_waste_report.py`

**Interfaces:**
- Consumes: `Stretch`, `iter_stretches` (Task 2); `_cause` from `meta_harness.replay` (Task 1)
- Produces: `waste_report(home: Path | None = None, project: str | None = None, limit: int | None = None, since: str = "") -> dict[str, Any]`

`since` is an ISO date; a transcript whose mtime predates it is skipped. An unparseable `since`
is ignored rather than fatal, and the report says nothing was filtered.

Report shape, fixed here because Task 4 and Task 11 both read it:

```python
{
  "sessions": int, "tool_calls": int, "stretches": int,
  "corrections": {"count": int, "median": int, "p75": int, "p90": int, "max": int, "calls_burned": int},
  "repeats": {"wasted_retries": int, "sessions_affected": int,
              "top": [{"cause": str, "tool": str, "wasted": int}]},
  "by_project": [{"project": str, "corrections": int, "calls_burned": int}],
  "caveat": str,
}
```

`corrections` percentiles use nearest-rank on the sorted lag list. An empty history returns the
same shape with zeros and an empty `top`/`by_project` — Review Focus item 1.

- [ ] **Step 1: Write the failing test**

Create `tests/test_waste_report.py`:

```python
import json
from pathlib import Path

import pytest

from meta_harness.waste import waste_report


def make_home(tmp_path: Path, sessions: dict[str, list[dict]]) -> Path:
    home = tmp_path / "projects"
    for name, records in sessions.items():
        directory = home / "proj"
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return home


def assistant(*names):
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "id": f"t{i}", "name": n, "input": {"file_path": "a.py"}}
        for i, n in enumerate(names)]}}


def user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def test_empty_history_reports_zeros_not_a_crash(tmp_path):
    report = waste_report(home=tmp_path / "nothing")
    assert report["sessions"] == 0
    assert report["corrections"]["count"] == 0
    assert report["corrections"]["median"] == 0
    assert report["by_project"] == []


def test_correction_lag_is_the_number_of_calls_in_the_stretch(tmp_path):
    home = make_home(tmp_path, {"s1": [
        user("go"), assistant("Read", "Edit", "Bash", "Read"), user("no, wrong"),
    ]})
    report = waste_report(home=home)
    assert report["corrections"]["count"] == 1
    assert report["corrections"]["median"] == 4
    assert report["corrections"]["calls_burned"] == 4


def test_percentiles_use_nearest_rank(tmp_path):
    # lags 3,4,5,6,100 -> median 5, max 100
    sessions = {}
    for i, extra in enumerate([0, 1, 2, 3, 97]):
        sessions[f"s{i}"] = [user("go"), assistant(*(["Read"] * (3 + extra))), user("no, wrong")]
    report = waste_report(home=make_home(tmp_path, sessions))
    assert report["corrections"]["median"] == 5
    assert report["corrections"]["max"] == 100


def test_a_stretch_that_ends_normally_is_not_counted_as_a_correction(tmp_path):
    home = make_home(tmp_path, {"s1": [user("go"), assistant("Read", "Edit", "Bash"), user("thanks")]})
    assert waste_report(home=home)["corrections"]["count"] == 0
    assert waste_report(home=home)["stretches"] == 1


def test_the_report_states_that_its_correction_count_is_a_heuristic(tmp_path):
    report = waste_report(home=tmp_path / "nothing")
    assert "heuristic" in report["caveat"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_waste_report.py -q`
Expected: FAIL — `ImportError: cannot import name 'waste_report'`

- [ ] **Step 3: Write minimal implementation**

Append to `meta_harness/waste.py`:

```python
import collections
from typing import Any

from .cc_history import claude_home, is_artifact_project
from .replay import _cause

CAVEAT = ("Correction counts come from a keyword heuristic over your own messages. It will both "
          "over- and under-count. Treat this as a cost estimate, not an audit.")


def _percentile(values: list[int], fraction: float) -> int:
    """Nearest-rank percentile. Empty input is 0, which is what an empty history should report."""
    if not values:
        return 0
    index = min(len(values) - 1, int(len(values) * fraction))
    return values[index]


def _transcripts(home: Path, project: str | None) -> list[Path]:
    if not home.exists():
        return []
    found: list[Path] = []
    for directory in sorted(home.iterdir()):
        if not directory.is_dir() or is_artifact_project(directory.name):
            continue
        if project and project not in directory.name:
            continue
        found.extend(sorted(directory.glob("*.jsonl")))
    return found


def waste_report(home: Path | None = None, project: str | None = None,
                 limit: int | None = None, since: str = "") -> dict[str, Any]:
    """What wrong-direction work and repeated failure cost, across the transcript store."""
    root = Path(home) if home is not None else claude_home() / "projects"
    paths = _transcripts(root, project)
    if since:
        try:
            cutoff = datetime.fromisoformat(since).timestamp()
        except ValueError:
            cutoff = 0.0          # an unusable date filters nothing rather than everything
        if cutoff:
            paths = [p for p in paths if p.stat().st_mtime >= cutoff]
    if limit:
        paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]

    lags: list[int] = []
    per_project: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"corrections": 0, "calls_burned": 0})
    repeats: collections.Counter[tuple[str, str]] = collections.Counter()
    sessions_with_repeat = 0
    calls = 0
    stretches = 0

    for path in paths:
        seen: collections.Counter[tuple[str, str]] = collections.Counter()
        for stretch in iter_stretches(path):
            stretches += 1
            calls += len(stretch.calls)
            if stretch.ended_by == "correction":
                lags.append(len(stretch.calls))
                bucket = per_project[stretch.project]
                bucket["corrections"] += 1
                bucket["calls_burned"] += len(stretch.calls)
        for key, count in _repeat_signatures(path).items():
            if count > 1:
                seen[key] += count - 1
        if seen:
            sessions_with_repeat += 1
            repeats.update(seen)

    lags.sort()
    return {
        "sessions": len(paths),
        "tool_calls": calls,
        "stretches": stretches,
        "corrections": {
            "count": len(lags),
            "median": _percentile(lags, 0.5),
            "p75": _percentile(lags, 0.75),
            "p90": _percentile(lags, 0.90),
            "max": lags[-1] if lags else 0,
            "calls_burned": sum(lags),
        },
        "repeats": {
            "wasted_retries": sum(repeats.values()),
            "sessions_affected": sessions_with_repeat,
            "top": [{"tool": tool, "cause": cause, "wasted": n}
                    for (tool, cause), n in repeats.most_common(10)],
        },
        "by_project": sorted(
            ({"project": name, **counts} for name, counts in per_project.items()),
            key=lambda row: row["calls_burned"], reverse=True),
        "caveat": CAVEAT,
    }


def _repeat_signatures(path: Path) -> dict[tuple[str, str], int]:
    """(tool, cause) -> occurrences, for error results in one transcript."""
    counts: collections.Counter[tuple[str, str]] = collections.Counter()
    names: dict[str, str] = {}
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    with handle:
        for line in handle:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            content = (record.get("message") or {}).get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use":
                    names[str(block.get("id"))] = str(block.get("name", "?"))
                elif block.get("type") == "tool_result" and block.get("is_error"):
                    body = block.get("content")
                    if isinstance(body, list):
                        body = " ".join(part.get("text", "") for part in body
                                        if isinstance(part, dict))
                    tool = names.get(str(block.get("tool_use_id")), "?")
                    counts[(tool, _cause(str(body)))] += 1
    return dict(counts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_waste_report.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Verify the tests discriminate**

Mutate, confirm the named test fails, `git checkout --`, confirm clean:

1. Make `_percentile` return `values[int(len(values) * fraction)]` without the `min` clamp — `test_percentiles_use_nearest_rank` must fail with `IndexError`.
2. Count every stretch as a correction — `test_a_stretch_that_ends_normally_is_not_counted_as_a_correction` must fail.
3. Return `{}` from `waste_report` when history is empty — `test_empty_history_reports_zeros_not_a_crash` must fail with `KeyError`.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/waste.py tests/test_waste_report.py
git commit -m "feat(waste): aggregate correction lag and repeated failure"
```

---

### Task 4: `meta-harness waste` command

**Files:**
- Modify: `meta_harness/__main__.py`
- Test: `tests/test_waste_cli.py`

**Interfaces:**
- Consumes: `waste_report` (Task 3)
- Produces: subcommand `waste` with `--this-project`, `--since`, `--limit`, `--json`; handler `_command_waste(args) -> int`

The human-readable rendering must print the caveat. `--json` prints `waste_report`'s dict verbatim
so Task 11 can consume it. Non-ASCII in a project name or error text must not raise on Windows —
Review Focus item 4: write with `sys.stdout.reconfigure(errors="replace")` guarded by `hasattr`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_waste_cli.py`:

```python
import json

import pytest

from meta_harness.__main__ import build_parser, main


def test_waste_json_prints_the_report_verbatim(tmp_path, capsys):
    code = main(["waste", "--json", "--home", str(tmp_path / "nothing")])
    captured = json.loads(capsys.readouterr().out)
    assert code == 0
    assert captured["sessions"] == 0
    assert "corrections" in captured


def test_waste_human_output_states_the_caveat(tmp_path, capsys):
    main(["waste", "--home", str(tmp_path / "nothing")])
    assert "heuristic" in capsys.readouterr().out.lower()


def test_waste_accepts_the_documented_flags():
    parser = build_parser()
    args = parser.parse_args(["waste", "--this-project", "--limit", "5", "--json"])
    assert args.this_project and args.limit == 5 and args.json
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_waste_cli.py -q`
Expected: FAIL — `argument command: invalid choice: 'waste'`

- [ ] **Step 3: Write minimal implementation**

In `build_parser`, alongside the existing subparsers:

```python
    waste = sub.add_parser("waste", help="what wrong-direction work and repeat failure cost")
    waste.add_argument("--this-project", action="store_true",
                       help="only sessions whose project matches this working directory")
    waste.add_argument("--since", default="", help="ISO date; ignore sessions older than this")
    waste.add_argument("--limit", type=int, default=None, help="most recent N sessions")
    waste.add_argument("--json", action="store_true", help="print the report as JSON")
    waste.add_argument("--home", default="", help="transcript root (defaults to ~/.claude/projects)")
```

And the handler:

```python
def _command_waste(args: argparse.Namespace) -> int:
    from .cc_history import project_slug
    from .waste import waste_report

    if hasattr(sys.stdout, "reconfigure"):   # a project name or error may not be cp1252-safe
        sys.stdout.reconfigure(errors="replace")

    project = project_slug(Path.cwd()) if args.this_project else None
    home = Path(args.home) if args.home else None
    report = waste_report(home=home, project=project, limit=args.limit, since=args.since)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    corrections = report["corrections"]
    print(f"sessions {report['sessions']}   tool calls {report['tool_calls']:,}")
    print()
    print("wrong-direction work")
    print(f"  corrections                {corrections['count']}")
    print(f"  calls burned before you spoke  median {corrections['median']}"
          f"  p75 {corrections['p75']}  p90 {corrections['p90']}  max {corrections['max']}")
    print(f"  total calls burned         {corrections['calls_burned']:,}")
    print()
    print("repeated identical failures")
    print(f"  wasted retries             {report['repeats']['wasted_retries']:,}"
          f"  (in {report['repeats']['sessions_affected']} sessions)")
    for row in report["repeats"]["top"][:5]:
        print(f"    {row['wasted']:5d}  {row['tool']}:{row['cause']}")
    if report["by_project"]:
        print()
        print("worst projects by calls burned")
        for row in report["by_project"][:5]:
            print(f"    {row['calls_burned']:5d}  {row['project']}  ({row['corrections']} corrections)")
    print()
    print(report["caveat"])
    return 0
```

Wire `waste` into the dispatch table beside the existing commands.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_waste_cli.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Smoke-test against real history**

Run: `python -m meta_harness waste --limit 50`
Expected: a report naming a correction count and a median. Paste the output into the task report.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/__main__.py tests/test_waste_cli.py
git commit -m "feat(cli): meta-harness waste"
```

---

### Task 5: Temporal weighting

**Files:**
- Create: `meta_harness/temporal.py`
- Modify: `meta_harness/learn.py` (`merge_failures`)
- Test: `tests/test_temporal.py`

**Interfaces:**
- Produces:
  - `HALF_LIFE_DAYS: float = 30.0`
  - `recency_weight(timestamp: str, now: datetime | None = None, half_life_days: float = HALF_LIFE_DAYS) -> float` — 1.0 for now, 0.5 at one half-life, floor 0.05. **An unparseable or empty timestamp returns 1.0** (no decay), never 0 — Review Focus item 5.
  - `version_weight(seen: str, current: str) -> float` — 1.0 same minor, 0.6 one minor apart, 0.3 further, 1.0 if either is unknown.
- Consumes in `learn.py`: `FailureClass` gains no new field; `merge_failures` gains `weights: Mapping[str, float] | None = None`, defaulting to `None` meaning every weight is 1.0, so the existing two-argument call site keeps working.

- [ ] **Step 1: Write the failing test**

Create `tests/test_temporal.py`:

```python
from datetime import datetime, timedelta, timezone

import pytest

from meta_harness.temporal import HALF_LIFE_DAYS, recency_weight, version_weight

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def test_recent_evidence_keeps_full_weight():
    assert recency_weight(NOW.isoformat(), now=NOW) == pytest.approx(1.0, abs=0.01)


def test_weight_halves_at_one_half_life():
    old = (NOW - timedelta(days=HALF_LIFE_DAYS)).isoformat()
    assert recency_weight(old, now=NOW) == pytest.approx(0.5, abs=0.02)


def test_ancient_evidence_floors_rather_than_reaching_zero():
    ancient = (NOW - timedelta(days=3650)).isoformat()
    assert recency_weight(ancient, now=NOW) == pytest.approx(0.05, abs=0.001)


@pytest.mark.parametrize("stamp", ["", "not-a-date", "2026-13-45T99:99:99"])
def test_an_unusable_timestamp_means_no_decay_not_no_weight(stamp):
    # A session with no usable clock must not have its evidence silently deleted.
    assert recency_weight(stamp, now=NOW) == 1.0


def test_a_future_timestamp_does_not_exceed_full_weight():
    ahead = (NOW + timedelta(days=100)).isoformat()
    assert recency_weight(ahead, now=NOW) <= 1.0


@pytest.mark.parametrize("seen,current,expected", [
    ("2.1.278", "2.1.278", 1.0),
    ("2.1.278", "2.2.001", 0.6),
    ("2.1.233", "2.4.000", 0.3),
    ("", "2.1.278", 1.0),
    ("2.1.278", "", 1.0),
])
def test_version_distance_discounts_stale_evidence(seen, current, expected):
    assert version_weight(seen, current) == pytest.approx(expected)


def test_merge_applies_weights_without_breaking_the_unweighted_call():
    from meta_harness.learn import FailureClass, merge_failures

    observed = [FailureClass(signature="tool_error:Bash:timeout", count=10, episodes=[])]
    mined = [FailureClass(signature="tool_error:Edit:unread-edit", count=6, episodes=[])]

    unweighted = merge_failures(observed, mined)
    assert unweighted[0].signature == "tool_error:Bash:timeout"

    weighted = merge_failures(observed, mined, weights={"tool_error:Bash:timeout": 0.1})
    assert weighted[0].signature == "tool_error:Edit:unread-edit"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_temporal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.temporal'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/temporal.py`:

```python
"""Evidence ages. A failure class fixed six months ago should not outrank this week's.

Weights are multiplicative and never zero: old evidence is quieter, never deleted. A session
with no usable clock or version keeps full weight, because absence of a timestamp is not
evidence of staleness.
"""
from __future__ import annotations

from datetime import datetime, timezone

HALF_LIFE_DAYS = 30.0
WEIGHT_FLOOR = 0.05


def _parse(stamp: str) -> datetime | None:
    if not stamp:
        return None
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def recency_weight(timestamp: str, now: datetime | None = None,
                   half_life_days: float = HALF_LIFE_DAYS) -> float:
    """1.0 for now, halving every `half_life_days`, floored so nothing vanishes entirely."""
    seen = _parse(timestamp)
    if seen is None:
        return 1.0
    moment = now or datetime.now(timezone.utc)
    days = (moment - seen).total_seconds() / 86400.0
    if days <= 0:                       # a clock ahead of ours is not extra credible
        return 1.0
    return max(WEIGHT_FLOOR, 0.5 ** (days / half_life_days))


def _minor(version: str) -> tuple[int, int] | None:
    parts = version.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def version_weight(seen: str, current: str) -> float:
    """Discount evidence from a distant Claude Code version: the tool may have changed."""
    left, right = _minor(seen), _minor(current)
    if left is None or right is None:
        return 1.0
    distance = abs(left[0] - right[0]) * 100 + abs(left[1] - right[1])
    if distance == 0:
        return 1.0
    return 0.6 if distance == 1 else 0.3
```

In `meta_harness/learn.py`, change `merge_failures` to accept optional weights:

```python
def merge_failures(observed: Sequence[FailureClass],
                   mined: Sequence[FailureClass],
                   weights: Mapping[str, float] | None = None) -> list[FailureClass]:
```

and multiply each signature's summed count by `weights.get(signature, 1.0)` before sorting, when
`weights` is not None. Keep the existing `(-count, signature)` tie-break so equal weights order
identically to before.

- [ ] **Step 3b: Build the weights at the one call site that has the sessions**

`merge_failures` accepts weights but nothing computes them. `select_target` in
`meta_harness/learn.py` is the only caller that holds the `Session` objects, so it builds the
dict there:

```python
def _signature_weights(sessions: Sequence[Session]) -> dict[str, float]:
    """Age each signature by the newest session that showed it, and by version distance."""
    from .temporal import recency_weight, version_weight

    current = max((s.version for s in sessions if s.version), default="")
    weights: dict[str, float] = {}
    for session in sessions:
        stamp = session.ended or session.started
        weight = recency_weight(stamp) * version_weight(session.version, current)
        for episode in failure_episodes(session):
            signature = episode_signature(episode)
            weights[signature] = max(weights.get(signature, 0.0), weight)
    return weights
```

and passes `weights=_signature_weights(sessions)` to `merge_failures`. Add a test asserting a
signature seen only in old sessions ranks below an equal-count signature seen this week.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_temporal.py -q`
Expected: PASS (12 tests)

- [ ] **Step 5: Verify the tests discriminate**

1. Make `_parse` failure return weight `0.0` — `test_an_unusable_timestamp_means_no_decay_not_no_weight` must fail.
2. Remove the `max(WEIGHT_FLOOR, ...)` clamp — `test_ancient_evidence_floors_rather_than_reaching_zero` must fail.
3. Ignore the `weights` argument in `merge_failures` — `test_merge_applies_weights_without_breaking_the_unweighted_call` must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/temporal.py meta_harness/learn.py tests/test_temporal.py
git commit -m "feat(temporal): age evidence so old failure classes stop dominating"
```

---

### Task 6: Artifact retirement

**Files:**
- Modify: `meta_harness/temporal.py`
- Modify: `meta_harness/__main__.py` (`learn --status` output)
- Test: `tests/test_retirement.py`

**Interfaces:**
- Consumes: `Artifact`, `HarnessStore` from `meta_harness.harness_store`
- Produces: `retirement_candidates(installed: Sequence[Artifact], live_signatures: Set[str], now: datetime | None = None, quiet_days: float = 90.0) -> list[tuple[Artifact, str]]` — returns `(artifact, reason)` pairs.

Retirement is **proposed, never performed**. An artifact that works suppresses its own evidence,
so silence is ambiguous; the human decides. `learn --status` grows a `retirement_candidates` key.

- [ ] **Step 1: Write the failing test**

Create `tests/test_retirement.py`:

```python
from datetime import datetime, timedelta, timezone

from meta_harness.harness_store import Artifact
from meta_harness.temporal import retirement_candidates

NOW = datetime(2026, 9, 26, tzinfo=timezone.utc)


def artifact(artifact_id: str, signature: str, created: datetime) -> Artifact:
    return Artifact(id=artifact_id, type="rule", origin={"signature": signature},
                    payload="x", replay={}, scores={}, sources=[], created=created.isoformat())


def test_an_artifact_whose_failure_stopped_appearing_is_proposed():
    old = artifact("a1", "tool_error:Bash:timeout", NOW - timedelta(days=200))
    candidates = retirement_candidates([old], live_signatures=set(), now=NOW)
    assert [a.id for a, _ in candidates] == ["a1"]


def test_an_artifact_whose_failure_still_appears_is_kept():
    old = artifact("a1", "tool_error:Bash:timeout", NOW - timedelta(days=200))
    assert retirement_candidates([old], {"tool_error:Bash:timeout"}, now=NOW) == []


def test_a_young_artifact_is_never_proposed_however_quiet():
    young = artifact("a2", "tool_error:Bash:timeout", NOW - timedelta(days=5))
    assert retirement_candidates([young], live_signatures=set(), now=NOW) == []


def test_the_reason_names_why_so_a_human_can_judge_it():
    old = artifact("a1", "tool_error:Bash:timeout", NOW - timedelta(days=200))
    _, reason = retirement_candidates([old], set(), now=NOW)[0]
    assert "tool_error:Bash:timeout" in reason and "90" in reason


def test_an_artifact_with_an_unparseable_created_date_is_not_proposed():
    stray = Artifact(id="a3", type="rule", origin={"signature": "s"}, payload="x",
                     replay={}, scores={}, sources=[], created="not-a-date")
    assert retirement_candidates([stray], set(), now=NOW) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_retirement.py -q`
Expected: FAIL — `ImportError: cannot import name 'retirement_candidates'`

- [ ] **Step 3: Write minimal implementation**

Append to `meta_harness/temporal.py`:

```python
from collections.abc import Sequence, Set as AbstractSet

QUIET_DAYS = 90.0


def retirement_candidates(installed: "Sequence[object]", live_signatures: AbstractSet[str],
                          now: datetime | None = None,
                          quiet_days: float = QUIET_DAYS) -> list[tuple[object, str]]:
    """Installed artifacts whose origin failure has not been seen for `quiet_days`.

    Proposed, never performed. An artifact that is doing its job prevents the very evidence
    that would justify keeping it, so silence is ambiguous and a human has to decide.
    """
    moment = now or datetime.now(timezone.utc)
    proposed: list[tuple[object, str]] = []
    for artifact in installed:
        signature = (getattr(artifact, "origin", {}) or {}).get("signature", "")
        if not signature or signature in live_signatures:
            continue
        created = _parse(getattr(artifact, "created", "") or "")
        if created is None:
            continue                     # no usable date is not evidence of staleness
        age_days = (moment - created).total_seconds() / 86400.0
        if age_days >= quiet_days:
            proposed.append((artifact, f"{signature} has not been seen in "
                                       f"{quiet_days:.0f} days (installed {age_days:.0f} days ago)"))
    return proposed
```

In `_command_learn`'s `--status` branch, add to the printed dict:

```python
        "retirement_candidates": [
            {"id": a.id, "reason": reason}
            for a, reason in retirement_candidates(store.list_installed(), live_signatures)
        ],
```

where `live_signatures` is the set of signatures in the current ranking.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_retirement.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Verify the tests discriminate**

1. Drop the `signature in live_signatures` guard — `test_an_artifact_whose_failure_still_appears_is_kept` must fail.
2. Drop the `age_days >= quiet_days` guard — `test_a_young_artifact_is_never_proposed_however_quiet` must fail.
3. Make an unparseable date count as infinitely old — `test_an_artifact_with_an_unparseable_created_date_is_not_proposed` must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/temporal.py meta_harness/__main__.py tests/test_retirement.py
git commit -m "feat(temporal): propose retirement for artifacts whose failure went quiet"
```

---

### Task 7: First-run bootstrap

**Files:**
- Modify: `hooks/harness.ts`
- Test: `hooks/harness.bootstrap.test.mts`, `tests/test_hook_bootstrap.py`

**Interfaces:**
- Consumes: `harnessHome`, `safely` (existing in `hooks/harness.ts`)
- Produces: `registerBootstrap(on: On): void`, registered from `register` alongside the existing three.

Behaviour: on the first `prompt.section` of the first session after install, if
`<harnessHome>/bootstrap.json` is absent, write it and return one line naming the measured cost
and the command. On every later session the file exists and the handler returns `{text: null}`
immediately. It never runs a subprocess and never blocks; the numbers come from a report written
by `meta-harness waste --json` if present, and the line is omitted entirely if it is not.

- [ ] **Step 1: Write the failing test**

Create `hooks/harness.bootstrap.test.mts` following the shape of `hooks/harness.injection.test.mts`
(fake `$` recording `fs` calls, driving the real registered handler):

```ts
// Asserts the first session gets exactly one line and later sessions get none.
// Run: node --experimental-strip-types --import ./hooks/loaders/preload.mjs hooks/harness.bootstrap.test.mts <tmpdir>
```

Assertions required:
1. With no `bootstrap.json` and a `waste.json` present, the returned section text contains the
   correction count and the string `harness waste`.
2. `bootstrap.json` is written as a side effect.
3. With `bootstrap.json` already present, the handler returns `{text: null}` and writes nothing.
4. With no `waste.json`, the handler returns `{text: null}` and still writes `bootstrap.json`,
   so a machine with no history is not asked again every session.
5. A throwing `fs.read` leaves the turn proceeding and returns `{text: null}`.

Add `tests/test_hook_bootstrap.py` shelling out to it, following
`tests/test_hook_injection.py` exactly, including
`subprocess.run(..., text=True, encoding="utf-8", errors="replace")` and a `pytest.skip` with a
stated reason when `node` is missing.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_bootstrap.py -q`
Expected: FAIL — `registerBootstrap` is not exported.

- [ ] **Step 3: Write minimal implementation**

In `hooks/harness.ts`, add `registerBootstrap` wrapped in `safely`, reading
`<harnessHome>/waste.json`, writing `<harnessHome>/bootstrap.json`, and returning a single line:

```
Analyzed <sessions> sessions. ~<calls_burned> tool calls went to wrong-direction work. `/harness waste` for the breakdown.
```

Register it in `register` alongside `registerObserver`, `registerRules` and `registerInjection`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_bootstrap.py -q`
Expected: PASS

- [ ] **Step 5: Verify the tests discriminate**

1. Always return the line regardless of `bootstrap.json` — assertion 3 must fail.
2. Never write `bootstrap.json` — assertion 2 must fail.
3. Remove the `safely` wrapper — assertion 5 must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add hooks/harness.ts hooks/harness.bootstrap.test.mts tests/test_hook_bootstrap.py
git commit -m "feat(hooks): say once what wrong-direction work has cost"
```

---

# Phase 2 — Drift detection

### Task 8: Stretch features and the path-overlap judge

**Files:**
- Create: `meta_harness/drift.py`
- Test: `tests/test_drift_heuristic.py`

**Interfaces:**
- Consumes: `Stretch` (Task 2)
- Produces:
  - `@dataclass Verdict: drifting: bool, score: float, reason: str`
  - `anchor_paths(stretch: Stretch, anchor_calls: int = 3) -> set[str]`
  - `judge_overlap(stretch: Stretch, at_call: int, min_calls: int = 8, anchor_calls: int = 3) -> Verdict`

`judge_overlap` returns `drifting=False` when fewer than `min_calls` calls have happened — the
detector is biased toward long silent stretches, because that is where the measured cost is
(p90 = 41) and short stretches are where false positives annoy.

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_heuristic.py`:

```python
from meta_harness.drift import Verdict, anchor_paths, judge_overlap
from meta_harness.waste import Stretch


def stretch(calls, paths):
    return Stretch("s", "p", 0, calls=list(calls), paths=list(paths))


def test_a_short_stretch_never_drifts_however_different():
    s = stretch(["Read"] * 4, ["a.py", "b.py", "z.py", "q.py"])
    assert judge_overlap(s, at_call=4, min_calls=8).drifting is False


def test_work_that_stays_on_the_anchor_files_is_not_drifting():
    s = stretch(["Read"] * 12, ["a.py"] * 12)
    assert judge_overlap(s, at_call=12, min_calls=8).drifting is False


def test_work_that_has_left_the_anchor_files_entirely_is_drifting():
    s = stretch(["Read"] * 12, ["a.py", "a.py", "a.py"] + ["far/away.py"] * 9)
    verdict = judge_overlap(s, at_call=12, min_calls=8)
    assert verdict.drifting is True
    assert "a.py" in verdict.reason or "overlap" in verdict.reason


def test_a_stretch_with_no_paths_at_all_is_not_called_drifting():
    # Shell-only work has no file anchor; claiming drift here would be guessing.
    s = stretch(["Bash"] * 12, [])
    assert judge_overlap(s, at_call=12, min_calls=8).drifting is False


def test_anchor_is_taken_from_the_first_calls_only():
    s = stretch(["Read"] * 10, ["first.py", "first.py", "first.py"] + ["later.py"] * 7)
    assert anchor_paths(s, anchor_calls=3) == {"first.py"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drift_heuristic.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.drift'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/drift.py` with `Verdict`, `anchor_paths` (the distinct paths seen in the
first `anchor_calls` calls) and `judge_overlap`: compute the fraction of paths seen since the
anchor that appear in the anchor set; `drifting` when that fraction is below `0.2` and at least
`min_calls` calls have passed and the anchor is non-empty. `reason` names the anchor and the
overlap fraction.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_drift_heuristic.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Verify the tests discriminate**

1. Drop the `min_calls` guard — `test_a_short_stretch_never_drifts_however_different` must fail.
2. Treat an empty anchor as total drift — `test_a_stretch_with_no_paths_at_all_is_not_called_drifting` must fail.
3. Build the anchor from all paths — `test_anchor_is_taken_from_the_first_calls_only` must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/drift.py tests/test_drift_heuristic.py
git commit -m "feat(drift): judge by overlap with the anchor the request established"
```

---

### Task 9: Temporal kNN judge

**Files:**
- Modify: `meta_harness/drift.py`
- Test: `tests/test_drift_knn.py`

**Interfaces:**
- Consumes: `Stretch`, `Verdict`, `iter_stretches`
- Produces:
  - `shape(stretch: Stretch, at_call: int | None = None, n: int = 2) -> collections.Counter[str]` — tool-name n-gram counts, the similarity feature.
  - `build_index(home: Path | None = None, limit: int | None = None) -> list[tuple[Counter[str], bool]]` — `(shape, ended_in_correction)` over history.
  - `judge_knn(stretch: Stretch, at_call: int, index: Sequence[tuple[Counter[str], bool]], k: int = 9, min_calls: int = 8, threshold: float = 0.6) -> Verdict`

Similarity is cosine over n-gram counts. `drifting` when the weighted share of the k nearest
neighbours that ended in a correction is at or above `threshold`. **An empty index returns
`drifting=False`** — every new install has no history, and a detector that fires on no evidence
is a nag. Review Focus item 1.

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_knn.py`:

```python
import collections

from meta_harness.drift import build_index, judge_knn, shape
from meta_harness.waste import Stretch


def stretch(calls):
    return Stretch("s", "p", 0, calls=list(calls))


def test_shape_counts_tool_name_ngrams():
    counted = shape(stretch(["Read", "Edit", "Read"]), n=2)
    assert counted["Read>Edit"] == 1 and counted["Edit>Read"] == 1


def test_an_empty_index_never_reports_drift():
    # A fresh install has no history. Firing here would be guessing.
    assert judge_knn(stretch(["Read"] * 12), at_call=12, index=[]).drifting is False


def test_a_shape_like_past_corrections_is_drifting():
    bad = shape(stretch(["Edit", "Bash", "Edit", "Bash"] * 3))
    index = [(bad, True)] * 8 + [(shape(stretch(["Read"] * 12)), False)]
    verdict = judge_knn(stretch(["Edit", "Bash"] * 6), at_call=12, index=index, k=9)
    assert verdict.drifting is True
    assert "correction" in verdict.reason


def test_a_shape_like_past_clean_stretches_is_not_drifting():
    good = shape(stretch(["Read"] * 12))
    index = [(good, False)] * 8 + [(shape(stretch(["Edit", "Bash"] * 6)), True)]
    assert judge_knn(stretch(["Read"] * 12), at_call=12, index=index, k=9).drifting is False


def test_a_short_stretch_never_drifts():
    index = [(shape(stretch(["Edit", "Bash"] * 6)), True)] * 9
    assert judge_knn(stretch(["Edit", "Bash"]), at_call=2, index=index, min_calls=8).drifting is False


def test_build_index_over_an_absent_home_is_empty_not_an_error(tmp_path):
    assert build_index(home=tmp_path / "nothing") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drift_knn.py -q`
Expected: FAIL — `ImportError: cannot import name 'shape'`

- [ ] **Step 3: Write minimal implementation**

Append `shape`, `build_index` and `judge_knn` to `meta_harness/drift.py`. Cosine similarity over
the two counters; take the `k` highest; `drifting` when the similarity-weighted fraction of
correction-ending neighbours is `>= threshold`, the index is non-empty and `at_call >= min_calls`.
`reason` names the neighbour count and the fraction.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_drift_knn.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Verify the tests discriminate**

1. Treat an empty index as full drift — `test_an_empty_index_never_reports_drift` must fail.
2. Ignore the similarity weighting and count raw neighbours — `test_a_shape_like_past_clean_stretches_is_not_drifting` must fail.
3. Drop the `min_calls` guard — `test_a_short_stretch_never_drifts` must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/drift.py tests/test_drift_knn.py
git commit -m "feat(drift): judge by how similar past stretches ended"
```

---

### Task 10: Model judge with fallback

**Files:**
- Modify: `meta_harness/drift.py`
- Test: `tests/test_drift_model.py`

**Interfaces:**
- Produces: `judge_model(stretch: Stretch, at_call: int, request: str, complete: Callable[[str], str] | None, fallback: Callable[[], Verdict]) -> Verdict`

`complete` is injected, never imported, so the test needs no model. **When `complete` is `None`,
or raises, or returns anything but a leading `DRIFT:` / `OK:` token, the fallback verdict is
returned** — the hook capability is unverified on this machine and the fallback is what makes
depending on it safe.

The prompt carries the request text and the tool names and path basenames only. **Never file
contents.** A test asserts this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_drift_model.py`:

```python
import pytest

from meta_harness.drift import Verdict, judge_model
from meta_harness.waste import Stretch


def stretch():
    return Stretch("s", "p", 0, calls=["Read", "Edit"] * 6,
                   paths=["/secret/path/config.py"] * 12)


def fallback():
    return Verdict(drifting=False, score=0.0, reason="fallback")


def test_absent_capability_falls_back():
    assert judge_model(stretch(), 12, "add a flag", None, fallback).reason == "fallback"


def test_a_raising_model_falls_back():
    def boom(prompt): raise RuntimeError("no model here")
    assert judge_model(stretch(), 12, "add a flag", boom, fallback).reason == "fallback"


def test_an_unparseable_answer_falls_back():
    assert judge_model(stretch(), 12, "r", lambda p: "I think maybe?", fallback).reason == "fallback"


@pytest.mark.parametrize("answer,expected", [("DRIFT: gone to unrelated files", True), ("OK: on task", False)])
def test_a_parseable_answer_is_used(answer, expected):
    verdict = judge_model(stretch(), 12, "add a flag", lambda p: answer, fallback)
    assert verdict.drifting is expected and verdict.reason != "fallback"


def test_the_prompt_carries_no_file_contents_and_no_full_paths():
    seen = {}
    def capture(prompt):
        seen["prompt"] = prompt
        return "OK: fine"
    judge_model(stretch(), 12, "add a flag", capture, fallback)
    assert "/secret/path/" not in seen["prompt"]
    assert "config.py" in seen["prompt"]
    assert "add a flag" in seen["prompt"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_drift_model.py -q`
Expected: FAIL — `ImportError: cannot import name 'judge_model'`

- [ ] **Step 3: Write minimal implementation**

Append `judge_model` to `meta_harness/drift.py`. Build the prompt from the request, the tool-name
sequence and `Path(p).name` for each distinct path. Wrap the call in `try/except Exception`.
Parse only a leading `DRIFT:` or `OK:`; anything else falls back.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_drift_model.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Verify the tests discriminate**

1. Remove the `try/except` — `test_a_raising_model_falls_back` must fail.
2. Treat any answer containing "drift" as a verdict — `test_an_unparseable_answer_falls_back` must fail.
3. Put full paths in the prompt — `test_the_prompt_carries_no_file_contents_and_no_full_paths` must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add meta_harness/drift.py tests/test_drift_model.py
git commit -m "feat(drift): optional model judge that degrades to the heuristic"
```

---

### Task 11: Tuning and the ship gate

**Files:**
- Create: `tools/tune_drift.py`
- Test: `tests/test_tune_drift.py`

**Interfaces:**
- Consumes: `iter_stretches`, `judge_overlap`, `judge_knn`, `build_index`
- Produces:
  - `evaluate(judge_name: str, home: Path | None = None, limit: int | None = None) -> dict[str, Any]` returning `{"judge", "fired", "corrections", "caught", "recall", "precision", "median_calls_saved", "ships"}`
  - `PRECISION_BAR: float = 0.5` and `MIN_CALLS_SAVED: int = 3`
  - `main(argv: list[str] | None = None) -> int`

**The gate, stated before tuning:** a judge ships only when `precision >= PRECISION_BAR` **and**
`median_calls_saved >= MIN_CALLS_SAVED`. `ships` is that boolean. Below the bar, the judge stays
off and the report says so. The plan does not move the bar to make a judge pass.

The index is built with leave-one-session-out: a session's own stretches are excluded from the
index used to judge it, or kNN scores itself and precision is meaningless.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tune_drift.py`:

```python
import pytest

from tools.tune_drift import MIN_CALLS_SAVED, PRECISION_BAR, gate


def test_a_judge_below_the_precision_bar_does_not_ship():
    assert gate(precision=PRECISION_BAR - 0.01, median_calls_saved=20) is False


def test_a_judge_that_fires_too_late_does_not_ship():
    assert gate(precision=0.95, median_calls_saved=MIN_CALLS_SAVED - 1) is False


def test_a_judge_clearing_both_ships():
    assert gate(precision=PRECISION_BAR, median_calls_saved=MIN_CALLS_SAVED) is True


def test_the_bar_is_a_constant_not_a_parameter_callers_can_soften():
    # The gate exists to be able to refuse. It takes no threshold argument.
    import inspect
    assert list(inspect.signature(gate).parameters) == ["precision", "median_calls_saved"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tune_drift.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.tune_drift'`

Create `tools/__init__.py` if absent.

- [ ] **Step 3: Write minimal implementation**

Create `tools/tune_drift.py` with `PRECISION_BAR`, `MIN_CALLS_SAVED`, `gate`, `evaluate` and a
`main` printing one row per judge. For each stretch of at least `min_calls`, replay the judge at
each call index; record the first call it fires. Recall is over correction-ending stretches;
precision is fired-and-correction over all fired; `median_calls_saved` is the median of
`len(stretch.calls) - first_fire` over caught corrections.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tune_drift.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the real tuning and record the numbers**

Run: `python -m tools.tune_drift --limit 400`

Paste the full table into the task report. **For each judge, state whether it ships.** If no
judge clears the gate, that is a valid outcome: record it, set the shipped default to the judge
with the best precision, and leave drift reporting disabled by default in Task 12. Do not adjust
`PRECISION_BAR` to make a judge pass — if the bar turns out to be wrong, say so in the report and
leave the decision to the final review.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add tools/tune_drift.py tests/test_tune_drift.py tools/__init__.py
git commit -m "feat(drift): tune the judges against real corrections, with a gate that can refuse"
```

---

### Task 12: Live drift surfacing

**Files:**
- Create: `hooks/drift.ts`
- Modify: `hooks/harness.ts`
- Test: `hooks/harness.drift.test.mts`, `tests/test_hook_drift.py`

**Interfaces:**
- Consumes: `safely`, `harnessHome`, `currentSessionId` from `hooks/harness.ts`
- Produces: `registerDrift(on: On): void`; config read from `<harnessHome>/drift.json` shaped
  `{"enabled": boolean, "min_calls": number, "judge": "knn"|"overlap"|"model"}`

Counts calls in a `tool.call` handler (a third one — it must call `next` exactly once, like the
existing two) and surfaces at most **one** line per stretch through `prompt.section`. Resets the
counter whenever a `prompt.section` fires, which is the hook-visible proxy for the user speaking.

A missing or corrupt `drift.json` means **disabled** — fail-open, and it is how Task 11's gate
result is honoured when no judge shipped.

- [ ] **Step 1: Write the failing test**

Create `hooks/harness.drift.test.mts` in the pattern of `hooks/harness.rules.test.mts`.
Assertions required:
1. With `{"enabled": true, "min_calls": 8}` and 10 recorded calls, `prompt.section` returns a line naming the count.
2. The same stretch does not produce a second line.
3. With `{"enabled": false}`, nothing is returned however many calls.
4. With no `drift.json`, nothing is returned.
5. With a corrupt `drift.json`, nothing is returned and the turn proceeds.
6. The `tool.call` handler calls `next` exactly once and returns its outcome unchanged.

Add `tests/test_hook_drift.py` shelling out to it, matching `tests/test_hook_rules.py` including
the encoding arguments and the `node`-missing skip with a stated reason.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_drift.py -q`
Expected: FAIL — `hooks/drift.ts` does not exist.

- [ ] **Step 3: Write minimal implementation**

Create `hooks/drift.ts` exporting `loadDriftConfig` and `shouldWarn(count, config)`, importing
from `'./rules.js'` where shared helpers are needed. Add `registerDrift` to `hooks/harness.ts`
wrapped in `safely`, register it in `register`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_drift.py -q`
Expected: PASS

- [ ] **Step 5: Verify the tests discriminate**

1. Treat a missing config as enabled — assertion 4 must fail.
2. Remove the once-per-stretch latch — assertion 2 must fail.
3. Make the `tool.call` handler skip `next` — assertion 6 must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add hooks/drift.ts hooks/harness.ts hooks/harness.drift.test.mts tests/test_hook_drift.py
git commit -m "feat(hooks): say once when a stretch has drifted"
```

---

# Phase 3 — Live control and surface

### Task 13: Rejection memory

**Files:**
- Modify: `hooks/rules.ts`, `hooks/harness.ts`
- Test: `hooks/harness.rejection.test.mts`, `tests/test_hook_rejection.py`

**Interfaces:**
- Consumes: `callKey`, `SessionState` from `hooks/rules.ts`
- Produces: `rememberRejection(state: SessionState, tool: string, input: unknown): void` and
  `wasRejected(state: SessionState, tool: string, input: unknown): boolean`

The largest single repeated failure on this machine: 98 retries of a call the human already
rejected. The observer records the rejection when a `tool_result` matches
`/doesn't want to proceed|tool use was rejected/i`; `tool.check` denies an identical re-proposal.

Memory is **session-scoped** and clears for that call key when the user's text mentions the
command again, so changing your mind works.

- [ ] **Step 1: Write the failing test**

Create `hooks/harness.rejection.test.mts`. Assertions required:
1. A call rejected once, then re-proposed identically, is denied with a reason naming the earlier rejection.
2. A *different* call is not denied.
3. After the user's text mentions the rejected command again, the same call is allowed.
4. A rejection in one session does not affect another session's state.
5. The `tool.check` handler still calls `next` on the allow path.

Add `tests/test_hook_rejection.py` in the established pattern.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_rejection.py -q`
Expected: FAIL — `rememberRejection` is not exported.

- [ ] **Step 3: Write minimal implementation**

Add a `rejected: Set<string>` to `SessionState`, keyed by `callKey`. Record in the `tool.call`
observer; check in `tool.check` before the rule loop; clear on the matching user mention.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_rejection.py -q`
Expected: PASS

- [ ] **Step 5: Verify the tests discriminate**

1. Never record the rejection — assertion 1 must fail.
2. Key the memory by tool name only — assertion 2 must fail.
3. Never clear on mention — assertion 3 must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add hooks/rules.ts hooks/harness.ts hooks/harness.rejection.test.mts tests/test_hook_rejection.py
git commit -m "feat(hooks): stop re-proposing what was already rejected"
```

---

### Task 14: Session-scoped natural-language rules

**Files:**
- Modify: `hooks/rules.ts`, `hooks/harness.ts`
- Test: `hooks/harness.nlrules.test.mts`, `tests/test_hook_nlrules.py`

**Interfaces:**
- Produces: `parseStopInstruction(text: string): {tool: string, pattern: string} | null` and
  `addSessionRule(state: SessionState, rule: {tool: string, pattern: string}): void`

"Stop doing X" / "don't run X again" takes effect **immediately, for this session only**. Nothing
is written to the installed store. At session end the pending session rules are written to
`<harnessHome>/pending-session-rules.json` so `/harness pending` can offer to keep them — writing
the *proposal*, never the installed artifact.

- [ ] **Step 1: Write the failing test**

Create `hooks/harness.nlrules.test.mts`. Assertions required:
1. `parseStopInstruction("stop running pytest without -q")` returns a rule naming `Bash` and a pattern containing `pytest`.
2. `parseStopInstruction("what does this function do?")` returns `null` — ordinary questions must not create rules.
3. A session rule added from such text denies the matching call on the next `tool.check`.
4. The session rule is **not** written to `installed.json`.
5. It is written to `pending-session-rules.json` for later review.

Add `tests/test_hook_nlrules.py` in the established pattern.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_nlrules.py -q`
Expected: FAIL — `parseStopInstruction` is not exported.

- [ ] **Step 3: Write minimal implementation**

Implement `parseStopInstruction` with a conservative leading-verb pattern
(`/^\s*(stop|don'?t|never)\s+(running|using|doing|calling)?\s*(.+)/i`) that returns `null` unless
the text starts with such an instruction. Store in `state.sessionRules`, checked in `tool.check`
after the rejection check and before the installed-rule loop.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_nlrules.py -q`
Expected: PASS

- [ ] **Step 5: Verify the tests discriminate**

1. Make `parseStopInstruction` match anywhere in the text — assertion 2 must fail.
2. Write the rule into the installed store — assertion 4 must fail.
3. Skip the session-rule check in `tool.check` — assertion 3 must fail.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add hooks/rules.ts hooks/harness.ts hooks/harness.nlrules.test.mts tests/test_hook_nlrules.py
git commit -m "feat(hooks): obey stop-doing-that for the rest of the session"
```

---

### Task 15: `/harness` commands

**Files:**
- Create: `commands/harness.md`
- Test: `tests/test_harness_commands.py`

**Interfaces:**
- Consumes: `meta-harness waste` (Task 4), `learn --status` (existing), retirement candidates (Task 6)

One command file with three documented modes: `waste`, `pending`, `why`. `pending` lists staged
artifacts, retirement candidates and `pending-session-rules.json`; `why` explains the most recent
denial or drift note and how to undo it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_harness_commands.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMMAND = ROOT / "commands" / "harness.md"


def test_the_command_file_exists_with_frontmatter():
    text = COMMAND.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "description:" in text


def test_all_three_modes_are_documented():
    text = COMMAND.read_text(encoding="utf-8")
    for mode in ("waste", "pending", "why"):
        assert f"/harness {mode}" in text


def test_every_command_it_names_exists_in_the_cli():
    from meta_harness.__main__ import build_parser
    text = COMMAND.read_text(encoding="utf-8")
    known = set(build_parser()._subparsers._group_actions[0].choices)
    import re
    for invoked in re.findall(r"python -m meta_harness (\w+)", text):
        assert invoked in known, f"{invoked} is documented but not a real subcommand"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_harness_commands.py -q`
Expected: FAIL — `FileNotFoundError: commands/harness.md`

- [ ] **Step 3: Write minimal implementation**

Create `commands/harness.md` with frontmatter (`description`, `argument-hint: waste | pending | why`)
and a body routing each mode to its command.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_harness_commands.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Verify the tests discriminate**

Change one documented invocation to `python -m meta_harness wastee` and confirm
`test_every_command_it_names_exists_in_the_cli` fails. Restore.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add commands/harness.md tests/test_harness_commands.py
git commit -m "feat(commands): /harness waste, pending and why"
```

---

### Task 16: Docs correction and version bump

**Files:**
- Modify: `README.md`, `skills/learning-from-failures/SKILL.md`, `hooks/README.md`,
  `docs/superpowers/specs/2026-09-20-learning-harness-design.md`, `.claude-plugin/plugin.json`
- Test: `tests/test_docs_claims.py`

**Interfaces:** none — documentation only, plus `version` to `0.7.0`.

Three corrections, each with a test that fails if the claim comes back:

1. The layer ordering `rule > injection > skill > doctrine` must **not** be attributed to
   `paper.pdf`. It stays, labelled as this project's design position.
2. The paper's actual findings must appear: raw traces beat summaries (Table 3: 34.6 / 34.9 / 50.0),
   and additive changes beat invasive ones (Appendix A.2, six consecutive regressions).
3. The measured negative result must be recorded: environment bootstrap does not transfer to a
   repository whose environment is already known (50 orientation calls across 344 sessions).

- [ ] **Step 1: Write the failing test**

Create `tests/test_docs_claims.py`:

```python
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = [ROOT / "README.md", ROOT / "skills" / "learning-from-failures" / "SKILL.md"]


def test_no_doc_attributes_the_layer_ordering_to_the_paper():
    # The ordering is this project's position. The paper does not contain it.
    pattern = re.compile(r"paper[^.\n]{0,80}(rule\s*>\s*injection|layer ordering)"
                         r"|(rule\s*>\s*injection)[^.\n]{0,80}paper", re.I)
    for doc in DOCS:
        assert not pattern.search(doc.read_text(encoding="utf-8")), doc


def test_the_papers_real_finding_is_stated_somewhere():
    joined = " ".join(d.read_text(encoding="utf-8") for d in DOCS)
    assert "50.0" in joined and "34.9" in joined      # the Table 3 ablation
    assert re.search(r"raw traces|execution traces", joined, re.I)


def test_the_measured_negative_result_is_recorded():
    joined = " ".join(d.read_text(encoding="utf-8") for d in DOCS)
    assert re.search(r"environment (snapshot|bootstrap)", joined, re.I)
    assert re.search(r"does not transfer|did not transfer", joined, re.I)


def test_plugin_version_bumped():
    import json
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "0.7.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_docs_claims.py -q`
Expected: FAIL on the version assertion and at least one claim assertion.

- [ ] **Step 3: Write minimal implementation**

Edit the four documents and bump `version` to `0.7.0`. In
`docs/superpowers/specs/2026-09-20-learning-harness-design.md`, add a note at the top pointing at
the new spec and marking §4's ordering as this project's position rather than the paper's.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_docs_claims.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Verify the tests discriminate**

Reintroduce the sentence "the paper's layer ordering is rule > injection > skill > doctrine" into
`README.md` and confirm `test_no_doc_attributes_the_layer_ordering_to_the_paper` fails. Remove it.
Roll `version` back to `0.6.0` and confirm `test_plugin_version_bumped` fails. Restore.

- [ ] **Step 6: Run the full suite and commit**

```bash
python -m pytest -q
git add README.md skills/ hooks/README.md docs/superpowers/specs/ .claude-plugin/plugin.json tests/test_docs_claims.py
git commit -m "docs: say what the paper found, and what we measured instead"
```

---

## Notes for the executing controller

- **Tasks 8, 9 and 10 all append to `meta_harness/drift.py`.** They must run in order; each
  consumes `Verdict` from Task 8.
- **Tasks 7, 12, 13 and 14 all modify `hooks/harness.ts`.** Sequential. Task 12 adds a *third*
  `tool.call` handler — the pre-existing two must keep working, and all three must call `next`.
- **Task 11 gates Task 12.** If no judge ships, Task 12 still builds, with `enabled` defaulting
  to false and the task report recording why.
- **Task 1 gates Tasks 3 and 9**: both consume `_cause`, and a judge trained on 80% `other` is a
  judge trained on noise.
- Parallelism: the implementer dispatches stay sequential — every task in Phase 2 and Phase 3
  shares a file with its neighbour. Reviews of completed tasks may run concurrently with the next
  task's implementation.
