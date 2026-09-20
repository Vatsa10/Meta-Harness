# Learning Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Claude Code learns from its own failures — each failure becomes a candidate harness artifact, scored against the failure that produced it, staged for acceptance, and then enforced by mechanism rather than suggested in prose.

**Architecture:** Three layers. **Observe** (`tool.call` hook writes failures to an append-only log; `cc_history` mines the existing 37 sessions). **Learn** (`meta-harness learn` selects a failure, asks the proposer for one artifact, scores it by replay, stages it). **Enforce** (`tool.check` denies, `prompt.section` injects, skills and doctrine for what mechanism cannot express).

**Tech Stack:** Python 3.10+ standard library for the engine; TypeScript function hooks (Claude Code 2.1.274+) for observe and enforce; existing `meta_harness` search, `cc_harness`, `cc_history` modules.

**Spec:** `docs/superpowers/specs/2026-09-20-learning-harness-design.md`

## Global Constraints

- Python 3.10+, **standard library only** in `meta_harness/**`. `pyproject.toml` keeps an empty `dependencies` list.
- Every `subprocess.run` with `text=True` passes `encoding="utf-8", errors="replace"`. An AST test in `tests/test_cc_harness.py` enforces this and must keep passing.
- Windows-compatible: no symlinks, no `signal.alarm`, no POSIX-only calls. No nested PowerShell inside bash quoting.
- Hook code never calls a model, never blocks on the network, and **fails open** — a hook that throws must let the turn proceed.
- Nothing is auto-installed. `learn` stages; a human accepts.
- Every artifact carries `sources` (session id and turn) and its scores.
- All state under `~/.claude/harness/`, overridable by `META_HARNESS_HOME` so tests never touch the real directory.
- Run the suite with `python -m pytest tests -q` from the repo root.
- **Commits: one per task minimum, small and frequent.** Conventional-commit subject, body explaining why.
- **No AI attribution anywhere.** No `Co-Authored-By` naming an assistant, no "Generated with" line, no assistant name in commit messages, authors, or contributor lists.

---

## Phase 1 — The learn engine (Python)

### Task 1: Artifact store

**Files:**
- Create: `meta_harness/harness_store.py`
- Test: `tests/test_harness_store.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `harness_home() -> Path` — `META_HARNESS_HOME` or `~/.claude/harness`
  - `@dataclass Artifact` with fields `id, type, origin, payload, replay, scores, sources, created`
  - `class HarnessStore` with `stage(artifact) -> Path`, `accept(id) -> Path`, `reject(id, wrong=False) -> Path`, `list_staged() -> list[Artifact]`, `list_installed() -> list[Artifact]`, `is_tombstoned(signature) -> bool`, `installed_signatures() -> set[str]`
  - `ARTIFACT_TYPES = ("rule", "injection", "skill", "doctrine")`

- [ ] **Step 1: Write the failing test**

Create `tests/test_harness_store.py`:

```python
import json
from pathlib import Path

import pytest

from meta_harness.harness_store import Artifact, HarnessStore, harness_home


def _artifact(id="a1", type="rule", signature="tool_error:Bash:cp1252"):
    return Artifact(id=id, type=type,
                    origin={"kind": "tool_error", "signature": signature,
                            "session": "s1", "turn": 12},
                    payload="deny Edit on unread files",
                    replay={"instruction": "x", "files": {}, "expect": {}},
                    scores={"origin_fixed": True}, sources=["s1#12"], created="2026-09-20")


def test_home_respects_the_override(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path / "h"))
    assert harness_home() == tmp_path / "h"


def test_stage_then_accept_moves_and_registers(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    assert [a.id for a in store.list_staged()] == ["a1"]
    store.accept("a1")
    assert store.list_staged() == []
    assert [a.id for a in store.list_installed()] == ["a1"]
    assert "tool_error:Bash:cp1252" in store.installed_signatures()


def test_reject_archives_without_tombstoning(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    store.reject("a1")
    assert store.list_staged() == []
    assert (tmp_path / "archive" / "a1" / "artifact.json").is_file()
    assert store.is_tombstoned("tool_error:Bash:cp1252") is False


def test_reject_as_wrong_tombstones_the_signature(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    store.reject("a1", wrong=True)
    # A tombstoned signature must never be proposed again.
    assert store.is_tombstoned("tool_error:Bash:cp1252") is True


def test_payload_round_trips(tmp_path: Path):
    store = HarnessStore(tmp_path)
    store.stage(_artifact())
    store.accept("a1")
    assert store.list_installed()[0].payload == "deny Edit on unread files"


def test_accepting_an_unknown_id_raises(tmp_path: Path):
    with pytest.raises(KeyError):
        HarnessStore(tmp_path).accept("nope")


def test_rejects_an_unknown_artifact_type(tmp_path: Path):
    with pytest.raises(ValueError):
        HarnessStore(tmp_path).stage(_artifact(type="telepathy"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_harness_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.harness_store'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/harness_store.py`:

```python
"""Storage for learned harness artifacts.

Nothing here installs anything into a live session. `stage` records a scored candidate,
`accept` promotes it, `reject` archives it, and a rejection marked wrong tombstones the failure
signature so the same idea is never proposed again.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ARTIFACT_TYPES = ("rule", "injection", "skill", "doctrine")


def harness_home() -> Path:
    return Path(os.environ.get("META_HARNESS_HOME") or (Path.home() / ".claude" / "harness"))


@dataclass
class Artifact:
    id: str
    type: str
    origin: dict[str, Any]
    payload: str
    replay: dict[str, Any]
    scores: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    created: str = ""

    @property
    def signature(self) -> str:
        return str(self.origin.get("signature", ""))


class HarnessStore:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root or harness_home())
        for name in ("artifacts", "staged", "archive"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.tombstone_path = self.root / "tombstones.json"
        self.installed_path = self.root / "installed.json"

    # --- reading -----------------------------------------------------------

    def _load_dir(self, directory: Path) -> list[Artifact]:
        out = []
        for path in sorted(directory.glob("*/artifact.json")):
            try:
                out.append(Artifact(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, TypeError, ValueError):
                continue
        return out

    def list_staged(self) -> list[Artifact]:
        return self._load_dir(self.root / "staged")

    def list_installed(self) -> list[Artifact]:
        return self._load_dir(self.root / "artifacts")

    def installed_signatures(self) -> set[str]:
        return {a.signature for a in self.list_installed() if a.signature}

    def tombstones(self) -> set[str]:
        if not self.tombstone_path.is_file():
            return set()
        try:
            return set(json.loads(self.tombstone_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return set()

    def is_tombstoned(self, signature: str) -> bool:
        return signature in self.tombstones()

    def covered(self) -> set[str]:
        """Signatures that must not be proposed again: installed or tombstoned."""
        return self.installed_signatures() | self.tombstones()

    # --- writing -----------------------------------------------------------

    def _write(self, directory: Path, artifact: Artifact) -> Path:
        target = directory / artifact.id
        target.mkdir(parents=True, exist_ok=True)
        payload = dict(asdict(artifact))
        (target / "artifact.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def stage(self, artifact: Artifact) -> Path:
        if artifact.type not in ARTIFACT_TYPES:
            raise ValueError(f"unknown artifact type: {artifact.type}")
        artifact.created = artifact.created or date.today().isoformat()
        return self._write(self.root / "staged", artifact)

    def _take_staged(self, artifact_id: str) -> Artifact:
        for artifact in self.list_staged():
            if artifact.id == artifact_id:
                return artifact
        raise KeyError(f"no staged artifact {artifact_id!r}")

    def accept(self, artifact_id: str) -> Path:
        artifact = self._take_staged(artifact_id)
        target = self._write(self.root / "artifacts", artifact)
        shutil.rmtree(self.root / "staged" / artifact_id, ignore_errors=True)
        registry = []
        if self.installed_path.is_file():
            try:
                registry = json.loads(self.installed_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                registry = []
        registry = [r for r in registry if r.get("id") != artifact.id]
        registry.append({"id": artifact.id, "type": artifact.type,
                         "signature": artifact.signature, "accepted": date.today().isoformat()})
        self.installed_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
        return target

    def reject(self, artifact_id: str, wrong: bool = False) -> Path:
        artifact = self._take_staged(artifact_id)
        target = self._write(self.root / "archive", artifact)
        shutil.rmtree(self.root / "staged" / artifact_id, ignore_errors=True)
        if wrong and artifact.signature:
            stones = self.tombstones() | {artifact.signature}
            self.tombstone_path.write_text(json.dumps(sorted(stones), indent=2), encoding="utf-8")
        return target


__all__ = ["ARTIFACT_TYPES", "Artifact", "HarnessStore", "harness_home"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_harness_store.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/harness_store.py tests/test_harness_store.py
git commit -m "feat(store): artifact storage with staging, archive and tombstones

Learned artifacts are staged, never auto-installed. Rejecting one as wrong
tombstones its failure signature so the same idea is not proposed again next
cycle."
```

---

### Task 2: Failure signatures

**Files:**
- Create: `meta_harness/replay.py`
- Test: `tests/test_replay_signature.py`

**Interfaces:**
- Consumes: `meta_harness.cc_history.FailureEpisode`.
- Produces:
  - `episode_signature(episode) -> str` — stable key like `tool_error:Bash:cp1252`
  - `ERROR_PATTERNS: tuple[tuple[str, str], ...]` — (regex, slug) pairs matched against error text

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_signature.py`:

```python
from meta_harness.cc_history import FailureEpisode
from meta_harness.replay import episode_signature


def _episode(kind="tool_error", tools=("Bash",), detail="", assistant_text=""):
    return FailureEpisode(session_id="s1", project="p", kind=kind, turn_index=3,
                          detail=detail, tools=list(tools), assistant_text=assistant_text)


def test_tool_error_signature_names_tool_and_cause():
    episode = _episode(assistant_text="UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d")
    assert episode_signature(episode) == "tool_error:Bash:unicode-decode"


def test_unmatched_error_falls_back_to_a_generic_slug():
    assert episode_signature(_episode(assistant_text="something odd")) == "tool_error:Bash:other"


def test_thrash_signature_names_the_tool():
    assert episode_signature(_episode(kind="thrash", tools=("Edit",))) == "thrash:Edit"


def test_signature_is_stable_across_wording():
    a = _episode(assistant_text="UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d")
    b = _episode(assistant_text="UnicodeDecodeError: 'charmap' codec can't decode byte 0x81")
    # Two instances of the same fault must share a signature, or dedupe never fires.
    assert episode_signature(a) == episode_signature(b)


def test_correction_signature_is_session_scoped():
    episode = _episode(kind="correction", tools=("Edit", "Write"))
    assert episode_signature(episode) == "correction:Edit+Write"


def test_missing_tools_still_yields_a_signature():
    assert episode_signature(_episode(tools=())) == "tool_error:unknown:other"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_replay_signature.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.replay'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/replay.py`:

```python
"""Turning an observed failure into a replay: the smallest run that shows whether it recurs.

A failure carries its own test. A tool error passes its replay when that error class does not
occur again; a thrash episode passes when the tool is no longer hammered inside the window.
Neither needs a human-written assertion, which is what makes ~1,266 of the mined episodes usable
without a hand-built benchmark.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .cc_history import FailureEpisode

# Matched against the error text of a failed tool result, most specific first. The slug is what
# makes two instances of one fault share a signature; without it dedupe never fires.
ERROR_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"UnicodeDecodeError|charmap|cp1252", "unicode-decode"),
    (r"UnicodeEncodeError", "unicode-encode"),
    (r"No such file or directory|cannot find the (file|path)", "missing-path"),
    (r"Permission denied|EACCES", "permission"),
    (r"command not found|is not recognized as", "missing-command"),
    (r"timed out|TimeoutExpired", "timeout"),
    (r"has not been read yet|must read.*before", "unread-edit"),
    (r"String to replace not found|old_string", "edit-mismatch"),
    (r"SyntaxError|unterminated", "syntax"),
)


def _cause(text: str) -> str:
    for pattern, slug in ERROR_PATTERNS:
        if re.search(pattern, text or "", re.IGNORECASE):
            return slug
    return "other"


def episode_signature(episode: FailureEpisode) -> str:
    """A stable key for one failure class, used for dedupe and tombstones."""
    tools = list(episode.tools or [])
    if episode.kind == "thrash":
        return f"thrash:{tools[0] if tools else 'unknown'}"
    if episode.kind == "correction":
        return "correction:" + "+".join(sorted(tools)) if tools else "correction:unknown"
    tool = tools[0] if tools else "unknown"
    return f"tool_error:{tool}:{_cause(episode.assistant_text or episode.detail or '')}"


__all__ = ["ERROR_PATTERNS", "episode_signature"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_replay_signature.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/replay.py tests/test_replay_signature.py
git commit -m "feat(replay): stable failure signatures for dedupe and tombstones

Two instances of one fault must share a key or dedupe never fires, so the
error text is reduced to a cause slug rather than kept verbatim."
```

---

### Task 3: Replay expectations

**Files:**
- Modify: `meta_harness/replay.py`
- Test: `tests/test_replay_expectations.py`

**Interfaces:**
- Consumes: `agent_step` trace events written by `cc_harness.run_claude_code`.
- Produces:
  - `expectation_for(episode) -> dict` — `{"no_tool_error": {...}}` or `{"no_thrash": {...}}`
  - `verify_expectation(expectation, events) -> bool`
  - `load_agent_steps(trace_path) -> list[dict]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_expectations.py`:

```python
import json
from pathlib import Path

from meta_harness.cc_history import FailureEpisode
from meta_harness.replay import expectation_for, load_agent_steps, verify_expectation


def _episode(kind="tool_error", tools=("Bash",), text="UnicodeDecodeError: charmap"):
    return FailureEpisode(session_id="s", project="p", kind=kind, turn_index=1,
                          tools=list(tools), assistant_text=text)


def _steps(*items):
    return list(items)


def test_expectation_for_a_tool_error_names_tool_and_cause():
    assert expectation_for(_episode()) == {
        "no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}}


def test_expectation_for_thrash_carries_window_and_threshold():
    assert expectation_for(_episode(kind="thrash", tools=("Edit",))) == {
        "no_thrash": {"tool": "Edit", "window": 6, "threshold": 4}}


def test_tool_error_expectation_fails_when_the_error_recurs():
    steps = _steps({"role": "tool_result", "content": "UnicodeDecodeError: charmap codec"})
    assert verify_expectation({"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}},
                              steps) is False


def test_tool_error_expectation_passes_when_it_does_not():
    steps = _steps({"role": "tool_result", "content": "ok"})
    assert verify_expectation({"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}},
                              steps) is True


def test_an_unrelated_error_does_not_fail_the_expectation():
    steps = _steps({"role": "tool_result", "content": "Permission denied"})
    # The replay asks whether THIS fault recurred, not whether the run was flawless.
    assert verify_expectation({"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}},
                              steps) is True


def test_thrash_expectation_fails_when_the_tool_is_hammered():
    steps = _steps(*[{"role": "tool_use", "name": "Edit"} for _ in range(5)])
    assert verify_expectation({"no_thrash": {"tool": "Edit", "window": 6, "threshold": 4}},
                              steps) is False


def test_thrash_expectation_passes_below_threshold():
    steps = _steps(*[{"role": "tool_use", "name": "Edit"} for _ in range(3)])
    assert verify_expectation({"no_thrash": {"tool": "Edit", "window": 6, "threshold": 4}},
                              steps) is True


def test_load_agent_steps_reads_only_agent_step_payloads(tmp_path: Path):
    trace = tmp_path / "t.jsonl"
    trace.write_text("\n".join(json.dumps(r) for r in [
        {"event": "agent_start", "payload": {"prompt": "x"}},
        {"event": "agent_step", "payload": {"role": "tool_use", "name": "Read"}},
        {"event": "agent_end", "payload": {"turns": 3}},
    ]), encoding="utf-8")
    assert load_agent_steps(trace) == [{"role": "tool_use", "name": "Read"}]


def test_empty_expectation_is_vacuously_true():
    assert verify_expectation({}, []) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_replay_expectations.py -q`
Expected: FAIL — `ImportError: cannot import name 'expectation_for'`

- [ ] **Step 3: Write minimal implementation**

Append to `meta_harness/replay.py`:

```python
import json
from pathlib import Path

THRASH_WINDOW = 6
THRASH_THRESHOLD = 4


def expectation_for(episode: FailureEpisode) -> dict[str, Any]:
    """The mechanical check that says whether this failure recurred."""
    tools = list(episode.tools or [])
    tool = tools[0] if tools else "unknown"
    if episode.kind == "thrash":
        return {"no_thrash": {"tool": tool, "window": THRASH_WINDOW,
                              "threshold": THRASH_THRESHOLD}}
    if episode.kind == "tool_error":
        return {"no_tool_error": {"tool": tool,
                                  "cause": _cause(episode.assistant_text or episode.detail or "")}}
    return {}


def load_agent_steps(trace_path: Path | str) -> list[dict[str, Any]]:
    """The `agent_step` payloads of one replay run, in order."""
    path = Path(trace_path)
    if not path.is_file():
        return []
    steps = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or '"agent_step"' not in line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("event") == "agent_step" and isinstance(record.get("payload"), dict):
            steps.append(record["payload"])
    return steps


def _pattern_for_cause(cause: str) -> str:
    for pattern, slug in ERROR_PATTERNS:
        if slug == cause:
            return pattern
    return ""


def verify_expectation(expectation: Mapping[str, Any],
                       steps: Sequence[Mapping[str, Any]]) -> bool:
    """True when the failure this replay was built from did not recur."""
    if not expectation:
        return True

    spec = expectation.get("no_tool_error")
    if spec:
        pattern = _pattern_for_cause(str(spec.get("cause", "")))
        if not pattern:
            return True
        for step in steps:
            if step.get("role") != "tool_result":
                continue
            if re.search(pattern, str(step.get("content", "")), re.IGNORECASE):
                return False
        return True

    spec = expectation.get("no_thrash")
    if spec:
        tool = spec.get("tool")
        window = int(spec.get("window", THRASH_WINDOW))
        threshold = int(spec.get("threshold", THRASH_THRESHOLD))
        uses = [index for index, step in enumerate(steps)
                if step.get("role") == "tool_use" and step.get("name") == tool]
        for start in uses:
            inside = sum(1 for index in uses if start <= index < start + window)
            if inside >= threshold:
                return False
        return True

    return True
```

Extend the module's `__all__` to `["ERROR_PATTERNS", "THRASH_THRESHOLD", "THRASH_WINDOW",
"episode_signature", "expectation_for", "load_agent_steps", "verify_expectation"]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_replay_expectations.py -q`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/replay.py tests/test_replay_expectations.py
git commit -m "feat(replay): verify a failure from the transcript, not a test command

A cp1252 decode failure is not reproducible as a pass/fail command, but it is
checkable from the agent_step events already captured: re-run the situation
and assert that error class does not recur. Same for thrash. That is what
makes the mined episodes usable without a hand-written benchmark."
```

---

### Task 4: Building a replay task

**Files:**
- Modify: `meta_harness/replay.py`
- Test: `tests/test_replay_build.py`

**Interfaces:**
- Consumes: `cc_history.Session`, `cc_history.read_file_history`, Task 3's `expectation_for`.
- Produces: `build_replay(episode, session, home=None, max_files=4) -> dict | None` returning `{"instruction", "files", "expect", "_origin"}`, or `None` when the episode cannot be replayed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_replay_build.py`:

```python
import json
from pathlib import Path

from meta_harness.cc_history import FailureEpisode, load_sessions, parse_session
from meta_harness.replay import build_replay


def _transcript(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _assistant(tools=(), text=""):
    content = [{"type": "text", "text": text}] if text else []
    for index, name in enumerate(tools):
        content.append({"type": "tool_use", "id": f"t{index}", "name": name, "input": {}})
    return {"type": "assistant", "message": {"content": content},
            "cwd": "D:\\repo", "timestamp": "2026-09-20T10:00:00Z"}


def _human(text):
    return {"type": "user", "message": {"content": text}}


def _session(tmp_path: Path):
    path = _transcript(tmp_path / "proj" / "s1.jsonl", [
        _human("make the parser handle empty input"),
        _assistant(tools=["Bash"]),
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t0", "is_error": True,
             "content": "UnicodeDecodeError: charmap"}]}},
    ])
    return parse_session(path, include_text=True)


def _episode():
    return FailureEpisode(session_id="s1", project="proj", kind="tool_error", turn_index=2,
                          tools=["Bash"], assistant_text="UnicodeDecodeError: charmap")


def test_replay_carries_instruction_and_expectation(tmp_path: Path):
    replay = build_replay(_episode(), _session(tmp_path), home=tmp_path / "claude")
    assert replay is not None
    assert "parser" in replay["instruction"]
    assert replay["expect"] == {"no_tool_error": {"tool": "Bash", "cause": "unicode-decode"}}
    assert replay["_origin"]["session"] == "s1"


def test_replay_without_a_request_is_not_replayable(tmp_path: Path):
    path = _transcript(tmp_path / "proj" / "s2.jsonl", [_assistant(tools=["Bash"])])
    assert build_replay(_episode(), parse_session(path, include_text=True),
                        home=tmp_path / "claude") is None


def test_replay_seeds_files_from_file_history(tmp_path: Path):
    home = tmp_path / "claude"
    _transcript(home / "projects" / "proj" / "s1.jsonl", [
        {"type": "file-history-delta", "trackingPath": "src/a.py",
         "backup": repr({"backupFileName": "h1@v2", "version": 2}),
         "timestamp": "2026-09-20T10:00:00Z"},
    ])
    blob = home / "file-history" / "s1" / "h1@v2"
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_text(json.dumps("original body"), encoding="utf-8")

    replay = build_replay(_episode(), _session(tmp_path), home=home)
    assert replay["files"]["src/a.py"] == "original body"


def test_replay_caps_the_number_of_seeded_files(tmp_path: Path):
    home = tmp_path / "claude"
    deltas = []
    for index in range(8):
        deltas.append({"type": "file-history-delta", "trackingPath": f"src/f{index}.py",
                       "backup": repr({"backupFileName": f"h{index}@v2", "version": 2}),
                       "timestamp": "2026-09-20T10:00:00Z"})
        blob = home / "file-history" / "s1" / f"h{index}@v2"
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_text(json.dumps("x" * (50 + index)), encoding="utf-8")
    _transcript(home / "projects" / "proj" / "s1.jsonl", deltas)

    replay = build_replay(_episode(), _session(tmp_path), home=home, max_files=3)
    assert len(replay["files"]) == 3


def test_correction_episodes_are_not_mechanically_replayable(tmp_path: Path):
    episode = FailureEpisode(session_id="s1", project="proj", kind="correction",
                             turn_index=2, tools=["Edit"], user_text="that's wrong")
    # No mechanical expectation exists, so there is nothing to verify without a human.
    assert build_replay(episode, _session(tmp_path), home=tmp_path / "claude") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_replay_build.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_replay'`

- [ ] **Step 3: Write minimal implementation**

Append to `meta_harness/replay.py`:

```python
from .cc_history import Session, read_file_history

MAX_REPLAY_FILE_CHARS = 8000


def _request_before(session: Session, turn_index: int) -> str:
    asks = [turn.text for turn in session.turns
            if turn.role == "user" and turn.text and turn.index < turn_index]
    return asks[-1] if asks else ""


def build_replay(episode: FailureEpisode, session: Session, home: Path | None = None,
                 max_files: int = 4) -> dict[str, Any] | None:
    """The smallest run that shows whether this failure recurs, or None if it cannot be built."""
    expectation = expectation_for(episode)
    if not expectation:
        return None
    instruction = _request_before(session, episode.turn_index)
    if not instruction:
        return None

    seeded: dict[str, str] = {}
    for version in read_file_history(session.session_id, home):
        if version.previous_content is None or not version.tracking_path:
            continue
        seeded.setdefault(version.tracking_path, version.previous_content)
    smallest = sorted(seeded.items(), key=lambda kv: len(kv[1]))[:max_files]

    return {
        "instruction": instruction,
        "files": {path.replace("\\", "/"): body[:MAX_REPLAY_FILE_CHARS]
                  for path, body in smallest},
        "expect": expectation,
        "_origin": {"kind": episode.kind, "session": session.session_id,
                    "turn": episode.turn_index, "signature": episode_signature(episode)},
    }
```

Add `"MAX_REPLAY_FILE_CHARS"` and `"build_replay"` to `__all__`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_replay_build.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/replay.py tests/test_replay_build.py
git commit -m "feat(replay): build a replay task from a failure episode

The request that preceded the failure becomes the instruction and the
pre-edit file contents seed the workspace. An episode with no mechanical
expectation - a correction - returns None rather than a task nobody can score."
```

---

### Task 5: Selecting what to learn next

**Files:**
- Create: `meta_harness/learn.py`
- Test: `tests/test_learn_select.py`

**Interfaces:**
- Consumes: `cc_history.load_sessions`, `cc_history.failure_episodes`, Task 2's `episode_signature`, Task 1's `HarnessStore.covered`.
- Produces:
  - `@dataclass FailureClass` with `signature, kind, tool, count, episodes`
  - `rank_failures(sessions) -> list[FailureClass]` — most frequent first
  - `select_target(sessions, store) -> FailureClass | None` — skips covered signatures

- [ ] **Step 1: Write the failing test**

Create `tests/test_learn_select.py`:

```python
import json
from pathlib import Path

from meta_harness.cc_history import parse_session
from meta_harness.harness_store import Artifact, HarnessStore
from meta_harness.learn import rank_failures, select_target


def _transcript(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


def _error_session(tmp_path: Path, name: str, tool: str, text: str, times: int):
    records = [{"type": "user", "message": {"content": "do the thing"}}]
    for index in range(times):
        records.append({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"t{index}", "name": tool, "input": {}}]},
            "cwd": "D:\\repo", "timestamp": "2026-09-20T10:00:00Z"})
        records.append({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": f"t{index}", "is_error": True,
             "content": text}]}})
    return parse_session(_transcript(tmp_path / "proj" / f"{name}.jsonl", records),
                         include_text=True)


def test_ranks_the_most_frequent_failure_first(tmp_path: Path):
    sessions = [
        _error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 3),
        _error_session(tmp_path, "b", "Read", "Permission denied", 1),
    ]
    ranked = rank_failures(sessions)
    assert ranked[0].signature == "tool_error:Bash:unicode-decode"
    assert ranked[0].count == 3


def test_select_skips_an_installed_signature(tmp_path: Path):
    store = HarnessStore(tmp_path / "store")
    store.stage(Artifact(id="a1", type="rule",
                         origin={"signature": "tool_error:Bash:unicode-decode"},
                         payload="p", replay={}))
    store.accept("a1")
    sessions = [
        _error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 3),
        _error_session(tmp_path, "b", "Read", "Permission denied", 1),
    ]
    assert select_target(sessions, store).signature == "tool_error:Read:permission"


def test_select_skips_a_tombstoned_signature(tmp_path: Path):
    store = HarnessStore(tmp_path / "store")
    store.stage(Artifact(id="a1", type="rule",
                         origin={"signature": "tool_error:Bash:unicode-decode"},
                         payload="p", replay={}))
    store.reject("a1", wrong=True)
    sessions = [_error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 3)]
    assert select_target(sessions, store) is None


def test_select_returns_none_when_nothing_is_left(tmp_path: Path):
    assert select_target([], HarnessStore(tmp_path / "store")) is None


def test_a_failure_class_keeps_its_episodes(tmp_path: Path):
    sessions = [_error_session(tmp_path, "a", "Bash", "UnicodeDecodeError: charmap", 2)]
    assert len(rank_failures(sessions)[0].episodes) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learn_select.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'meta_harness.learn'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/learn.py`:

```python
"""The learn cycle: pick the failure that costs most, propose one artifact, score it by replay."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from .cc_history import FailureEpisode, Session, failure_episodes
from .harness_store import HarnessStore
from .replay import episode_signature


@dataclass
class FailureClass:
    signature: str
    kind: str
    tool: str
    count: int = 0
    episodes: list[FailureEpisode] = field(default_factory=list)


def rank_failures(sessions: Sequence[Session]) -> list[FailureClass]:
    """Every observed failure class, most frequent first."""
    grouped: dict[str, FailureClass] = {}
    for session in sessions:
        for episode in failure_episodes(session):
            signature = episode_signature(episode)
            tools = list(episode.tools or [])
            entry = grouped.get(signature)
            if entry is None:
                entry = FailureClass(signature=signature, kind=episode.kind,
                                     tool=tools[0] if tools else "unknown")
                grouped[signature] = entry
            entry.count += 1
            entry.episodes.append(episode)
    return sorted(grouped.values(), key=lambda f: (-f.count, f.signature))


def select_target(sessions: Sequence[Session], store: HarnessStore) -> FailureClass | None:
    """The most frequent failure not already covered by an installed artifact or a tombstone."""
    covered = store.covered()
    for failure in rank_failures(sessions):
        if failure.signature not in covered:
            return failure
    return None


__all__ = ["FailureClass", "rank_failures", "select_target"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_learn_select.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/learn.py tests/test_learn_select.py
git commit -m "feat(learn): rank failure classes and skip what is already covered

The cycle works on what actually costs time, and never re-proposes a
signature that is installed or tombstoned."
```

---

### Task 6: Proposing one artifact

**Files:**
- Modify: `meta_harness/learn.py`
- Create: `meta_harness/learn_prompt.md`
- Test: `tests/test_learn_propose.py`

**Interfaces:**
- Consumes: Task 5's `FailureClass`, Task 4's `build_replay`, `providers.model_from_environment`.
- Produces:
  - `build_proposal_prompt(failure, replay, installed) -> str`
  - `parse_proposal(text) -> tuple[str, str]` — `(artifact_type, payload)`
  - `propose_artifact(failure, replay, installed, model) -> Artifact`

- [ ] **Step 1: Write the failing test**

Create `tests/test_learn_propose.py`:

```python
import pytest

from meta_harness.cc_history import FailureEpisode
from meta_harness.learn import FailureClass, build_proposal_prompt, parse_proposal, propose_artifact

REPLAY = {"instruction": "fix it", "files": {}, "expect": {"no_tool_error": {"tool": "Bash"}},
          "_origin": {"signature": "tool_error:Bash:unicode-decode", "session": "s1", "turn": 4}}
FAILURE = FailureClass(signature="tool_error:Bash:unicode-decode", kind="tool_error",
                       tool="Bash", count=7,
                       episodes=[FailureEpisode("s1", "p", "tool_error", 4,
                                                assistant_text="UnicodeDecodeError: charmap")])


def test_prompt_states_the_layer_ordering():
    prompt = build_proposal_prompt(FAILURE, REPLAY, installed=[])
    # The ordering is the design's central rule; a proposer that does not see it will
    # write prose for something a rule could enforce.
    assert "rule" in prompt and "injection" in prompt and "doctrine" in prompt
    assert "7" in prompt          # how often the failure happened
    assert "UnicodeDecodeError" in prompt


def test_prompt_lists_installed_artifacts_to_avoid_duplication():
    prompt = build_proposal_prompt(FAILURE, REPLAY, installed=["read-before-edit"])
    assert "read-before-edit" in prompt


def test_parse_proposal_reads_type_and_payload():
    text = 'TYPE: rule\nPAYLOAD:\n```\ndeny Edit when the file was not Read\n```'
    assert parse_proposal(text) == ("rule", "deny Edit when the file was not Read")


def test_parse_proposal_accepts_an_unfenced_payload():
    assert parse_proposal("TYPE: doctrine\nPAYLOAD:\nRead before you edit.") == (
        "doctrine", "Read before you edit.")


def test_parse_proposal_rejects_an_unknown_type():
    with pytest.raises(ValueError, match="unknown artifact type"):
        parse_proposal("TYPE: telepathy\nPAYLOAD:\nx")


def test_parse_proposal_rejects_a_missing_payload():
    with pytest.raises(ValueError, match="payload"):
        parse_proposal("TYPE: rule\n")


def test_propose_artifact_carries_provenance():
    model = lambda prompt, **_: "TYPE: rule\nPAYLOAD:\n```\ndeny it\n```"
    artifact = propose_artifact(FAILURE, REPLAY, installed=[], model=model)
    assert artifact.type == "rule"
    assert artifact.payload == "deny it"
    assert artifact.origin["signature"] == "tool_error:Bash:unicode-decode"
    assert artifact.sources == ["s1#4"]
    assert artifact.replay == REPLAY
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learn_propose.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_proposal_prompt'`

- [ ] **Step 3: Write minimal implementation**

Create `meta_harness/learn_prompt.md`:

```markdown
You are proposing one harness artifact that stops a failure Claude Code keeps making.

Express the fix at the STRONGEST layer that can carry it. The ordering is binding:

| type | mechanism | can the model ignore it? |
|---|---|---|
| `rule` | a tool.check that denies the call, with a reason | No |
| `injection` | text placed into the turn when relevant | No |
| `skill` | a SKILL.md loaded by description | Yes |
| `doctrine` | a paragraph in CLAUDE.md, paid every turn forever | Yes |

Choose `rule` when the failure is decidable from the call and the session state alone.
Choose `injection` when the agent needs a fact at a particular moment.
Choose `skill` or `doctrine` only when no mechanism can express the fix.

Write ONE artifact, targeting only this failure. Do not bundle several changes: a bundled
artifact cannot be attributed when it regresses.

Answer in exactly this form:

TYPE: <rule|injection|skill|doctrine>
PAYLOAD:
```
<the artifact body>
```
```

Append to `meta_harness/learn.py`:

```python
import re
from pathlib import Path
from typing import Any, Callable, Mapping

from .harness_store import ARTIFACT_TYPES, Artifact
from .replay import build_replay  # noqa: F401  (re-exported for the CLI)

PROMPT_PATH = Path(__file__).resolve().parent / "learn_prompt.md"


def build_proposal_prompt(failure: FailureClass, replay: Mapping[str, Any],
                          installed: Sequence[str]) -> str:
    sample = failure.episodes[0] if failure.episodes else None
    evidence = (sample.assistant_text or sample.detail or "") if sample else ""
    parts = [PROMPT_PATH.read_text(encoding="utf-8"),
             "\n## The failure\n",
             f"signature: {failure.signature}",
             f"kind: {failure.kind}   tool: {failure.tool}",
             f"observed {failure.count} times",
             f"error text: {evidence[:400]}" if evidence else "",
             "\n## The replay it must fix\n",
             f"instruction: {str(replay.get('instruction', ''))[:400]}",
             f"files: {sorted((replay.get('files') or {}))}",
             f"expectation: {replay.get('expect')}"]
    if installed:
        parts.append("\n## Already installed - do not duplicate\n" + "\n".join(
            f"- {name}" for name in installed))
    return "\n".join(part for part in parts if part)


def parse_proposal(text: str) -> tuple[str, str]:
    match = re.search(r"TYPE:\s*([a-z]+)", text, re.IGNORECASE)
    if not match:
        raise ValueError("no TYPE in proposal")
    artifact_type = match.group(1).lower()
    if artifact_type not in ARTIFACT_TYPES:
        raise ValueError(f"unknown artifact type: {artifact_type}")
    body = text.split("PAYLOAD:", 1)
    if len(body) != 2 or not body[1].strip():
        raise ValueError("no payload in proposal")
    payload = body[1].strip()
    fenced = re.search(r"```[a-zA-Z]*\s*\n(.*?)```", payload, re.DOTALL)
    if fenced:
        payload = fenced.group(1)
    return artifact_type, payload.strip()


def propose_artifact(failure: FailureClass, replay: Mapping[str, Any],
                     installed: Sequence[str], model: Callable[..., str]) -> Artifact:
    artifact_type, payload = parse_proposal(model(
        build_proposal_prompt(failure, replay, installed)))
    origin = dict(replay.get("_origin") or {})
    origin.setdefault("signature", failure.signature)
    return Artifact(
        id=f"{failure.signature.replace(':', '-')}-{origin.get('session', 'x')[:8]}",
        type=artifact_type,
        origin=origin,
        payload=payload,
        replay=dict(replay),
        sources=[f"{origin.get('session', '')}#{origin.get('turn', '')}"],
    )
```

Extend `__all__` with `"build_proposal_prompt"`, `"parse_proposal"`, `"propose_artifact"`, and add `meta_harness = ["skill/*.md", "learn_prompt.md"]` to `[tool.setuptools.package-data]` in `pyproject.toml`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_learn_propose.py -q`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/learn.py meta_harness/learn_prompt.md tests/test_learn_propose.py pyproject.toml
git commit -m "feat(learn): propose one artifact at the strongest applicable layer

The proposer is shown the layer ordering explicitly, because a proposer that
cannot see it writes a paragraph for something tool.check could enforce."
```

---

### Task 7: Scoring by replay

**Files:**
- Modify: `meta_harness/learn.py`
- Test: `tests/test_learn_score.py`

**Interfaces:**
- Consumes: `cc_harness.AgentConfig`, `cc_harness.ClaudeCodeHarness`, `replay.verify_expectation`, `replay.load_agent_steps`, `core.TraceRecorder`.
- Produces:
  - `config_with(artifact, base=None) -> AgentConfig` — applies an artifact to a config
  - `run_replay(artifact, apply, workspace_root) -> tuple[bool, dict]`
  - `score_artifact(artifact, regression_tasks, ...) -> dict` with keys `origin_fixed`, `regression_score`, `baseline_score`, `context`, `kept`
  - `RETENTION_REASONS` explaining each rejection

- [ ] **Step 1: Write the failing test**

Create `tests/test_learn_score.py`:

```python
from meta_harness.cc_harness import AgentConfig
from meta_harness.harness_store import Artifact
from meta_harness.learn import config_with, decide_retention


def _artifact(type="doctrine", payload="Read before you edit."):
    return Artifact(id="a1", type=type, origin={"signature": "s"}, payload=payload, replay={})


def test_doctrine_lands_in_the_system_prompt():
    config = config_with(_artifact(), AgentConfig(append_system_prompt="base"))
    assert "base" in config.append_system_prompt
    assert "Read before you edit." in config.append_system_prompt


def test_skill_lands_in_the_skills_map():
    config = config_with(_artifact(type="skill", payload="Body of the skill."))
    assert "Body of the skill." in "".join(config.skills.values())


def test_injection_is_appended_to_the_framing():
    config = config_with(_artifact(type="injection", payload="Remember the encoding."),
                         AgentConfig(prompt_template="{instruction}"))
    assert "{instruction}" in config.prompt_template
    assert "Remember the encoding." in config.prompt_template


def test_a_rule_does_not_change_the_agent_config():
    # Rules act through tool.check, not through the prompt; applying one must be a no-op here.
    base = AgentConfig(append_system_prompt="base")
    assert config_with(_artifact(type="rule"), base).append_system_prompt == "base"


def test_retention_requires_the_origin_to_be_fixed():
    decision = decide_retention(origin_fixed=False, candidate_score=1.0, baseline_score=1.0,
                                candidate_context=100.0, baseline_context=200.0)
    assert decision["kept"] is False
    assert "origin" in decision["reason"]


def test_retention_rejects_a_regression():
    decision = decide_retention(origin_fixed=True, candidate_score=0.8, baseline_score=1.0,
                                candidate_context=100.0, baseline_context=200.0)
    assert decision["kept"] is False
    assert "regress" in decision["reason"]


def test_retention_keeps_a_fix_that_holds_score():
    decision = decide_retention(origin_fixed=True, candidate_score=1.0, baseline_score=1.0,
                                candidate_context=210.0, baseline_context=200.0)
    # Context is the tiebreak, not a veto: fixing the failure is the point.
    assert decision["kept"] is True


def test_retention_notes_a_context_win():
    decision = decide_retention(origin_fixed=True, candidate_score=1.0, baseline_score=1.0,
                                candidate_context=150.0, baseline_context=200.0)
    assert decision["kept"] is True
    assert decision["context_delta"] == -50.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learn_score.py -q`
Expected: FAIL — `ImportError: cannot import name 'config_with'`

- [ ] **Step 3: Write minimal implementation**

Append to `meta_harness/learn.py`:

```python
from .cc_harness import AgentConfig, ClaudeCodeHarness, prepare_workspace, run_claude_code
from .core import TraceRecorder
from .replay import load_agent_steps, verify_expectation


def config_with(artifact: Artifact, base: AgentConfig | None = None) -> AgentConfig:
    """Apply one artifact to an agent configuration.

    A `rule` acts through tool.check and therefore changes nothing here; it is scored by
    running the replay with the rule installed in the hook layer.
    """
    config = base or AgentConfig()
    if artifact.type == "doctrine":
        joined = "\n\n".join(part for part in (config.append_system_prompt, artifact.payload) if part)
        return replace_config(config, append_system_prompt=joined)
    if artifact.type == "skill":
        skills = dict(config.skills or {})
        skills[artifact.id] = artifact.payload
        return replace_config(config, skills=skills)
    if artifact.type == "injection":
        return replace_config(config, prompt_template=config.prompt_template + "\n\n" + artifact.payload)
    return config


def replace_config(config: AgentConfig, **changes: Any) -> AgentConfig:
    from dataclasses import replace

    return replace(config, **changes)


def decide_retention(origin_fixed: bool, candidate_score: float, baseline_score: float,
                     candidate_context: float, baseline_context: float) -> dict[str, Any]:
    """Keep an artifact only if it fixes what it was born from and regresses nothing."""
    context_delta = round(candidate_context - baseline_context, 2)
    if not origin_fixed:
        return {"kept": False, "reason": "origin replay still fails", "context_delta": context_delta}
    if candidate_score < baseline_score:
        return {"kept": False,
                "reason": f"regressed the task set ({candidate_score:.3f} < {baseline_score:.3f})",
                "context_delta": context_delta}
    return {"kept": True, "reason": "origin fixed, no regression", "context_delta": context_delta}


def run_replay(artifact: Artifact, workspace_root: Path, binary: str = "claude",
               timeout: float = 900.0) -> tuple[bool, dict[str, Any]]:
    """Run the artifact's replay and report whether the original failure recurred."""
    replay = artifact.replay or {}
    task = {"instruction": replay.get("instruction", ""), "files": replay.get("files", {})}
    workspace = prepare_workspace(task, workspace_root)
    trace_path = Path(workspace_root) / f"{artifact.id}-replay.jsonl"
    with TraceRecorder(trace_path) as trace:
        run = run_claude_code(workspace, config_with(artifact), task, trace,
                              binary=binary, timeout=timeout)
    steps = load_agent_steps(trace_path)
    fixed = verify_expectation(replay.get("expect") or {}, steps)
    return fixed, {"turns": run.turns, "input_tokens": run.input_tokens,
                   "workspace": str(workspace), "trace": str(trace_path)}
```

Extend `__all__` with `"config_with"`, `"decide_retention"`, `"replace_config"`, `"run_replay"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_learn_score.py -q`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add meta_harness/learn.py tests/test_learn_score.py
git commit -m "feat(learn): retention by replay, with context as the tiebreak

An artifact is kept only if the failure it was born from stops happening and
the task set does not regress. Context is a tiebreak rather than a veto: on a
saturated task set the mean cannot move, so fixing the specific failure is the
only signal that means anything."
```

---

### Task 8: The learn CLI

**Files:**
- Modify: `meta_harness/__main__.py`
- Modify: `meta_harness/__init__.py`
- Test: `tests/test_learn_cli.py`

**Interfaces:**
- Consumes: Tasks 1, 4, 5, 6, 7.
- Produces: `meta-harness learn [--limit N] [--dry-run]`, `learn --status`, `learn --accept <id>`, `learn --reject <id> [--wrong]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_learn_cli.py`:

```python
import json
from pathlib import Path

from meta_harness import __main__ as cli
from meta_harness.harness_store import Artifact, HarnessStore


def test_status_reports_staged_and_installed(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    store = HarnessStore(tmp_path)
    store.stage(Artifact(id="a1", type="rule", origin={"signature": "s1"}, payload="p",
                         replay={}, scores={"kept": True}))
    assert cli.main(["learn", "--status"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["staged"][0]["id"] == "a1"
    assert payload["installed"] == []


def test_accept_moves_a_staged_artifact(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    HarnessStore(tmp_path).stage(Artifact(id="a1", type="doctrine",
                                          origin={"signature": "s1"}, payload="p", replay={}))
    assert cli.main(["learn", "--accept", "a1"]) == 0
    assert [a.id for a in HarnessStore(tmp_path).list_installed()] == ["a1"]


def test_reject_wrong_tombstones(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    HarnessStore(tmp_path).stage(Artifact(id="a1", type="rule",
                                          origin={"signature": "sig1"}, payload="p", replay={}))
    assert cli.main(["learn", "--reject", "a1", "--wrong"]) == 0
    assert HarnessStore(tmp_path).is_tombstoned("sig1") is True


def test_dry_run_selects_without_proposing(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "load_sessions", lambda **kw: [])
    assert cli.main(["learn", "--dry-run"]) == 0
    assert "no uncovered failure" in capsys.readouterr().out


def test_accepting_an_unknown_id_reports_an_error(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("META_HARNESS_HOME", str(tmp_path))
    assert cli.main(["learn", "--accept", "nope"]) == 1
    assert "nope" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learn_cli.py -q`
Expected: FAIL — `argument command: invalid choice: 'learn'`

- [ ] **Step 3: Write minimal implementation**

In `meta_harness/__main__.py`, add the imports:

```python
from .cc_history import load_sessions  # already imported; keep one import line
from .harness_store import Artifact, HarnessStore, harness_home
from .learn import propose_artifact, run_replay, select_target
from .replay import build_replay
```

Add the subparser inside `build_parser()`:

```python
    learn = sub.add_parser("learn", help="turn an observed failure into a scored harness artifact")
    learn.add_argument("--status", action="store_true", help="show staged and installed artifacts")
    learn.add_argument("--accept", metavar="ID")
    learn.add_argument("--reject", metavar="ID")
    learn.add_argument("--wrong", action="store_true",
                       help="with --reject: tombstone the signature so it is never re-proposed")
    learn.add_argument("--limit", type=int, default=60, help="sessions of history to read")
    learn.add_argument("--dry-run", action="store_true", help="select a target, propose nothing")
    learn.add_argument("--model", default="opus", help="model for the proposer")
    learn.add_argument("--provider", default="claude-cli")
```

Add the command implementation:

```python
def _command_learn(args) -> int:
    store = HarnessStore(harness_home())

    if args.status:
        print(json.dumps({
            "home": str(store.root),
            "staged": [{"id": a.id, "type": a.type, "signature": a.signature,
                        "scores": a.scores} for a in store.list_staged()],
            "installed": [{"id": a.id, "type": a.type, "signature": a.signature}
                          for a in store.list_installed()],
            "tombstones": sorted(store.tombstones()),
        }, indent=2))
        return 0

    if args.accept:
        try:
            target = store.accept(args.accept)
        except KeyError as error:
            print(f"cannot accept {args.accept}: {error}")
            return 1
        print(f"installed {args.accept} -> {target}")
        return 0

    if args.reject:
        try:
            target = store.reject(args.reject, wrong=args.wrong)
        except KeyError as error:
            print(f"cannot reject {args.reject}: {error}")
            return 1
        print(f"archived {args.reject} -> {target}" + (" (tombstoned)" if args.wrong else ""))
        return 0

    sessions = load_sessions(limit=args.limit, include_text=True)
    failure = select_target(sessions, store)
    if failure is None:
        print("no uncovered failure class found")
        return 0
    print(f"target: {failure.signature} ({failure.count} occurrences)")

    replay = None
    by_session = {s.session_id: s for s in sessions}
    for episode in failure.episodes:
        session = by_session.get(episode.session_id)
        if session is not None:
            replay = build_replay(episode, session)
            if replay:
                break
    if replay is None:
        print("no replayable episode for this failure class")
        return 1
    if args.dry_run:
        print(json.dumps({"signature": failure.signature, "replay": replay}, indent=2))
        return 0

    model = model_from_environment(args.provider, args.model)
    artifact = propose_artifact(failure, replay,
                               [a.id for a in store.list_installed()], model)
    workspace_root = Path(args.root).parent / ".meta-harness-workspaces" \
        if hasattr(args, "root") else Path(".meta-harness-workspaces")
    workspace_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("META_HARNESS_WORKSPACE_ROOT", str(workspace_root.resolve()))
    fixed, detail = run_replay(artifact, workspace_root)
    artifact.scores = {"origin_fixed": fixed, **detail}
    store.stage(artifact)
    print(json.dumps({"staged": artifact.id, "type": artifact.type,
                      "origin_fixed": fixed, "detail": detail}, indent=2))
    return 0
```

Route it in `main()` next to the other commands:

```python
    if args.command == "learn":
        return _command_learn(args)
```

Export `Artifact`, `HarnessStore`, `harness_home` from `meta_harness/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_learn_cli.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/__main__.py meta_harness/__init__.py tests/test_learn_cli.py
git commit -m "feat(cli): meta-harness learn, status, accept and reject

learn stages a scored artifact and stops there. Installation is a separate,
explicit command because an accepted artifact changes every future session in
every repository."
```

---

## Phase 2 — The hook layer (TypeScript)

### Task 9: Hook scaffolding that fails open

**Files:**
- Create: `hooks/hooks.json`
- Create: `hooks/harness.ts`
- Create: `hooks/README.md`
- Modify: `.claude-plugin/plugin.json`
- Test: `tests/test_hook_assets.py`

**Interfaces:**
- Produces: a registered function-hook module exporting `register`, plus `safely(fn)` — the wrapper that guarantees a throwing hook never breaks a turn.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hook_assets.py`:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_hooks_json_registers_the_module():
    config = json.loads((ROOT / "hooks" / "hooks.json").read_text(encoding="utf-8"))
    assert config["modules"] == ["./harness.ts"]


def test_hook_module_exports_register():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "export const register" in source


def test_every_hook_is_wrapped_so_it_fails_open():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # A learning system that can break a session is worse than no learning system.
    for event in ("tool.call", "tool.check", "prompt.section", "turn.complete"):
        if f"'{event}'" in source:
            assert "safely(" in source, f"{event} must be wrapped"


def test_plugin_declares_the_hooks_directory():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest.get("hooks", "./hooks/hooks.json").endswith("hooks.json")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_assets.py -q`
Expected: FAIL — `FileNotFoundError: hooks/hooks.json`

- [ ] **Step 3: Write minimal implementation**

Create `hooks/hooks.json`:

```json
{ "modules": ["./harness.ts"] }
```

Create `hooks/harness.ts`:

```typescript
/**
 * Meta-Harness function hooks: observe failures, enforce learned artifacts.
 *
 * Every handler is wrapped in `safely`, which swallows errors and falls through to `next`.
 * A learning system that can break a session is worse than no learning system.
 */

import type { On, PluginOptions, Register } from 'claude-code';

export type Fallible<E, R> = (dollar: any, event: E, next: (e: E) => Promise<R>) => Promise<R>;

/** Wrap a handler so a throw becomes a pass-through instead of a broken turn. */
export function safely<E, R>(name: string, handler: Fallible<E, R>): Fallible<E, R> {
  return async (dollar, event, next) => {
    try {
      return await handler(dollar, event, next);
    } catch (error) {
      try {
        dollar.ui.log(
          `meta-harness ${name} skipped (${error instanceof Error ? error.message : String(error)})`,
        );
      } catch {
        // logging must never be the thing that breaks the turn either
      }
      return next(event);
    }
  };
}

export const register: Register = (on: On, options: PluginOptions) => {
  void options;
  void on;
};
```

Create `hooks/README.md`:

```markdown
# Meta-Harness function hooks

Observe failures and enforce learned artifacts, in-process.

| Hook | Job |
|---|---|
| `tool.call` | Append failures (error results, repeated identical calls) to the observation log |
| `tool.check` | Apply installed `rule` artifacts; deny with the reason and the artifact id |
| `prompt.section` | Inject installed `injection` artifacts relevant to this turn |
| `turn.complete` | Count turns since the last learn cycle |

Function hooks are early access. Enable them before loading the plugin:

```sh
export CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1
claude --plugin-dir .
```

Every handler is wrapped in `safely`, so a hook that throws logs and lets the turn proceed.
Hooks never call a model and never block on the network; anything expensive belongs in
`python -m meta_harness`.
```

Add `"hooks": "./hooks/hooks.json"` to `.claude-plugin/plugin.json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_assets.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add hooks/ .claude-plugin/plugin.json tests/test_hook_assets.py
git commit -m "feat(hooks): scaffolding that fails open by construction

Every handler goes through safely(), which turns a throw into a pass-through.
A learning system that can break a session is worse than no learning system."
```

---

### Task 10: Observing failures live

**Files:**
- Modify: `hooks/harness.ts`
- Test: `tests/test_hook_observer.py`

**Interfaces:**
- Produces: a `tool.call` handler appending one JSON line per failure to `~/.claude/harness/observed.jsonl` with `{ts, tool, kind, cause, input}`; `kind` is `tool_error` or `repeat`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hook_assets.py`:

```python
def test_observer_writes_one_line_per_failure():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "observed.jsonl" in source
    assert "'tool.call'" in source
    # Observation must be append-only: a rewrite loses concurrent sessions' records.
    assert "append" in source.lower()


def test_observer_records_repeats_as_well_as_errors():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "tool_error" in source and "repeat" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_assets.py -q`
Expected: FAIL — `assert 'observed.jsonl' in source`

- [ ] **Step 3: Write minimal implementation**

Add to `hooks/harness.ts`:

```typescript
const REPEAT_WINDOW = 6;
const REPEAT_THRESHOLD = 4;

type Recent = { tool: string; key: string };

function harnessHome(dollar: any): string {
  const override = dollar.env?.get?.('META_HARNESS_HOME');
  return override || `${dollar.env?.get?.('USERPROFILE') || dollar.env?.get?.('HOME')}/.claude/harness`;
}

/** Append one JSON line. Append-only: a rewrite would lose concurrent sessions' records. */
async function observe(dollar: any, record: Record<string, unknown>): Promise<void> {
  const path = `${harnessHome(dollar)}/observed.jsonl`;
  const line = `${JSON.stringify({ ts: new Date().toISOString(), ...record })}\n`;
  const existing = (await dollar.fs.exists(path)) ? await dollar.fs.read(path) : '';
  await dollar.fs.write(path, existing + line);
}

function isError(result: unknown): boolean {
  const text = typeof result === 'string' ? result : JSON.stringify(result ?? '');
  return /is_error|error:|Traceback|not recognized|No such file/i.test(text);
}

export function registerObserver(on: On): void {
  const recent: Recent[] = [];

  on('tool.call', safely('tool.call', async (dollar, event: any, next) => {
    const outcome = await next(event);
    const tool = String(event?.tool ?? 'unknown');
    const key = `${tool}:${JSON.stringify(event?.input ?? {}).slice(0, 200)}`;

    recent.push({ tool, key });
    if (recent.length > REPEAT_WINDOW) recent.shift();
    const repeats = recent.filter((entry) => entry.key === key).length;
    if (repeats >= REPEAT_THRESHOLD) {
      await observe(dollar, { kind: 'repeat', tool, input: event?.input });
    }
    if (isError((outcome as any)?.result)) {
      await observe(dollar, { kind: 'tool_error', tool, input: event?.input,
                              text: String((outcome as any)?.result).slice(0, 400) });
    }
    return outcome;
  }));
}
```

Call `registerObserver(on)` from `register`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_assets.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add hooks/harness.ts tests/test_hook_assets.py
git commit -m "feat(hooks): observe tool errors and repeats live

New evidence arrives through the hook rather than by parsing transcripts
afterwards. Appends only, so concurrent sessions do not overwrite each other."
```

---

### Task 11: Enforcing rules

**Files:**
- Modify: `hooks/harness.ts`
- Create: `hooks/rules.ts`
- Test: `tests/test_hook_rules.py`

**Interfaces:**
- Produces: `loadRules(dollar)` reading installed `rule` artifacts; `evaluateRule(rule, event, state) -> {deny: boolean, reason?: string}`; a `tool.check` handler returning `{decision: "deny", reason}`. The built-in `read-before-edit` rule denies `Edit`/`Write` on a path not `Read` in this session.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hook_rules.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_rules_module_exists_and_exports_the_evaluator():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    assert "export function evaluateRule" in source
    assert "export function loadRules" in source


def test_read_before_edit_is_a_built_in_rule():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    # This is the discovered doctrine's top clause, expressed as a mechanism.
    assert "read-before-edit" in source
    assert "Edit" in source and "Read" in source


def test_deny_carries_a_reason_and_the_artifact_id():
    source = (ROOT / "hooks" / "rules.ts").read_text(encoding="utf-8")
    assert "reason" in source
    assert "artifactId" in source or "artifact_id" in source


def test_tool_check_is_registered_and_wrapped():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "'tool.check'" in source
    assert "safely('tool.check'" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_rules.py -q`
Expected: FAIL — `FileNotFoundError: hooks/rules.ts`

- [ ] **Step 3: Write minimal implementation**

Create `hooks/rules.ts`:

```typescript
/**
 * Installed `rule` artifacts, applied at tool.check.
 *
 * A rule is the strongest layer available: it costs no standing tokens and cannot be talked
 * around. `read-before-edit` is built in because it is the discovered doctrine's top clause,
 * and a clause a mechanism can enforce should not be a paragraph.
 */

export type Rule = {
  artifactId: string;
  kind: string;
  tools: string[];
  reason: string;
};

export type SessionState = {
  readPaths: Set<string>;
};

export const BUILT_IN_RULES: Rule[] = [
  {
    artifactId: 'read-before-edit',
    kind: 'read-before-edit',
    tools: ['Edit', 'Write'],
    reason: 'Read this file in the session before editing it: an edit written from an assumption about its contents lands in the wrong place.',
  },
];

/** Installed rule artifacts, plus the built-ins. Missing or malformed files are ignored. */
export async function loadRules(dollar: any, home: string): Promise<Rule[]> {
  const rules = [...BUILT_IN_RULES];
  const registry = `${home}/installed.json`;
  if (!(await dollar.fs.exists(registry))) return rules;
  let entries: Array<{ id: string; type: string }> = [];
  try {
    entries = JSON.parse(await dollar.fs.read(registry));
  } catch {
    return rules;
  }
  for (const entry of entries) {
    if (entry.type !== 'rule') continue;
    const path = `${home}/artifacts/${entry.id}/artifact.json`;
    if (!(await dollar.fs.exists(path))) continue;
    try {
      const artifact = JSON.parse(await dollar.fs.read(path));
      rules.push({
        artifactId: artifact.id,
        kind: artifact.origin?.kind ?? 'custom',
        tools: artifact.origin?.tools ?? [],
        reason: String(artifact.payload ?? '').slice(0, 400),
      });
    } catch {
      continue;
    }
  }
  return rules;
}

export function evaluateRule(
  rule: Rule,
  event: { tool?: string; input?: Record<string, unknown> },
  state: SessionState,
): { deny: boolean; reason?: string } {
  const tool = String(event.tool ?? '');
  if (!rule.tools.includes(tool)) return { deny: false };

  if (rule.kind === 'read-before-edit') {
    const path = String((event.input as any)?.file_path ?? '');
    if (!path || state.readPaths.has(path)) return { deny: false };
    return { deny: true, reason: `${rule.reason} [${rule.artifactId}]` };
  }

  return { deny: false };
}
```

Add to `hooks/harness.ts`:

```typescript
import { evaluateRule, loadRules, type Rule, type SessionState } from './rules.js';

export function registerRules(on: On): void {
  const state: SessionState = { readPaths: new Set<string>() };
  let rules: Rule[] | null = null;

  on('tool.call', safely('tool.call:read-tracking', async (dollar, event: any, next) => {
    if (event?.tool === 'Read') {
      const path = String(event?.input?.file_path ?? '');
      if (path) state.readPaths.add(path);
    }
    return next(event);
  }));

  on('tool.check', safely('tool.check', async (dollar, event: any, next) => {
    if (rules === null) rules = await loadRules(dollar, harnessHome(dollar));
    for (const rule of rules) {
      const verdict = evaluateRule(rule, event, state);
      if (verdict.deny) return { decision: 'deny', reason: verdict.reason };
    }
    return next(event);
  }));
}
```

Call `registerRules(on)` from `register`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_rules.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add hooks/ tests/test_hook_rules.py
git commit -m "feat(hooks): enforce learned rules at tool.check

read-before-edit ships built in: it is the discovered doctrine's top clause,
and a clause a mechanism can enforce should not be a paragraph paid for on
every turn."
```

---

### Task 12: Injecting learned context

**Files:**
- Modify: `hooks/harness.ts`
- Test: `tests/test_hook_injection.py`

**Interfaces:**
- Produces: a `prompt.section` handler returning `{text}` when an installed `injection` artifact's triggers match the turn, `{text: null}` otherwise.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hook_injection.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_prompt_section_is_registered_and_wrapped():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    assert "'prompt.section'" in source
    assert "safely('prompt.section'" in source


def test_injection_returns_null_when_nothing_matches():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # Returning null leaves the section out entirely, which is why this costs nothing
    # on turns where no artifact is relevant.
    assert "text: null" in source


def test_injection_records_what_it_injected():
    source = (ROOT / "hooks" / "harness.ts").read_text(encoding="utf-8")
    # The hook knows what it placed, so use is recorded by construction rather than by
    # asking the model to self-report a retrieval it might forget.
    assert "injected" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hook_injection.py -q`
Expected: FAIL — `assert "'prompt.section'" in source`

- [ ] **Step 3: Write minimal implementation**

Add to `hooks/harness.ts`:

```typescript
type Injection = { artifactId: string; triggers: string[]; text: string };

async function loadInjections(dollar: any, home: string): Promise<Injection[]> {
  const registry = `${home}/installed.json`;
  if (!(await dollar.fs.exists(registry))) return [];
  let entries: Array<{ id: string; type: string }> = [];
  try {
    entries = JSON.parse(await dollar.fs.read(registry));
  } catch {
    return [];
  }
  const out: Injection[] = [];
  for (const entry of entries) {
    if (entry.type !== 'injection') continue;
    const path = `${home}/artifacts/${entry.id}/artifact.json`;
    if (!(await dollar.fs.exists(path))) continue;
    try {
      const artifact = JSON.parse(await dollar.fs.read(path));
      out.push({
        artifactId: artifact.id,
        triggers: artifact.origin?.triggers ?? [],
        text: String(artifact.payload ?? ''),
      });
    } catch {
      continue;
    }
  }
  return out;
}

export function registerInjection(on: On): void {
  let injections: Injection[] | null = null;

  on('prompt.section', safely('prompt.section', async (dollar, event: any, next) => {
    if (injections === null) injections = await loadInjections(dollar, harnessHome(dollar));
    if (injections.length === 0) return next(event);

    const haystack = JSON.stringify(event ?? {}).toLowerCase();
    const matched = injections.filter((injection) =>
      injection.triggers.some((trigger) => haystack.includes(String(trigger).toLowerCase())));
    if (matched.length === 0) return { text: null };

    for (const injection of matched) {
      await observe(dollar, { kind: 'injected', artifactId: injection.artifactId });
    }
    return { text: matched.map((injection) => injection.text).join('\n\n') };
  }));
}
```

Call `registerInjection(on)` from `register`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_hook_injection.py -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add hooks/harness.ts tests/test_hook_injection.py
git commit -m "feat(hooks): inject learned context per turn, and record the hit

Returning text: null leaves the section out, so a turn with nothing relevant
pays nothing. The hook knows what it placed, so use is recorded by
construction instead of asking the model to report a retrieval it may forget."
```

---

### Task 13: Learning from what the hook observed

**Files:**
- Modify: `meta_harness/learn.py`
- Modify: `meta_harness/__main__.py`
- Test: `tests/test_learn_observed.py`

**Interfaces:**
- Produces: `observed_failures(home=None) -> list[FailureClass]` reading `observed.jsonl`, merged into `select_target` ahead of mined history.

- [ ] **Step 1: Write the failing test**

Create `tests/test_learn_observed.py`:

```python
import json
from pathlib import Path

from meta_harness.harness_store import HarnessStore
from meta_harness.learn import observed_failures, select_target


def _log(home: Path, records):
    home.mkdir(parents=True, exist_ok=True)
    (home / "observed.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_reads_observed_failures(tmp_path: Path):
    _log(tmp_path, [
        {"kind": "tool_error", "tool": "Bash", "text": "UnicodeDecodeError: charmap"},
        {"kind": "tool_error", "tool": "Bash", "text": "UnicodeDecodeError: charmap"},
    ])
    failures = observed_failures(tmp_path)
    assert failures[0].signature == "tool_error:Bash:unicode-decode"
    assert failures[0].count == 2


def test_repeat_records_become_thrash_classes(tmp_path: Path):
    _log(tmp_path, [{"kind": "repeat", "tool": "Edit"}])
    assert observed_failures(tmp_path)[0].signature == "thrash:Edit"


def test_a_missing_log_is_not_an_error(tmp_path: Path):
    assert observed_failures(tmp_path / "nothing") == []


def test_observed_failures_outrank_mined_history(tmp_path: Path):
    home = tmp_path / "home"
    _log(home, [{"kind": "tool_error", "tool": "Bash", "text": "Permission denied"}])
    store = HarnessStore(tmp_path / "store")
    # Live observation is fresher evidence than a transcript from weeks ago.
    assert select_target([], store, home=home).signature == "tool_error:Bash:permission"


def test_covered_signatures_are_still_skipped(tmp_path: Path):
    from meta_harness.harness_store import Artifact

    home = tmp_path / "home"
    _log(home, [{"kind": "tool_error", "tool": "Bash", "text": "Permission denied"}])
    store = HarnessStore(tmp_path / "store")
    store.stage(Artifact(id="a1", type="rule",
                         origin={"signature": "tool_error:Bash:permission"},
                         payload="p", replay={}))
    store.accept("a1")
    assert select_target([], store, home=home) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learn_observed.py -q`
Expected: FAIL — `ImportError: cannot import name 'observed_failures'`

- [ ] **Step 3: Write minimal implementation**

Append to `meta_harness/learn.py`:

```python
import json

from .harness_store import harness_home
from .replay import _cause


def observed_failures(home: Path | None = None) -> list[FailureClass]:
    """Failure classes the hook recorded live. Fresher evidence than mined transcripts."""
    path = Path(home or harness_home()) / "observed.jsonl"
    if not path.is_file():
        return []
    grouped: dict[str, FailureClass] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        tool = str(record.get("tool", "unknown"))
        kind = record.get("kind")
        if kind == "repeat":
            signature, klass = f"thrash:{tool}", "thrash"
        elif kind == "tool_error":
            signature = f"tool_error:{tool}:{_cause(str(record.get('text', '')))}"
            klass = "tool_error"
        else:
            continue
        entry = grouped.setdefault(signature, FailureClass(signature=signature, kind=klass,
                                                           tool=tool))
        entry.count += 1
    return sorted(grouped.values(), key=lambda f: (-f.count, f.signature))
```

Change `select_target` to take the live log first:

```python
def select_target(sessions: Sequence[Session], store: HarnessStore,
                  home: Path | None = None) -> FailureClass | None:
    """The most frequent uncovered failure, live observations before mined history."""
    covered = store.covered()
    for failure in list(observed_failures(home)) + rank_failures(sessions):
        if failure.signature not in covered:
            return failure
    return None
```

Extend `__all__` with `"observed_failures"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_learn_observed.py -q`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add meta_harness/learn.py meta_harness/__main__.py tests/test_learn_observed.py
git commit -m "feat(learn): prefer live observations over mined history

What the hook saw this week is fresher evidence than a transcript from weeks
ago, so observed failures are considered first. Covered signatures are still
skipped from either source."
```

---

## Phase 3 — The experiment and the docs

### Task 14: Prose versus mechanism

**Files:**
- Create: `tools/prose_vs_rule.py`
- Test: `tests/test_prose_vs_rule.py`

**Interfaces:**
- Produces: `build_arms(doctrine_path, tasks_path) -> dict` returning two named arms — `prose` (doctrine in `append_system_prompt`) and `rule` (doctrine removed, enforcement left to `tool.check`) — and `summarize(results) -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prose_vs_rule.py`:

```python
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from prose_vs_rule import build_arms, summarize


def test_two_arms_differ_only_in_where_the_doctrine_lives(tmp_path: Path):
    doctrine = tmp_path / "d.md"
    doctrine.write_text("Read before you edit.", encoding="utf-8")
    arms = build_arms(doctrine, tmp_path / "tasks.jsonl")
    assert set(arms) == {"prose", "rule"}
    assert "Read before you edit." in arms["prose"]["append_system_prompt"]
    # The rule arm carries no doctrine text: enforcement is the hook's job.
    assert arms["rule"]["append_system_prompt"] == ""


def test_both_arms_share_the_same_task_set(tmp_path: Path):
    doctrine = tmp_path / "d.md"
    doctrine.write_text("x", encoding="utf-8")
    arms = build_arms(doctrine, tmp_path / "tasks.jsonl")
    assert arms["prose"]["tasks"] == arms["rule"]["tasks"]


def test_summary_names_which_objective_moved():
    text = summarize({"prose": {"score": 1.0, "context": 130000.0},
                      "rule": {"score": 1.0, "context": 96000.0}})
    assert "context" in text.lower()
    assert "26" in text or "34000" in text


def test_summary_refuses_to_claim_an_accuracy_win_when_scores_tie():
    text = summarize({"prose": {"score": 1.0, "context": 100.0},
                      "rule": {"score": 1.0, "context": 100.0}})
    assert "no difference" in text.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prose_vs_rule.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'prose_vs_rule'`

- [ ] **Step 3: Write minimal implementation**

Create `tools/prose_vs_rule.py`:

```python
"""Is the harness the words, or the mechanism?

Runs one doctrine two ways against the same tasks: as prose in the system prompt, and as
tool.check rules with the prose removed. The paper asks whether scaffolding beats the model;
this asks whether the scaffolding has to be read to work.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def build_arms(doctrine_path: Path | str, tasks_path: Path | str) -> dict[str, dict[str, Any]]:
    """Two arms differing only in where the doctrine lives."""
    doctrine = Path(doctrine_path).read_text(encoding="utf-8").strip()
    tasks = str(tasks_path)
    return {
        "prose": {"append_system_prompt": doctrine, "rules_enabled": False, "tasks": tasks},
        "rule": {"append_system_prompt": "", "rules_enabled": True, "tasks": tasks},
    }


def summarize(results: Mapping[str, Mapping[str, float]]) -> str:
    prose, rule = results.get("prose", {}), results.get("rule", {})
    score_delta = float(rule.get("score", 0)) - float(prose.get("score", 0))
    context_delta = float(rule.get("context", 0)) - float(prose.get("context", 0))
    if score_delta == 0 and context_delta == 0:
        return "no difference between prose and mechanism on this task set"
    lines = []
    if score_delta:
        lines.append(f"score moved {score_delta:+.3f} (prose {prose.get('score')} -> "
                     f"rule {rule.get('score')})")
    if context_delta:
        share = abs(context_delta) / max(1.0, float(prose.get("context", 1)))
        lines.append(f"context moved {context_delta:+.0f} tokens ({share:.0%})")
    if not score_delta:
        lines.append("accuracy unchanged: this is a context result, not an accuracy one")
    return "; ".join(lines)


if __name__ == "__main__":
    import sys

    arms = build_arms(sys.argv[1], sys.argv[2])
    print(json.dumps(arms, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prose_vs_rule.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/prose_vs_rule.py tests/test_prose_vs_rule.py
git commit -m "feat(experiment): run one doctrine as prose and as mechanism

Two arms, same tasks, differing only in where the doctrine lives. The summary
refuses to report an accuracy win when the scores tie, which is the failure
mode this whole line of work keeps running into."
```

---

### Task 15: Documentation and the learn skill

**Files:**
- Create: `skills/learning-from-failures/SKILL.md`
- Modify: `README.md`
- Modify: `docs/plugin.md`
- Modify: `.claude-plugin/plugin.json`
- Test: `tests/test_learn_skill.py`

**Interfaces:**
- Produces: a model-invoked skill telling Claude when to run a learn cycle and how to read its output.

- [ ] **Step 1: Write the failing test**

Create `tests/test_learn_skill.py`:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKILL = ROOT / "skills" / "learning-from-failures" / "SKILL.md"


def test_skill_exists_with_frontmatter():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---")
    assert "name: learning-from-failures" in text
    assert "description: Use when" in text


def test_description_states_triggers_not_workflow():
    text = SKILL.read_text(encoding="utf-8")
    description = text.split("description:", 1)[1].split("\n---", 1)[0]
    # A description that summarises the workflow gets followed instead of the skill body.
    assert "mine" not in description.lower()
    assert "step" not in description.lower()


def test_skill_names_the_staging_gate():
    text = SKILL.read_text(encoding="utf-8")
    assert "--accept" in text
    assert "never" in text.lower()


def test_plugin_version_bumped():
    manifest = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert manifest["version"] >= "0.6.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_learn_skill.py -q`
Expected: FAIL — `FileNotFoundError: skills/learning-from-failures/SKILL.md`

- [ ] **Step 3: Write minimal implementation**

Create `skills/learning-from-failures/SKILL.md`:

```markdown
---
name: learning-from-failures
description: Use when the same failure keeps recurring in this repo - a tool erroring the same way, a loop of retries, a correction you have given before - or when someone asks whether the harness can be improved from what already went wrong
---

# Learning from failures

## Overview

Every failure Claude Code produces is evidence. A learn cycle turns one of them into a harness
artifact, scores it against the failure that produced it, and stages it for review.

**Core principle:** An artifact earns its place by stopping the specific failure it was born
from, without regressing anything else. Nothing is adopted on a score alone.

## When to Use

- The same tool errors the same way more than once
- A retry loop burns turns on one tool
- A correction repeats something you have already said

**Skip when:** it happened once. One failure is an accident, not a pattern.

## Run a cycle

```bash
python -m meta_harness learn            # select, propose, score, stage
python -m meta_harness learn --status   # what is staged, installed, tombstoned
```

`learn` picks the most frequent failure not already covered by an installed artifact or a
tombstone, builds a replay from the request and the pre-edit file contents, asks the proposer for
one artifact at the strongest applicable layer, runs the replay, and stages the result.

## Read the result before accepting

```bash
python -m meta_harness learn --accept <id>          # install it
python -m meta_harness learn --reject <id>          # archive it
python -m meta_harness learn --reject <id> --wrong  # archive and never propose it again
```

**Never accept without reading the payload.** An installed artifact changes every future session
in every repository; the score justifies proposing it, never adopting it unseen. Check that a
rule cannot deny legitimate work, and that doctrine names no file from the replay - naming one is
memorisation, and it will not transfer.

## The layer ordering

| Type | Mechanism | Ignorable? |
|---|---|---|
| `rule` | `tool.check` denies the call | No |
| `injection` | text placed into the turn | No |
| `skill` | `SKILL.md` | Yes |
| `doctrine` | `CLAUDE.md`, paid every turn | Yes |

A fix expressed as a rule costs nothing per turn and cannot be talked around. Prefer the
strongest layer that can carry the fix; reach for doctrine only when no mechanism can.

**REQUIRED BACKGROUND:** `optimizing-harnesses` - the three laws this cycle enforces.
```

Bump `.claude-plugin/plugin.json` to `"version": "0.6.0"`.

Add to `README.md` under the Claude Code section:

```markdown
### Learning from failures

```bash
python -m meta_harness learn            # failure -> artifact -> replay-scored -> staged
python -m meta_harness learn --status
python -m meta_harness learn --accept <id>
```

Each artifact is kept only if it fixes the failure it was born from without regressing the task
set, and nothing installs itself. See
[the design](docs/superpowers/specs/2026-09-20-learning-harness-design.md).
```

Add a `Learning` row to the skills table in `docs/plugin.md` naming `learning-from-failures`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_learn_skill.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the whole suite and validate the plugin**

Run: `python -m pytest tests -q && claude plugin validate .`
Expected: PASS, and `✔ Validation passed`

- [ ] **Step 6: Commit**

```bash
git add skills/ README.md docs/plugin.md .claude-plugin/plugin.json tests/test_learn_skill.py
git commit -m "docs(learn): ship the learning-from-failures skill

The skill states the staging gate and the layer ordering, because the two ways
this goes wrong are accepting an artifact unread and writing a paragraph for
something a mechanism could enforce."
```

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| §3 Observe | 10, 13 |
| §3 Learn | 5, 6, 7, 8 |
| §3 Enforce | 11, 12 |
| §4 Artifact layers | 6 (proposal), 7 (application), 11/12 (enforcement) |
| §5.1 Select | 5, 13 |
| §5.2 Replay | 2, 3, 4 |
| §5.3 Propose | 6 |
| §5.4 Score | 7 |
| §5.5 Stage, never auto-install | 1, 8, 15 |
| §6 Storage | 1 |
| §7 Hook layer, fails open | 9, 10, 11, 12 |
| §9 Measurement | 7 (replay), existing `compare` |
| §10 The experiment | 14 |
| §13 Risks — rule denies legitimate work | 15 (skill tells the reader to check) |

**Placeholder scan:** no TBDs; every code step carries runnable code; no "same as Task N".

**Type consistency checks performed:**
- `Artifact` field order and names are fixed in Task 1 and used unchanged in Tasks 6, 7, 8, 11, 12.
- `episode_signature` output format `tool_error:<tool>:<cause>` is asserted in Task 2 and consumed verbatim by Tasks 5, 13.
- `expectation_for` returns `{"no_tool_error"|"no_thrash": {...}}` in Task 3 and is read by `verify_expectation` in the same task and `run_replay` in Task 7.
- `build_replay` returns keys `instruction/files/expect/_origin` in Task 4; Task 6 reads `_origin`, Task 7 reads `instruction/files/expect`.
- `select_target(sessions, store)` gains a third parameter `home` in Task 13; Task 8's call site passes two positionally and stays valid.
- `harness_home()` is the single source of the root in Tasks 1, 8, 13 and mirrored in TypeScript by `harnessHome()` in Tasks 10, 11, 12.
- `installed.json` entry shape `{id, type, signature, accepted}` is written in Task 1 and read by Tasks 11 and 12.

**Ordering constraint:** Tasks 2→3→4 all extend `meta_harness/replay.py` and must run in order. Tasks 5→6→7→8 all extend `meta_harness/learn.py` and must run in order. Task 13 modifies `select_target` from Task 5. Tasks 9→10→11→12 all extend `hooks/harness.ts` and must run in order. Task 14 is independent and may run any time after Task 1.

**Known gap, stated rather than hidden:** a `rule` artifact cannot be scored end-to-end by `run_replay` in Phase 1, because enforcement lives in the hook layer built in Phase 2. Until Task 11 lands, `config_with` treats a rule as a no-op and its replay measures the unmodified harness — so a rule will not show `origin_fixed` and will not be staged as kept. Task 11's landing is what makes rule artifacts scorable; propose doctrine or injection artifacts before then.
