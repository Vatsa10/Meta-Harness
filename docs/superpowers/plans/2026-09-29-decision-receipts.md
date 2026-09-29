# Decision Receipts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether each plugin intervention helps by recording a receipt when it fires, attaching what happened next, and withholding a random 10% of learned interventions as a control.

**Architecture:** Hooks stay dumb: they append receipts and a per-session call sequence number to JSONL files under the harness home, and apply a holdout draw to learned interventions only. All judgement lives in Python: `meta_harness/receipts.py` joins receipts to observed failures by `(session, call)` and computes per-artifact verdicts behind a minimum-sample gate. The CLI and `learn --status` surface those verdicts.

**Tech Stack:** Python 3.10+ stdlib; TypeScript Claude Code function hooks (Node 22 `--experimental-strip-types` for tests); pytest.

**Spec:** `docs/superpowers/specs/2026-09-29-decision-receipts-design.md`

## Global Constraints

- Python stdlib only; `dependencies = []` in `pyproject.toml` stays empty.
- Every `subprocess.run(..., text=True)` also passes `encoding="utf-8", errors="replace"`.
- Explicit `encoding="utf-8"` on every file read and write; Windows-safe `pathlib` paths.
- Hook loader rules (enforced by `tests/test_hook_literals.py` and `tests/test_plugin_validate.py`): every `on()` hook argument is a function literal; each event is registered once (features add to `register`'s per-event chain); `$` is never passed, bound or spread into a helper — helpers receive the `io` object; `$.env.get` takes a literal name.
- `tool.call` arguments are top-level on the event; read them only through `toolArgs(event)`. `tool.check` reads `event.input`.
- Nothing ever listens on `prompt.section`.
- Every hook path fails open: a receipt write failure is logged and never blocks, alters or retries a decision.
- Receipts carry no message text and no tool input.
- `claude plugin validate .claude-plugin/plugin.json` passes.
- Tests discriminate: each new test is verified to FAIL against a deliberate mutation before commit; restore via a scratch copy, never `git checkout --` over uncommitted work.
- No AI attribution in commits, authors, comments or docs. No `Co-Authored-By` or `Claude-Session` lines.
- `keel/` is an untracked checkout in the repo root: never `git add` it.

## Review Focus

1. **No harness home, or an unwritable one** — receipts must fail open; the decision is unchanged. (Task 1)
2. **Corrupt or out-of-range `receipts.json`** — default rate 0.1; rates above 0.5 clamp to 0.5; negative clamps to 0. (Task 1)
3. **A session rule or rejection memory must never be held out**, whatever the draw says. (Task 3)
4. **Receipts with no matching observations** (a quiet session) — recurrence is 0, not an error; an artifact with too few receipts reports "not enough data", never a verdict. (Task 5)
5. **An observation from a different session with the same signature** must never count as a recurrence. (Task 5)

---

### Task 1: Receipt writer, call sequence and holdout draw

**Files:**
- Create: `hooks/receipts.ts`
- Test: `hooks/harness.receipts.test.mts`, `tests/test_hook_receipts.py`

**Interfaces:**
- Produces (all exported from `hooks/receipts.ts`):
  - `bumpCall(sessionId: string): number` — increments and returns the session's call count.
  - `currentCall(sessionId: string): number` — returns it without incrementing (0 when unseen).
  - `holdoutRate(io: any, home: string): Promise<number>` — reads `<home>/receipts.json`.
  - `setDraw(fn: () => number): void` and `draw(): number` — the single randomness source.
  - `shouldHold(rate: number): boolean` — `draw() < rate`.
  - `writeReceipt(io: any, home: string, sessionId: string, record: ReceiptFields): Promise<void>`.
  - `type ReceiptFields = { event: string; source: string; artifact: string | null; signature: string | null; tool: string; decision: 'acted' | 'held' }`.

- [ ] **Step 1: Write the failing test** — create `hooks/harness.receipts.test.mts`:

```ts
import assert from 'node:assert/strict';
import { bumpCall, currentCall, holdoutRate, setDraw, shouldHold, writeReceipt } from './receipts.ts';

function fakeIo(files: Map<string, string>, failWrite = false) {
  return {
    fs: {
      exists: async (p: string) => files.has(p),
      read: async (p: string) => { if (!files.has(p)) throw new Error('missing'); return files.get(p)!; },
      write: async (p: string, t: string) => { if (failWrite) throw new Error('disk full'); files.set(p, t); },
    },
    log: () => {},
  };
}

// call sequence is per session
assert.equal(currentCall('s1'), 0);
assert.equal(bumpCall('s1'), 1);
assert.equal(bumpCall('s1'), 2);
assert.equal(currentCall('s2'), 0, 'sessions must not share a counter');

// holdout rate: default, clamps, corrupt
const home = '/h';
assert.equal(await holdoutRate(fakeIo(new Map()), home), 0.1, 'missing config means 0.1');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', '{"holdout_rate": 0.9}']])), home), 0.5, 'clamped to 0.5');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', '{"holdout_rate": -1}']])), home), 0, 'clamped to 0');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', 'not json']])), home), 0.1, 'corrupt means default');
assert.equal(await holdoutRate(fakeIo(new Map([['/h/receipts.json', '{"holdout_rate": 0}']])), home), 0, 'zero disables');

// the draw is replaceable
setDraw(() => 0.05);
assert.equal(shouldHold(0.1), true);
setDraw(() => 0.5);
assert.equal(shouldHold(0.1), false);
assert.equal(shouldHold(0), false, 'a zero rate never holds');

// receipts append, one JSON object per line, no text or input
const files = new Map<string, string>();
await writeReceipt(fakeIo(files), home, 's1', { event: 'tool.check', source: 'learned-rule', artifact: 'a1', signature: 'tool_error:Bash:x', tool: 'Bash', decision: 'acted' });
await writeReceipt(fakeIo(files), home, 's1', { event: 'tool.check', source: 'learned-rule', artifact: 'a1', signature: 'tool_error:Bash:x', tool: 'Bash', decision: 'held' });
const lines = files.get('/h/receipts-s1.jsonl')!.trim().split('\n').map((l) => JSON.parse(l));
assert.equal(lines.length, 2, 'receipts must append');
assert.equal(lines[0].call, 2, 'a receipt carries the current call number');
assert.equal(lines[1].decision, 'held');
assert.ok(!('input' in lines[0]) && !('text' in lines[0]), 'receipts carry no input or text');

// fail open
await writeReceipt(fakeIo(new Map(), true), home, 's1', { event: 'x', source: 'drift-note', artifact: null, signature: null, tool: 'Bash', decision: 'acted' });
console.log('all assertions passed');
```

Create `tests/test_hook_receipts.py`, copying the node-runner shape of `tests/test_hook_drift.py` exactly (same `--experimental-strip-types`, `--import` preload, `encoding="utf-8", errors="replace"`, and a `pytest.skip` with a stated reason when node is missing), pointed at `hooks/harness.receipts.test.mts`.

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest -q tests/test_hook_receipts.py`
Expected: FAIL — `Cannot find module './receipts.ts'`.

- [ ] **Step 3: Implement** — create `hooks/receipts.ts`:

```ts
/**
 * Decision receipts (spec: docs/superpowers/specs/2026-09-29-decision-receipts-design.md).
 * A receipt is written when the plugin acts - or, for a held-out case, would have acted - and
 * the outcome is attributed offline by meta_harness/receipts.py. Helpers take `io`, never `$`.
 */
export type ReceiptFields = {
  event: string; source: string; artifact: string | null;
  signature: string | null; tool: string; decision: 'acted' | 'held';
};

const DEFAULT_RATE = 0.1;
const MAX_RATE = 0.5;
const calls = new Map<string, number>();
let drawFn: () => number = Math.random;

export function bumpCall(sessionId: string): number {
  const next = (calls.get(sessionId) ?? 0) + 1;
  calls.set(sessionId, next);
  return next;
}

export function currentCall(sessionId: string): number {
  return calls.get(sessionId) ?? 0;
}

export function setDraw(fn: () => number): void { drawFn = fn; }
export function draw(): number { return drawFn(); }
export function shouldHold(rate: number): boolean { return rate > 0 && draw() < rate; }

export async function holdoutRate(io: any, home: string): Promise<number> {
  try {
    const path = `${home}/receipts.json`;
    if (!(await io.fs.exists(path))) return DEFAULT_RATE;
    const value = Number(JSON.parse(await io.fs.read(path))?.holdout_rate);
    if (!Number.isFinite(value)) return DEFAULT_RATE;
    return Math.min(MAX_RATE, Math.max(0, value));
  } catch {
    return DEFAULT_RATE;
  }
}

export async function writeReceipt(io: any, home: string, sessionId: string,
                                   record: ReceiptFields): Promise<void> {
  try {
    const path = `${home}/receipts-${sessionId}.jsonl`;
    const line = JSON.stringify({ ts: new Date().toISOString(), session: sessionId,
                                  call: currentCall(sessionId), ...record }) + '\n';
    const prior = (await io.fs.exists(path)) ? await io.fs.read(path) : '';
    await io.fs.write(path, prior + line);
  } catch (error) {
    try { io.log?.(`meta-harness: receipt not written: ${String(error)}`); } catch { /* fail open */ }
  }
}
```

- [ ] **Step 4: Verify it passes, then mutations**

Run: `python -m pytest -q tests/test_hook_receipts.py` → PASS.
Mutations (each must fail the test, then restore from a scratch copy): remove the `MAX_RATE` clamp; make `shouldHold` ignore `rate > 0`; make `writeReceipt` overwrite instead of append.

- [ ] **Step 5: Full suite, validate, commit**

```bash
python -m pytest -q
claude plugin validate .claude-plugin/plugin.json
git add hooks/receipts.ts hooks/harness.receipts.test.mts tests/test_hook_receipts.py
git commit -m "feat(hooks): receipt writer, per-session call sequence and holdout draw"
```

---

### Task 2: Observations carry the call sequence

**Files:**
- Modify: `hooks/harness.ts` (`registerObserver`, ~line 361)
- Test: extend `hooks/harness.observer.test.mts`

**Interfaces:**
- Consumes: `bumpCall`, `currentCall` from Task 1.
- Produces: every `observed-<session>.jsonl` record has an integer `call` field; the call counter advances once per `tool.call`.

- [ ] **Step 1: Write the failing test** — in `hooks/harness.observer.test.mts`, drive three `tool.call` events through the observer with the real top-level argument shape; the second returns an error result. Assert the single observed record has `call === 2`.

```ts
// three calls, the second fails: the observation must name call 2
const seen: any[] = [];
// (reuse the file's existing fake io and registration helper; the observed file's lines are
// parsed after the run)
assert.equal(observedRecords.length, 1);
assert.equal(observedRecords[0].call, 2, 'an observation carries its call number');
```

- [ ] **Step 2: Run to verify it fails** — `python -m pytest -q tests/test_hook_assets.py -k observer` → FAIL (`call` undefined).

- [ ] **Step 3: Implement** — in `registerObserver`, import `bumpCall` from `./receipts.js`. Inside the `guardAfter` body, first line after resolving `sessionId`: `const call = bumpCall(sessionId);` and add `call` to both `observe(...)` records. Move the `sessionId` lookup above the repeat check so the counter advances on every call, error or not.

- [ ] **Step 4: Verify and mutate** — PASS; mutation: bump only on error → test fails. Restore.

- [ ] **Step 5: Commit** — `git commit -m "feat(hooks): observations carry the session call number"` after full suite and validate.

---

### Task 3: Receipts and holdout at tool.check

**Files:**
- Modify: `hooks/rules.ts` (`Rule` type, `loadRules`), `hooks/harness.ts` (`registerRules` tool.check)
- Test: extend `hooks/harness.rules.test.mts`

**Interfaces:**
- Consumes: `writeReceipt`, `holdoutRate`, `shouldHold`, `setDraw` from Task 1.
- Produces: `Rule` gains `signature: string` (from the registry row, `''` when absent).

- [ ] **Step 1: Write the failing tests** — with an installed learned rule that denies a call:
  - draw `0.99`, rate 0.1 → denied, one receipt `{source: 'learned-rule', decision: 'acted', artifact, signature}`.
  - draw `0.01`, rate 0.1 → **allowed** (next called once), one receipt `decision: 'held'`.
  - a session rule ("stop running pytest") with draw `0.01` → still **denied**, receipt `source: 'session-rule', decision: 'acted'`.
  - a rejection-memory denial with draw `0.01` → still denied, receipt `source: 'rejection-memory', decision: 'acted'`.
  - `receipts.json` rate `0` with draw `0.0` → denied (a zero rate never holds).

- [ ] **Step 2: Verify they fail.**

- [ ] **Step 3: Implement.** In `loadRules`, copy the registry row's `signature` onto each `Rule`. In `registerRules`' tool.check, resolve `home` and `sessionId` once. For the session-rule and rejection-memory deny branches, `await writeReceipt(io, home, sessionId, {event: 'tool.check', source: 'session-rule' | 'rejection-memory', artifact: null, signature: null, tool, decision: 'acted'})` before returning the deny. For the learned-rule loop: on a deny verdict, `const held = shouldHold(await holdoutRate(io, home));` write the receipt with `decision: held ? 'held' : 'acted'`, `artifact: rule.artifactId`, `signature: rule.signature`; if held, `continue` so the call proceeds as though the rule were not installed; otherwise return the deny.

- [ ] **Step 4: Verify and mutate** — mutations: hold session rules too (the session-rule test fails); ignore the draw (the held test fails). Restore each.

- [ ] **Step 5: Commit** — `git commit -m "feat(hooks): receipts and a holdout for learned rules at tool.check"` after full suite and validate.

---

### Task 4: Receipts and holdout for injections and the drift note

**Files:**
- Modify: `hooks/harness.ts` (`registerInjection` ~579, `registerDrift` ~619), `hooks/rules.ts` or wherever `Injection` rows load (`loadInjections` ~557) to carry `signature`
- Test: extend `hooks/harness.injection.test.mts` and `hooks/harness.drift.test.mts`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces: `Injection` gains `signature: string`.

- [ ] **Step 1: Write the failing tests**
  - Injection matched, draw `0.99` → context attached, receipt `learned-injection` `acted`.
  - Injection matched, draw `0.01` → **no context attached**, event reaches core unchanged, receipt `held`.
  - Drift enabled and crossing `min_calls`, draw `0.99` → note on the tool result, receipt `drift-note` `acted`, `artifact: null`.
  - Same with draw `0.01` → outcome returned unchanged (same object), receipt `held`, and the latch is still set so no second note in the stretch.
- [ ] **Step 2: Verify they fail.**
- [ ] **Step 3: Implement** — injection: after `matched` is computed and non-empty, decide per injection; attach only acted ones; if none acted, `return next(event)`. Drift: when `shouldWarn` fires, draw; write the receipt; set the latch either way; return the note only when acted.
- [ ] **Step 4: Verify and mutate** — mutations: attach held injections anyway; do not set the latch on a held drift note (a second note appears). Restore each.
- [ ] **Step 5: Commit** — `git commit -m "feat(hooks): receipts and a holdout for injections and the drift note"` after full suite, validate, and the existing system-prompt contract test still passing.

---

### Task 5: Outcome attribution and verdicts

**Files:**
- Create: `meta_harness/receipts.py`
- Test: `tests/test_receipts.py`

**Interfaces:**
- Produces:
  - `K = 10`, `MIN_PER_ARM = 5`, `MIN_EFFECT = 0.2`, `Z = 1.96`.
  - `load_receipts(home: Path) -> list[dict]`, `load_observations(home: Path) -> list[dict]`.
  - `attribute(receipts, observations) -> list[dict]` — each receipt plus `recurred: bool`.
  - `summarize(attributed) -> list[dict]` — per `(source, artifact)`: `acted, held, acted_rate, held_rate, effect, verdict` where `verdict ∈ {"helps", "no measurable effect", "not enough data", "no control arm"}`.

- [ ] **Step 1: Write the failing test** — create `tests/test_receipts.py`:

```python
from meta_harness.receipts import attribute, summarize, MIN_PER_ARM


def r(session, call, decision, artifact="a1", source="learned-rule", sig="tool_error:Bash:x"):
    return {"session": session, "call": call, "decision": decision, "artifact": artifact,
            "source": source, "signature": sig}


def o(session, call, sig="tool_error:Bash:x"):
    return {"session": session, "call": call, "kind": "tool_error",
            "tool": sig.split(":")[1], "cause": sig.split(":")[2]}


def test_recurrence_is_within_k_calls_same_session_same_signature():
    out = attribute([r("s1", 5, "acted")], [o("s1", 9)])
    assert out[0]["recurred"] is True
    assert attribute([r("s1", 5, "acted")], [o("s1", 16)])[0]["recurred"] is False  # past K
    assert attribute([r("s1", 5, "acted")], [o("s2", 6)])[0]["recurred"] is False   # other session
    assert attribute([r("s1", 5, "acted")], [o("s1", 5)])[0]["recurred"] is False   # not after
    assert attribute([r("s1", 5, "acted")], [o("s1", 6, "tool_error:Bash:y")])[0]["recurred"] is False


def test_a_quiet_session_is_no_recurrence_not_an_error():
    assert attribute([r("s1", 1, "acted")], [])[0]["recurred"] is False


def test_too_few_receipts_is_not_enough_data():
    rows = [r("s1", i, "acted") for i in range(MIN_PER_ARM)] + [r("s2", 1, "held")]
    assert summarize(attribute(rows, []))[0]["verdict"] == "not enough data"


def test_a_clear_effect_helps():
    acted = [r(f"a{i}", 1, "acted") for i in range(20)]
    held = [r(f"h{i}", 1, "held") for i in range(20)]
    obs = [o(f"h{i}", 3) for i in range(16)]                      # held recurs 80%
    row = summarize(attribute(acted + held, obs))[0]
    assert row["verdict"] == "helps" and row["held_rate"] == 0.8 and row["acted_rate"] == 0.0


def test_equal_rates_is_no_measurable_effect():
    acted = [r(f"a{i}", 1, "acted") for i in range(20)]
    held = [r(f"h{i}", 1, "held") for i in range(20)]
    obs = [o(f"a{i}", 3) for i in range(10)] + [o(f"h{i}", 3) for i in range(10)]
    assert summarize(attribute(acted + held, obs))[0]["verdict"] == "no measurable effect"


def test_user_driven_sources_never_get_a_verdict():
    rows = [r(f"s{i}", 1, "acted", artifact=None, source="session-rule") for i in range(30)]
    assert summarize(attribute(rows, []))[0]["verdict"] == "no control arm"
```

- [ ] **Step 2: Verify it fails** — `ModuleNotFoundError: meta_harness.receipts`.

- [ ] **Step 3: Implement** — create `meta_harness/receipts.py`:

```python
"""Attribute outcomes to decision receipts (spec 2026-09-29-decision-receipts-design.md)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

K = 10
MIN_PER_ARM = 5
MIN_EFFECT = 0.2
Z = 1.96
USER_DRIVEN = {"session-rule", "rejection-memory"}


def _read_jsonl(paths) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_receipts(home: Path) -> list[dict[str, Any]]:
    return _read_jsonl(sorted(Path(home).glob("receipts-*.jsonl"))) if Path(home).exists() else []


def load_observations(home: Path) -> list[dict[str, Any]]:
    return _read_jsonl(sorted(Path(home).glob("observed-*.jsonl"))) if Path(home).exists() else []


def _signature(obs: dict[str, Any]) -> str:
    return f"tool_error:{obs.get('tool')}:{obs.get('cause')}"


def attribute(receipts: Sequence[dict[str, Any]],
              observations: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for obs in observations:
        if obs.get("kind") == "tool_error" and isinstance(obs.get("call"), int):
            by_session[str(obs.get("session", ""))].append(obs)
    out = []
    for receipt in receipts:
        call, signature = receipt.get("call"), receipt.get("signature")
        recurred = isinstance(call, int) and bool(signature) and any(
            call < obs["call"] <= call + K and _signature(obs) == signature
            for obs in by_session.get(str(receipt.get("session", "")), []))
        out.append({**receipt, "recurred": recurred})
    return out


def _verdict(source: str, acted: list, held: list) -> tuple[str, float | None, float | None]:
    acted_rate = sum(r["recurred"] for r in acted) / len(acted) if acted else None
    held_rate = sum(r["recurred"] for r in held) / len(held) if held else None
    if source in USER_DRIVEN:
        return "no control arm", acted_rate, held_rate
    if len(acted) < MIN_PER_ARM or len(held) < MIN_PER_ARM:
        return "not enough data", acted_rate, held_rate
    pooled = (sum(r["recurred"] for r in acted) + sum(r["recurred"] for r in held)) / (len(acted) + len(held))
    se = math.sqrt(pooled * (1 - pooled) * (1 / len(acted) + 1 / len(held)))
    effect = held_rate - acted_rate
    z = effect / se if se > 0 else 0.0
    if effect >= MIN_EFFECT and z >= Z:
        return "helps", acted_rate, held_rate
    return "no measurable effect", acted_rate, held_rate


def summarize(attributed: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for row in attributed:
        groups[(str(row.get("source", "")), row.get("artifact"))].append(row)
    table = []
    for (source, artifact), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        acted = [r for r in rows if r.get("decision") == "acted"]
        held = [r for r in rows if r.get("decision") == "held"]
        verdict, acted_rate, held_rate = _verdict(source, acted, held)
        effect = (held_rate - acted_rate) if acted_rate is not None and held_rate is not None else None
        table.append({"source": source, "artifact": artifact, "acted": len(acted), "held": len(held),
                      "acted_rate": acted_rate, "held_rate": held_rate, "effect": effect,
                      "verdict": verdict})
    return table
```

- [ ] **Step 4: Verify and mutate** — PASS. Mutations: drop the same-session filter; use `<= call` instead of `<`; remove the `MIN_PER_ARM` gate; remove the `USER_DRIVEN` branch. Each must fail a test. Restore.

- [ ] **Step 5: Commit** — `git commit -m "feat(receipts): attribute outcomes and decide verdicts behind a minimum-sample gate"`.

---

### Task 6: `meta-harness receipts report | export`

**Files:**
- Modify: `meta_harness/__main__.py` (new `receipts` subcommand)
- Test: `tests/test_receipts_cli.py`

**Interfaces:**
- Consumes: Task 5's `load_receipts`, `load_observations`, `attribute`, `summarize`; `harness_home()`.
- Produces: `meta-harness receipts report [--json]` and `meta-harness receipts export [--out PATH]`.

- [ ] **Step 1: Write the failing tests** — with `META_HARNESS_HOME` at `tmp_path` holding a `receipts-s1.jsonl` and an `observed-s1.jsonl`:
  - `report --json` prints the `summarize` table as JSON (parse it; assert the row's `verdict`).
  - `report` (human) prints each artifact with its verdict and states the thresholds (`K`, `MIN_PER_ARM`).
  - `report` on an empty home prints "no receipts yet" and exits 0.
  - `export --out case.jsonl` writes one JSON line per receipt, each with `recurred`.
- [ ] **Step 2: Verify they fail** — `invalid choice: 'receipts'`.
- [ ] **Step 3: Implement** — add a `receipts` subparser with a positional `action` in `{"report", "export"}`, `--json`, `--out`; add the branch to `main()`'s if-chain calling a `_command_receipts(args)` handler. Keep edits confined to the new subcommand.
- [ ] **Step 4: Verify and mutate** — mutation: export drops `recurred` → test fails. Restore.
- [ ] **Step 5: Commit** — `git commit -m "feat(cli): meta-harness receipts report and export"`.

---

### Task 7: Receipt verdicts in `learn --status` and retirement

**Files:**
- Modify: `meta_harness/__main__.py` (`learn --status`), `meta_harness/temporal.py` (`retirement_candidates`)
- Test: extend `tests/test_learn_cli.py`, `tests/test_retirement.py`

**Interfaces:**
- Consumes: Task 5 `summarize`.
- Produces: installed entries in `learn --status` carry `receipt_verdict`; `retirement_candidates(..., verdicts: Mapping[str, str] | None = None)` additionally proposes any installed artifact whose verdict is `"no measurable effect"`, with a reason naming the receipt evidence. The existing two-argument call keeps working.

- [ ] **Step 1: Write the failing tests** — an installed artifact with a `no measurable effect` verdict is proposed (reason mentions receipts); one with `helps` or `not enough data` is not.
- [ ] **Step 2: Verify they fail.**
- [ ] **Step 3: Implement** — compute the verdict map in `learn --status` from `summarize(attribute(load_receipts(home), load_observations(home)))`, keyed by artifact id; pass it to `retirement_candidates`.
- [ ] **Step 4: Verify and mutate** — mutation: propose on `not enough data` → test fails. Restore.
- [ ] **Step 5: Commit** — `git commit -m "feat(learn): receipt verdicts in --status and as evidence for retirement"`.

---

### Task 8: Docs, command and live verification

**Files:**
- Modify: `README.md`, `hooks/README.md`, `commands/harness.md`, `skills/learning-from-failures/SKILL.md`, `.claude-plugin/plugin.json` (`version` → `0.8.0` only)
- Test: extend `tests/test_docs_claims.py`

- [ ] **Step 1: Write the failing doc tests** — README mentions `receipts report`, the 10% default holdout and that it is configurable to 0; README states session rules and rejection memory are never held out; `/meta-harness:harness why` notes that a held-out learned rule does not deny; `plugin.json` version is `0.8.0`.
- [ ] **Step 2: Verify they fail.**
- [ ] **Step 3: Update the docs** — plainly: what a receipt is, the holdout and its cost, how to set `receipts.json`, what `receipts report` verdicts mean, and that evidence accumulates over weeks of use.
- [ ] **Step 4: Live verification** (containment as before: a scratch dir whose name contains `cc-probe`, a throwaway `META_HARNESS_HOME`, never the real `~/.claude/harness` — confirm its sha256 is unchanged; `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1`; `--model haiku`):
  - A session with "Stop running pytest without -q." then `python -m pytest tests/` → denied, and `receipts-<session>.jsonl` contains a `session-rule` `acted` receipt with a `call` number.
  - A failing Bash call → `observed-<session>.jsonl` record carries `call`.
  - `python -m meta_harness receipts report` with that home prints the rows, with "no control arm" for the session rule.
  - The debug log shows the module loaded; `claude plugin validate` passes.
- [ ] **Step 5: Full suite, validate, commit** — `git commit -m "docs: decision receipts, the holdout and how to read the report"`.
