# Drift and Waste Design

Status: approved for planning, 2026-09-26.

Supersedes nothing. Extends `2026-09-20-learning-harness-design.md`, and corrects two claims
that document makes about `paper.pdf`. Where the two disagree, this document wins.

The system this describes has one job: **cut the number of tool calls a developer loses to work
that was going the wrong way**, using evidence already sitting on their machine.

---

## 1. Goal

A developer working with Claude Code loses turns in three ways. Ranked by measured cost on this
machine, across 344 sessions and 31,566 tool calls:

| Loss | Measured |
|---|---|
| Wrong-direction work before a human intervenes | **1,463 calls across 98 corrections** — median 8 before the human spoke, p90 41, max 76 |
| Repeated identical failures | 378 wasted retries, concentrated in 24 of 344 sessions |
| Environment rediscovery | 50 calls across 344 sessions |

The first is an order of magnitude larger than the others and nothing in the Claude Code
ecosystem addresses it. It is invisible because **every call in a drifting stretch succeeds** —
there is no error to catch. The system detects that stretch and says so, early.

Success is a measurable fall in the correction-lag median and p90, computed by the same command
that produced the baseline, against the same machine's transcripts.

---

## 2. Evidence this is built on

Measured from `~/.claude/projects/**/*.jsonl` on 2026-09-26, 344 sessions, 31,566 tool calls:

- Error rate 3.7%, and most errors self-recover. Errors are cheap.
- 98 corrections that followed at least three tool calls. Median 8 calls burned, p90 41, max 76.
- 378 repeated identical failures. Top classes: re-proposing a rejected command (98), Chrome tab
  targeting (51), Bash quoting (24), read-before-edit (30, already fixed by a shipped rule).
- Orientation commands: 50 across all sessions.
- Sessions span Claude Code `2.1.233` through `2.1.278`.

### 2.1 Two corrections to the previous spec

`2026-09-20-learning-harness-design.md` attributes claims to `paper.pdf` that the paper does not
make. Both are corrected here and must be corrected in the docs.

**The layer ordering is ours, not the paper's.** `rule > injection > skill > doctrine` appears
nowhere in `paper.pdf`. It may still be a sound design position and we keep it — but as our
position, labelled as such.

**The paper's actual findings are different, and better evidenced:**

- *Raw traces beat summaries.* Table 3: scores-only 34.6 median, scores-plus-summary 34.9, full
  execution traces 50.0. Summaries "may even hurt by compressing away diagnostically useful
  details". This has a direct consequence for us — see §3.1.
- *Additive beats invasive.* Appendix A.2: the proposer regressed six consecutive times editing
  prompts and control flow, diagnosed the confound, then won with a purely additive change. Every
  intervention in this spec is additive: we add information, we do not rewrite prompts.

**One measured negative result.** The paper's winning discovery — an environment snapshot before
the first model call — does **not** transfer here. It earned its gain in fresh sandboxes where
the environment is unknown. In a developer's own repository the environment is already known:
50 orientation calls across 344 sessions. We do not build it, and the docs record why.

---

## 3. Architecture

Three phases. Each is useful alone; each later phase depends on the one before.

```
transcripts (~/.claude/projects)
        |
        v
  [1] experience + measurement  ---> thresholds, baseline, one-line onboarding
        |
        v
  [2] drift detection (3 judges) ---> tuned offline, silent until it clears the gate
        |
        v
  [3] live control + surface     ---> rejection memory, NL rules, /harness commands
```

### 3.1 The trace-fidelity defect, fixed first

`meta_harness/cc_history.py` writes failure episodes whose `matched` field is empty even under
`--include-text`. The episode records `"Bash returned an error"` — a summary — and discards the
error text. Every cause therefore classifies as `other`.

By Table 3 this is the losing ablation, implemented by accident. Every judge in Phase 2 depends
on this text. It is fixed before anything is built on it.

---

## 4. Phase 1 — Experience and measurement

### 4.1 `meta-harness waste`

Offline. No model calls. No network. Reads the transcript store, writes a report.

Reports:
- correction-lag distribution: count, median, p75, p90, max, total calls burned
- repeated identical failures: total wasted retries, top classes by cost, sessions affected
- rejection retries: re-proposals of an already-rejected call
- orientation cost
- per-project breakdown

Flags: `--this-project`, `--since <date>`, `--limit <n>`, `--json`.

Definitions, fixed so the number means one thing:

- A **stretch** is a run of assistant tool calls with no intervening user text.
- A **correction** is user text ending a stretch of at least 3 calls and matching the correction
  vocabulary (`no,` / `wrong` / `stop` / `revert` / `that's not` / `i said` / `undo` / `instead of`
  / `you broke` / `still broken` / `still failing` / `don't do` / `why did you` / `actually,`).
- **Correction lag** is the number of tool calls in that stretch.
- A **repeated identical failure** is the 2nd and later occurrence of the same
  `(tool, normalised error)` in one session. Normalisation replaces paths, hex ids and digits.

The correction vocabulary is a heuristic and will both over- and under-count. The command states
this in its own output. It is a cost estimate, not an audit.

### 4.2 Temporal layer

The store today has no notion of time. Three additions:

- **Recency weighting** in `merge_failures`: a signature's weight decays with the age of its
  evidence, so a class fixed six months ago stops outranking this week's.
- **Version proximity**: episodes carry the Claude Code version. Evidence from a distant version
  is down-weighted, because the tool it describes may no longer behave that way.
- **Artifact retirement**: an installed artifact whose origin failure class has not appeared for
  a configured period is proposed for retirement, never silently removed.

Decay parameters live in config, not code.

### 4.3 First-run bootstrap

On first session after install, mine existing history in the background and emit exactly one
line naming the measured cost and the command to see detail. It runs once, is silent on every
later session, and never blocks a turn.

---

## 5. Phase 2 — Drift detection

### 5.1 What fires

A checkpoint is considered when a stretch crosses a configured length. The judges decide.

**Temporal kNN (default).** Retrieve the k most similar past stretches by tool-sequence shape and
score by how many ended in a correction. No model call. Uses raw history, which is the paper's
actual finding.

**Path-overlap heuristic (fallback).** Fires when the paths being touched stop overlapping with
the anchor set — the paths touched in the first calls after the user last spoke. Used when
history is too thin for kNN, which is every new install.

**Model judge (opt-in).** A short call carrying the last user message and recent tool names and
targets — never file contents. If the capability is absent it falls back to the heuristic. The
`claude-code` type declarations are not present on this machine, so its availability is assumed,
not verified; the fallback is what makes that safe.

Selection is config. Default is kNN with heuristic fallback. The model judge is off by default.

### 5.2 How it surfaces

One line injected through `prompt.section` on the following turn, naming how many calls have
passed and what the work has moved toward.

It never blocks a tool call. It never interrupts mid-tool. It does not depend on
`turn.complete`, which `hooks/README.md` records as sketched and unimplemented.

### 5.3 Tuning, and the ship gate

Replay every judge over the 344 sessions:

- **Recall**: of the 98 corrections, how many would have fired before the human spoke, and how
  many calls earlier.
- **Precision**: how often it fires in stretches that did not end in a correction.

Thresholds live in config. A judge ships **only** if it clears a precision bar stated in the plan
and fires meaningfully earlier than the human did. A judge that cannot is reported as not
shipping, not tuned until it passes.

n = 98 is thin. The plan states the bar in advance, and no judge speaks to a user until it clears
it on this data.

---

## 6. Phase 3 — Live control and surface

### 6.1 Rejection memory

The largest single repeated failure is re-proposing a rejected call: 98 wasted retries. The
observer records the rejection signature; `tool.check` denies an immediate re-proposal and cites
the earlier rejection. Memory clears when the user raises that command again, so changing their
mind works. No model, no tuning.

### 6.2 Natural-language control

"Stop doing X" / "don't run that again" takes effect **immediately and for this session only**.
At session end the user is asked once whether to keep it. Nothing becomes permanent without that
answer.

### 6.3 Commands

Three, deliberately:

- `/harness waste` — the report
- `/harness pending` — review and accept what has been learned
- `/harness why` — explain the note or block just seen, and retire the artifact behind it

Natural-language triggers carry the rest: asking why Claude keeps failing runs the report; asking
whether you are on track runs a drift check; asking why something was blocked explains the
artifact.

---

## 7. Constraints

- Python stdlib only. `dependencies = []` stays empty. No new dependencies, no build step.
- Windows-compatible. Explicit `encoding="utf-8"` on every file read and write. Every
  `subprocess.run(..., text=True)` also passes `encoding="utf-8", errors="replace"`, enforced by
  the existing repo-wide AST test.
- Every hook path fails open. A throwing judge, a corrupt store and a missing config all leave
  the turn proceeding. The one deliberate exception is a `tool.check` denial, which is the
  feature working.
- Transcripts never leave the machine. The store keeps signatures and counts, not message text.
- Sibling TypeScript imports use `.js` specifiers; the test-only resolver under `hooks/loaders/`
  stays as it is.
- Tests are hermetic and offline, and must discriminate: each is verified to fail against a
  deliberately broken implementation before it is committed.
- No AI attribution in commits, authors, comments, docs or contributor lists.

---

## 8. Out of scope

- Temporal graph learning, sequence models, embedding stores. The signal is 98 corrections;
  lexical and structural similarity beats anything learned at this scale.
- Environment bootstrap. Measured not to transfer (§2.1).
- `skill` and `doctrine` enforcement. Still scored and recorded, still not enforced by any hook.
- Running the prose-versus-mechanism experiment. It remains built and unrun.
- Harness search over Claude Code's own artifacts. A later spec.

---

## 9. Risks

- **n = 98 is thin for tuning.** Mitigated by stating the bar before tuning and by the ship gate.
- **False positives end adoption faster than misses.** Mitigated by biasing toward long silent
  stretches, by a single injected line rather than a block, and by `/harness why` making every
  intervention explainable and reversible.
- **The correction detector is a keyword heuristic.** It will miscount. The command says so in
  its output, and the tuning set inherits that noise.
- **The model judge depends on an unverified capability.** Mitigated by the fallback and by
  shipping it off by default.
- **Retirement could remove an artifact still doing its job silently** — an artifact that works
  prevents its own evidence. Mitigated by proposing retirement rather than performing it.
