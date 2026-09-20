---
name: learning-from-failures
description: Use when the same failure keeps recurring in this repo - a tool erroring the same way, a loop of retries, a correction you have given before - or when someone asks whether the harness can be improved from what already went wrong
---

# Learning from failures

## Overview

Every failure Claude Code produces is evidence. A learn cycle turns one of them into a harness
artifact, scores it against the failure that produced it, and stages it for review.

**Core principle:** An artifact earns its place by stopping the specific failure it was born
from. An artifact whose origin replay still fails is archived, not staged. Pass `--tasks <path>`
to also require that it does not regress a task set; without one, that half of the gate does not
run and the verdict says so. Nothing is adopted on a score alone.

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

Selection draws on two sources of evidence: episodes mined from session history, and failures the
`tool.call` hook observed live during the current session, merged by failure signature. A failure
seen *only* through live observation has no mined episode to build a replay from, and `learn`
cannot score what it cannot replay: it prints `no replayable episode for this failure class` and
exits 1. That failure is now visible — the hook logged it — but not yet learnable. This is a known
gap, not a bug to work around; give it another session or two to accumulate mined history.

## Read the result before accepting

```bash
python -m meta_harness learn --accept <id>          # install it
python -m meta_harness learn --reject <id>          # archive it
python -m meta_harness learn --reject <id> --wrong  # archive and never propose it again
```

Accepting records the artifact in `installed.json`. Only `rule` and `injection` artifacts are
enforced from there today, by the `tool.check` and `prompt.section` hooks; an accepted `skill` or
`doctrine` artifact is recorded and scored but has no automatic enforcement path yet, so treat it
as a note to apply by hand.

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

This ordering is an argued design claim, not a measured one. `tools/prose_vs_rule.py` sets up the
comparison it would take to measure it, but ships as two pure functions with no live execution
wired up — the experiment has not been run in this repo. Where the paper reports numbers for
prose-vs-mechanism framing, they are the paper's numbers, not a result produced here.

Function hooks (`tool.call`, `tool.check`, `prompt.section`) are early access in Claude Code and
need `CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1`; see `hooks/README.md`.

**REQUIRED BACKGROUND:** `optimizing-harnesses` - the three laws this cycle enforces.
