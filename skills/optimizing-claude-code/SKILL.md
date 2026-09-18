---
name: optimizing-claude-code
description: Use when Claude Code itself keeps failing a kind of task in this repo - stopping at partial fixes, missing project conventions, burning turns, editing tests to pass - or when someone wants to improve a CLAUDE.md, skill, subagent or agent scaffold rather than accept it as given
---

# Optimizing Claude Code's own harness

## Overview

Claude Code is a harness: a system prompt, a tool set, a turn budget, and a framing around a
frozen model. In the paper's Table 7 it is one of the compared baselines — 58.0% on Opus 4.6,
27.5% on Haiku 4.5 — and searched harnesses beat it by 18.4 and 10.1 points **on the same model**.
That gap is scaffolding, not intelligence.

**Core principle:** When Claude Code fails a task repeatedly, the fixable thing is usually the
harness, not the model. Treat CLAUDE.md, skills and agent definitions as candidates to be scored,
not as settings to be argued about.

## When to Use

- The same class of failure keeps recurring in this repo
- Someone is editing CLAUDE.md or a skill by feel and wants to know whether it helped
- Choosing between two agent scaffolds, turn budgets or tool sets
- A cheaper model would do if the scaffolding were better

**Skip when:** it happened once, or the fix is a missing fact that belongs in CLAUDE.md as plain
documentation. Not everything is a search problem.

## What counts as the harness here

| Knob | Example change |
|---|---|
| Appended system doctrine | "Read the tests before editing. Never stop at a partial fix." |
| Instruction framing | Prepending the file list, or restating the acceptance criterion last |
| Tool set | Denying `Bash` so the agent edits rather than shells out |
| Turn budget | 15 vs 30 vs 60 turns |
| Subagents | A dedicated reviewer agent invoked before finishing |
| Model | Haiku with better doctrine vs Sonnet with none |

All six are searchable. Arguing about which is best without scoring them is the thing this skill
exists to stop.

## Build the benchmark from real failures

The tasks are the hard part, and they must come from your repo:

```jsonl
{"instruction": "<what a developer would have asked>",
 "files": {"src/thing.py": "<the broken state>"},
 "test_files": {"tests/test_thing.py": "<the verification>"},
 "test_command": "python -m pytest -q"}
```

**Hidden tests are the point.** `files` is seeded into a throwaway workspace before the agent
starts; `test_files` is written only at scoring time, after the session ends. The agent cannot
read the assertions to reverse-engineer the spec, and cannot edit the tests to pass. Put the
requirement in the instruction as prose, exactly as a developer would state it.

Source tasks from failures you actually had: the PR that came back wrong, the refactor that
missed a call site, the bug fix that broke a neighbour. Six to ten real ones beat fifty invented.

**Check for headroom before you search.** Run the baseline first:

```bash
python -m meta_harness run --task-type agent --tasks tasks.jsonl --iterations 0 \
  --root .meta-harness-agent
```

If the default harness already passes everything, there is nothing to find and the only live
objective is cost. Say so and either write harder tasks or stop. A benchmark with no headroom
produces a search that optimizes tokens and a report that sounds like an accuracy win — the most
common way this work goes wrong.

## Running it

```bash
python -m meta_harness run \
  --task-type agent --tasks tasks.jsonl \
  --iterations 10 --candidates 1 --max-workers 2 \
  --root .meta-harness-agent
```

Seeds are `baselines/cc_default.py` (Claude Code as shipped — the Table 7 row) and
`baselines/cc_guided.py` (hand-written doctrine). Each candidate is a `ClaudeCodeHarness`
subclass returning an `AgentConfig`. Every session runs in a throwaway workspace; your repository
is never touched.

Keep `--max-workers` at 2 or 3: each concurrent candidate is a live Claude Code session.

## Reading the traces

`agent_step` events carry the whole transcript — the agent's text, every tool call with its
input, every tool result. That is where the failure is legible, and it is what the scores cannot
tell you:

| Trace pattern | Harness fix it suggests |
|---|---|
| Edited a file it never read | Doctrine: read before write |
| Declared done with the requirement unmet | Doctrine: restate and check the criterion before stopping |
| Burned turns re-reading the same file | Framing: put the file list in the prompt |
| Hit `error_max_turns` | Turn budget, or doctrine that wastes turns |
| Left a TODO or a stub | Explicit prohibition in the doctrine |

**REQUIRED SUB-SKILL:** `reading-execution-traces` for the method.

## Shipping the winner

The winning candidate's `append_system_prompt` is the artifact. It becomes your CLAUDE.md
section, your skill body, or your subagent's system prompt — carrying the run root, candidate id
and held-out score in a comment so the claim stays checkable.

Audit it first: doctrine naming your benchmark's specific files or functions is memorization, and
it will not transfer to the next task.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Inventing tasks instead of using real failures | The distribution you invent is the one you already pass |
| Shipping tests inside `files` | The agent reads the assertions and writes to them |
| Reporting a cost win as an accuracy win | Name which objective moved |
| Searching before checking headroom | Run `--iterations 0` first |
| Ten concurrent workers | Ten live agent sessions; keep it to 2-3 |

**REQUIRED BACKGROUND:** `optimizing-harnesses` — the three laws this loop enforces.
