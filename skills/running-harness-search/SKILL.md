---
name: running-harness-search
description: Use when hand-tuning a prompt or agent has stalled after several attempts, when more than a handful of variants need comparing, or when someone asks to automate prompt/harness optimization rather than iterate by hand
---

# Running a harness search

## Overview

Past three or four hand-written candidates, you are the bottleneck and the archive gets sloppy.
The automated loop proposes, evaluates and logs candidates until the frontier stops moving.

**Core principle:** Escalate to the loop when the eval run is cheaper than your attention, not
before.

## When to Use

- Hand-tuning stalled: several candidates, no clear winner, diminishing returns
- More than ~5 variants worth comparing
- The user asked for automated prompt or harness optimization

**Do NOT use when:**
- No eval set exists yet — **REQUIRED SUB-SKILL:** `building-eval-sets`
- Fewer than 3 candidates have been tried by hand. Try them; you will often find it in two
- The artifact ships once and is never touched again
- Each eval costs minutes of wall clock and you have 10 cases — just run them yourself

The loop is not free: it spends model calls on every candidate against every case.

## What the loop does

1. Seeds a population from real baselines (zero-shot, few-shot, reflective memory, skill library).
2. A coding agent reads the whole archive — every prior candidate's source, score, trace and the
   reasoning that produced it — and writes the next candidate.
3. The candidate is validated, then scored on the **search split** only.
4. Everything is written to the archive. Repeat.
5. The Pareto frontier over (accuracy, context tokens) is scored once on the **held-out split**.

The proposer never sees held-out results. That is structural, not a convention.

## Running it

From the plugin root:

```bash
python -m meta_harness run \
  --tasks <eval.jsonl> \
  --task-type classification \
  --iterations 10 --candidates 1 --max-workers 4 \
  --root .meta-harness
```

Defaults: the harness model runs through the local `claude` CLI (`--model haiku`, no API key), and
the proposer is Claude Code itself. `--task-type` is `classification`, `math`, `terminal`, or
`agent` — the last searches over Claude Code's own scaffold, covered by
`optimizing-claude-code`.
`--provider unikey --model <id>` routes the harness model through a gateway instead, and
`--proposer-command "python tools/llm_proposer.py"` replaces the agent proposer with a plain API
call when no agent CLI is available.

Tell the user the model pair, iteration count and dataset size before launching. A run costs real
tokens and takes minutes.

## Reading the result

The run prints the frontier and the held-out path. For anything beyond that, use the
`meta-harness:harness-analyst` agent — it reads the archive and reports the mechanism, the
confidence, and whether the winner overfit.

Report the **held-out** number. When baselines already score at ceiling, the search can only
optimize context cost — say that plainly instead of narrating an accuracy win that did not happen.

## Terminal and agent harnesses

`--task-type terminal` searches over an agent's own scaffold: system prompt, step budget, how
execution history is fed back. Each row needs `instruction`, `test_command`, and an `image`;
commands run in that Docker container and a task scores 1.0 only if `test_command` exits 0.

Rows without an `image` are refused unless `--allow-local-shell` is passed, which runs
model-authored shell commands on this machine. Never add that flag on the user's behalf.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Reaching for the loop before trying 3 candidates by hand | Hand attempts are faster and usually enough |
| Running it with no eval set | There is nothing to optimize against |
| Reporting the search-split score | The held-out number is the result |
| Shipping the winner unread | LLM-written code overfits visibly; read it first |
| Treating a context-cost-only win as an accuracy win | Say which objective moved |

**REQUIRED BACKGROUND:** `optimizing-harnesses` — the three laws the loop enforces, and why.
