---
name: optimize-agent
description: Search over a terminal/coding-agent's own harness - system prompt, step budget, how execution history is fed back - scored by whether each task's test command passes. Use when the user wants to improve an agent scaffold, tune an agent's system prompt against real tasks, or reproduce the paper's TerminalBench-style result with Haiku 4.5.
---

# Optimize an agent's harness

This is the paper's §4.3 setup: a strong proposer tunes a weaker executor's scaffolding. On
TerminalBench-2 with **Claude Haiku 4.5** as the executor, searched harnesses reached 37.6%,
above every hand-engineered agent reported on that model (Goose 35.5, Terminus-KIRA 33.7,
Mini-SWE-Agent 29.8, Terminus 2 28.3, Claude Code 27.5, OpenHands 13.9). That is why Haiku is
the default executor here — the headroom the search exploits lives in the scaffolding.

## Task file

One JSON object per line. `image` is required in practice:

```jsonl
{"instruction": "Fix the failing test in /app so pytest passes", "test_command": "pytest -q", "image": "python:3.12", "workdir": "/app"}
{"instruction": "The CLI crashes on empty input. Make it exit 0 with a usage message.", "test_command": "./run_tests.sh", "image": "node:22", "workdir": "/srv"}
```

Each task starts its own container, the agent's commands run inside it via `docker exec`, and
the task scores 1.0 only if `test_command` exits 0. Build images that already contain the
broken state; the harness does not set tasks up for you.

**Safety.** A row with no `image` is refused unless `--allow-local-shell` is passed, which runs
model-authored shell commands on this machine. Never add that flag on the user's behalf. If they
ask for it, confirm they are in a disposable environment first, and say plainly what it does.

## Run

```bash
cd "${CLAUDE_PLUGIN_ROOT}"
docker info >/dev/null || echo "Docker is not running - start it first"

python -m meta_harness run \
  --task-type terminal \
  --tasks <tasks.jsonl> \
  --provider unikey --model claude-haiku-4-5-20251001 \
  --proposer-model claude-opus-4-8 \
  --iterations 15 --candidates 1 --max-workers 2 \
  --root .meta-harness-agent
```

The seed is `baselines/terminal_basic.py`, a Terminus-2-shaped loop. What the proposer edits:

- the bootstrap system prompt (how the agent is told to inspect, act, and verify)
- `max_steps` — the step budget before the loop gives up
- how execution history is packed back into each prompt (the usual win: the naive seed re-sends
  the entire history as JSON every turn, which is both expensive and distracting)
- when to declare done, and whether to self-verify first

Keep `--max-workers` low: every concurrent candidate holds its own container.

## Cost

Terminal tasks are the expensive domain — each task is many model calls plus container time.
Start with 5-10 tasks and 5 iterations to confirm the loop runs end to end, then scale. Model
calls are cached by `(model, prompt, kwargs)`, so re-running an unchanged candidate is free;
container work is not.

## Report

Per-candidate pass rate, plus which tasks flipped from fail to pass between the seed and the
winner. A harness that only wins by spending more steps is a different claim from one that wins
by prompting better — check `metrics.model_calls` before crediting the prompt.
