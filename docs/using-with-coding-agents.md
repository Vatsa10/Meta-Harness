# Using Meta-Harness with coding agents

Three distinct ways a coding agent and this repository meet. They are independent — pick the
one that matches what you are trying to do.

## 1. Claude Code as the proposer (the paper's setup)

The search loop needs something that can read a filesystem larger than any context window and
decide what to look at. That is exactly what a coding agent does. This is the default proposer.

```bash
export UNIKEY_API_KEY=sk-...
export ANTHROPIC_BASE_URL=https://www.getunikey.ai   # route Claude Code via Unikey
export ANTHROPIC_AUTH_TOKEN=$UNIKEY_API_KEY

meta-harness run \
  --tasks data/ticket_intents.jsonl \
  --provider unikey --model google/gemini-3.1-flash-lite \
  --proposer-binary claude --proposer-model claude-opus-4-8 \
  --iterations 20 --candidates 2 --root .meta-harness
```

Per iteration the runner invokes:

```
claude -p "<proposer prompt>" --permission-mode acceptEdits --model claude-opus-4-8
```

with the working directory set to a view of the experience filesystem and these variables set:

| Variable | Meaning |
|---|---|
| `META_HARNESS_ROOT` | the view the agent may read |
| `META_HARNESS_OUTPUT` | the only directory it may write into |
| `META_HARNESS_ITERATION` | iteration number |
| `META_HARNESS_COUNT` | how many candidates to produce |
| `META_HARNESS_VIEW` | `scores` / `summary` / `full` |

The contract it follows is `meta_harness/skill/SKILL.md`, copied into that directory each
iteration. Nothing else is prompt-engineered: the agent greps, cats, forms a hypothesis, and
writes `candidate-NN.py` plus `candidate-NN.reasoning.md`. That reasoning file is filed with
the candidate, so the next iteration's agent can read why the last change was made.

**Any agent works.** `--proposer-command` replaces the Claude Code invocation with anything
that writes `.py` files into `$META_HARNESS_OUTPUT`:

```bash
meta-harness run ... --proposer-command "codex exec --full-auto 'follow SKILL.md'"
meta-harness run ... --proposer-command "python tools/llm_proposer.py"   # plain API, no agent
```

`tools/llm_proposer.py` is the no-agent fallback: it packs a bounded slice of the experience
into one prompt and asks a gateway model. Weaker than an agent (it cannot choose what to read
— the paper's Table 3 is about exactly that gap), but it needs no extra tooling.

## 2. Optimizing a coding agent's own harness

The thing being optimized can itself be an agent loop. `--task-type terminal` searches over
terminal-agent harnesses: system prompt, step budget, how execution history is summarized back
into the next prompt, when to stop.

```jsonl
{"instruction": "Fix the failing test in /app and make pytest pass", "test_command": "pytest -q", "image": "python:3.12"}
```

```bash
meta-harness run --task-type terminal --tasks data/terminal.jsonl \
  --provider unikey --model claude-haiku-4-5-20251001 \
  --proposer-model claude-opus-4-8 --iterations 20
```

Commands run inside the named Docker image; `terminal_metric` scores a task by running its
`test_command` and checking the exit code. This is the shape of the paper's §4.3 result, where
a searched harness beat hand-engineered agents on TerminalBench-2 — a strong proposer tuning a
weaker executor's scaffolding.

Rows without an `image` are refused unless you pass `--allow-local-shell`, which runs
model-authored commands on your machine. Use a throwaway environment if you do.

## 3. Shipping a discovered harness back into your agent

A search run produces readable Python, not weights. The frontier is in
`.meta-harness/frontier.json`; each entry points at `candidates/<id>/harness.py`.

```bash
meta-harness inspect .meta-harness
cat .meta-harness/candidates/<winning-id>/harness.py
cat .meta-harness/candidates/<winning-id>/proposer_reasoning.md
```

The prompt shape, retrieval policy, and memory rules in that file are the transferable part —
lift them into your application, or turn the harness into a skill your agent loads. The paper's
§4.2 result is that the same discovered retrieval policy transferred across five models it was
never searched against, so a harness found with a cheap model is often worth keeping when you
upgrade.

## Cost control

- Every model call is cached on disk by `(model, prompt, kwargs)`, so `--repeats` and re-runs
  of the same candidate are free. The run summary prints hits/misses.
- Search-set size drives cost linearly. Start with 20-40 tasks.
- The proposer is usually the expensive model and the executor the cheap one — that asymmetry
  is the point.
- `--max-workers N` evaluates candidates concurrently (never tasks within one evaluation;
  harnesses are stateful across the task stream by design).
