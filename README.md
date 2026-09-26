# Meta-Harness

Filesystem-backed end-to-end optimization of executable LLM harnesses, following
*Meta-Harness: End-to-End Optimization of Model Harnesses* (`paper.pdf`).

A **harness** is the code around a fixed model: what it stores, retrieves, and shows the model
at each step. This repository searches over that code. A coding-agent proposer reads the full
experience filesystem — every prior candidate's source, scores, execution traces, and the
reasoning that produced it — and writes new candidates. Candidates are evaluated on a search
split; the Pareto frontier over (accuracy, context tokens) is scored once on a held-out split
the proposer never sees.

## Use it from Claude Code

The repository is also a Claude Code plugin. Not a wrapper around the CLI — model-invoked
skills that change how Claude works on prompts, agent scaffolds and retrieval, plus a hook layer
that observes failures and enforces what has been learned from them:

```bash
claude --plugin-dir .
```

`optimizing-harnesses` carries the discipline (one change per candidate, every candidate kept, no
score without a trace, held-out split read once). `building-eval-sets`, `reading-execution-traces`
and `running-harness-search` handle the pieces, `optimizing-claude-code` points the search at
Claude Code's own scaffold, and `learning-from-failures` drives the loop below. The engine is the
escalation path when hand-tuning stalls; the skills need nothing installed.

See [docs/plugin.md](docs/plugin.md) for the skill list, the baseline testing behind it, and why
Haiku 4.5 is the default harness model.

### Learning from failures

```bash
python -m meta_harness learn            # failure -> artifact -> replay-scored -> staged
python -m meta_harness learn --tasks tasks.json   # also require no regression on a task set
python -m meta_harness learn --status
python -m meta_harness learn --accept <id>
```

Three layers. **Observe:** a `tool.call` hook records tool errors and repeated calls to a
per-session log. **Learn:** the CLI above ranks those failures together with ones mined from past
transcripts, proposes one artifact at the strongest layer that can carry it, and replays it.
**Enforce:** a `tool.check` hook applies accepted rules and a `prompt.section` hook injects
accepted context.

An artifact is staged only if it fixes the failure it was born from; one whose origin replay
still fails is archived with the verdict that killed it. With `--tasks`, it must also score no
worse than the baseline on that task set, with context cost as the tiebreak; without `--tasks`,
that check does not run and the recorded verdict says so. Nothing installs itself.

Artifacts sit at four layers, strongest first: `rule` (a `tool.check`, costing no standing
tokens and not ignorable) > `injection` (a prompt section, paid every turn) > `skill` >
`doctrine`. Only `rule` and `injection` are enforced by the hooks today. **That ordering is this
project's own design position — it appears nowhere in `paper.pdf`.** `tools/prose_vs_rule.py` is
the experiment that would test it against a measured baseline, and it has not been run here.

What the paper does establish (Table 3, online text classification, median/best score): a
proposer given scores only reaches 34.6/41.3, scores plus an LLM summary reaches 34.9/38.7, and
full raw execution traces reach 50.0/56.7. Raw trace access is the paper's key ingredient — a
summary "may even hurt by compressing away diagnostically useful details." Appendix A.2 adds a
second, independently evidenced principle: on TerminalBench-2 the proposer regressed six
consecutive iterations while editing prompts and control flow, diagnosed the shared prompt edit
as the confound, and then won with a purely additive change. Additive beats invasive.

One measured negative result of our own: the paper's winning TerminalBench-2 discovery was an
environment snapshot injected before the first model call. That does not transfer to a
developer's own repository, where the environment is already known — 50 orientation calls across
344 sessions on this machine. We do not build it.

The hooks need Claude Code's function-hook surface, which is early access:

```bash
export CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1   # PowerShell: $env:CLAUDE_CODE_ENABLE_FUNCTION_HOOKS="1"
claude --plugin-dir .
```

See [hooks/README.md](hooks/README.md) for the hook layer and
[the design](docs/superpowers/specs/2026-09-20-learning-harness-design.md) for the rest.

## Install

```bash
python -m venv venv
venv/Scripts/activate          # Windows;  source venv/bin/activate elsewhere
pip install -e .
pip install -e ".[dev]"        # adds pytest
```

No runtime dependencies. Python 3.10+.

## Configure the model gateway (Unikey)

All model traffic goes through [Unikey](https://www.getunikey.ai), an OpenAI-compatible
gateway (`https://www.getunikey.ai/v1`, `Authorization: Bearer $UNIKEY_API_KEY`).

```bash
export UNIKEY_API_KEY=sk-...            # PowerShell: $env:UNIKEY_API_KEY="sk-..."
meta-harness models --provider unikey   # GET /v1/models
```

The proposer is a coding agent, billed separately. Unikey also exposes an
Anthropic-compatible `/v1/messages`, so Claude Code can be routed through it:

```bash
export ANTHROPIC_BASE_URL=https://www.getunikey.ai
export ANTHROPIC_AUTH_TOKEN=$UNIKEY_API_KEY
```

`--provider openai|anthropic|compatible` still work, reading `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY` and their `*_BASE_URL` overrides.

## Run a search

```bash
meta-harness run \
  --tasks data/classification.jsonl \
  --task-type classification \
  --provider unikey --model gpt-5.2 \
  --iterations 20 --candidates 2 --repeats 3 --max-workers 4 \
  --proposer-model claude-sonnet-4-6 \
  --root .meta-harness
```

Dataset formats: `.jsonl`, `.json`, `.csv`.

| `--task-type` | required fields | optional | metric |
|---|---|---|---|
| `classification` | `input`, `label` | `labels` | exact match |
| `math` | `problem` | `answer` | `\boxed{}` + numeric equivalence |
| `terminal` | `instruction` | `test_command`, `image`, `workdir`, `timeout` | test command exit code |

Without `--test-tasks`, `--tasks` is split 70/30 (`--search-fraction`, `--split-seed`).
Without `--baseline`, the seeds in `baselines/` for that task type are used.
Without `--proposer-command`, the proposer is the `claude` CLI; use `--proposer-command` to
drive any other agent, or `python tools/llm_proposer.py` for a plain-API proposer that needs no
agent installed. See [docs/using-with-coding-agents.md](docs/using-with-coding-agents.md).

Two sample datasets ship in `data/`.

## What a run produces

```
.meta-harness/
  run.json                       config + the search-split tasks
  frontier.json                  Pareto frontier over (score, context_cost)
  candidates/<id>/harness.py     candidate source
  candidates/<id>/scores.json    score, context_cost, repeats, score_std, valid, error
  candidates/<id>/traces.jsonl   every model call, task start/end, harness event
  candidates/<id>/traces-N.jsonl additional repeats
  candidates/<id>/proposer_reasoning.md
  proposals/iteration-NNNN/      proposer stdout/stderr
  views/iteration-NNNN/          what the proposer was allowed to read (ablation modes)
.meta-harness-test/
  test_results.json              held-out scores, written once, outside the proposer's reach
.meta-harness-cache/             model-call cache, keyed by (model, prompt, kwargs)
```

Inspect a finished run: `meta-harness inspect .meta-harness`

## Proposer-view ablation (paper Table 3)

```bash
meta-harness run ... --proposer-view scores    # source + scores only
meta-harness run ... --proposer-view summary   # + LLM summaries, no raw traces
meta-harness run ... --proposer-view full      # everything (default)
```

## Agentic coding domain

Terminal tasks run the model's commands inside Docker:

```bash
meta-harness run --task-type terminal --tasks data/terminal.jsonl \
  --provider unikey --model claude-sonnet-4-6
```

Each row needs an `image`. Rows without one are refused unless you pass
`--allow-local-shell`, which executes model-authored commands **on your machine** — only do
that in a throwaway environment.

## Writing your own harness

```python
class Harness:
    def run(self, task, model, trace):
        prompt = f"Classify: {task['input']}"
        trace.event("prompt", {"prompt": prompt})
        return model(prompt)
```

The instance persists across the tasks of one evaluation (that is your memory) and is recreated
per evaluation. `model(prompt) -> str`; `model.call(prompt) -> (str, usage)`. Every call is
priced in input tokens and written to the trace. `meta_harness.retrieval` provides
`TfidfIndex`, `BM25Index`, and `reciprocal_rank_fusion`.

Seven working examples live in `baselines/`; the exact contract handed to the proposer is
`meta_harness/skill/SKILL.md`.

## Design notes

- `meta_harness/core.py` — the outer loop, experience filesystem, Pareto frontier, metering.
- `meta_harness/sandbox.py` — interface validation in a subprocess under a timeout, so an
  LLM-written `while True` cannot hang the search.
- `meta_harness/agent_proposer.py` — the coding-agent proposer and the view ablation.
- `meta_harness/cache.py` — disk cache so `--repeats` and re-runs do not re-bill.
- `tools/llm_proposer.py` — proposer that needs only an API key, no coding-agent CLI.
- `meta_harness/replay.py` — failure signatures, and the replay a proposed artifact is scored on.
- `meta_harness/learn.py` — ranking, proposal, and the retention decision.
- `meta_harness/harness_store.py` — staged and installed artifacts, and the accept/reject gate.
- `hooks/` — the function-hook layer: observe, enforce, inject. Fails open by construction.
- `tools/prose_vs_rule.py` — the prose-versus-mechanism experiment (built, not yet run).
- `docs/plugin.md` — the Claude Code plugin: skills, agent, install, Haiku defaults.
- `docs/using-with-coding-agents.md` — Claude Code as proposer, optimizing agent harnesses,
  and shipping a discovered harness back into your own agent.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — the spec and task-by-task plan
  this implementation follows.

## Tests

```bash
python -m pytest -q
```

252 tests, no network calls. The hook layer is covered by Node tests under `hooks/` that the
Python suite shells out to; they are skipped, with the reason stated, when `node` is absent.
