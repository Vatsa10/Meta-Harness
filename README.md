# Meta-Harness

Filesystem-backed end-to-end optimization of executable LLM harnesses, following
*Meta-Harness: End-to-End Optimization of Model Harnesses* (`paper.pdf`).

A **harness** is the code around a fixed model: what it stores, retrieves, and shows the model
at each step. This repository searches over that code. A coding-agent proposer reads the full
experience filesystem — every prior candidate's source, scores, execution traces, and the
reasoning that produced it — and writes new candidates. Candidates are evaluated on a search
split; the Pareto frontier over (accuracy, context tokens) is scored once on a held-out split
the proposer never sees.

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
- `docs/using-with-coding-agents.md` — Claude Code as proposer, optimizing agent harnesses,
  and shipping a discovered harness back into your own agent.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` — the spec and task-by-task plan
  this implementation follows.

## Tests

```bash
python -m pytest tests -q
```

78 tests, no network calls.
