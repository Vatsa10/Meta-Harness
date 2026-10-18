# Production Readiness Design

Status: approved by owner 2026-10-18 ("we do this all today, and make this prod ready").
Builds on `2026-10-11-standalone-engine-design.md`.

## Goal

A production agent team can adopt Meta-Harness without writing a JSONL eval set or rewriting
their agent:

1. They point it at their agent's production traces.
2. It builds an eval set.
3. It searches the knobs of their agent through an adapter.
4. It guards the winner in CI.
5. It rolls the winner out behind a measured holdout.

## Components

All components use only the stdlib. Each is a new module plus a command registered through
the command registry.

### P1. Trace import (`meta_harness/traceimport.py`, `commands/traces.py`)

`meta-harness traces import --format otel|langfuse|jsonl --in PATH --out evals.jsonl`

- **Inputs:**
  - OpenTelemetry JSON exports, using the GenAI semantic-convention attributes
    `gen_ai.prompt`/`gen_ai.completion` or `gen_ai.input.messages`/`gen_ai.output.messages`.
  - Langfuse trace export JSON, with observations, input/output and scores.
  - Generic JSONL with `{input, output, feedback?, error?}`.
- **Output:** one task per trace: `input`, the original `output` as `reference_output`,
  `criteria` (from feedback/score/error), and `source_trace_id`. The `criteria` field makes
  the existing blind judge metric usable.
- **Selection:**
  - Failures first: errors, low scores, negative feedback.
  - Deduplication by normalised input.
  - A seeded split into train/search/test.
- **Privacy:** redaction runs over every text field. An optional `--pii` scrub covers emails,
  phone numbers and card-like digits.

`meta-harness traces stats` summarises what was found.

### P2. Agent adapters: config harness (`meta_harness/adapters/`)

Production agents live in LangGraph, the OpenAI Agents SDK, the Claude Agent SDK and custom
services. We do not import those frameworks. Instead the harness becomes a config.

- **The config:** a JSON file holding prompts, tool descriptions, few-shot examples, routing
  thresholds, retrieval k and memory policy.
- **The user's entrypoint:** `module:function(task, config, model) -> dict`. It runs their
  real agent with that config.
- **The candidate:** a generated `harness.py` that loads its sibling `config.json` and calls
  the entrypoint.
- **What the proposer edits:** in config mode it edits only `config.json`. A schema with
  types and bounds is validated by the gate. Code edits are refused when `--edit config`
  is set.
- **Provided:**
  - `adapters/config_harness.py`, the runtime.
  - `meta-harness adapt init --entrypoint mod:fn --config cfg.json`, which scaffolds the
    seed.
  - Examples for a LangGraph-style and an Agents-SDK-style app, written as plain-Python
    stand-ins with no framework dependency, plus docs showing the real wiring.

### P3. CI gate (`commands/check.py`)

`meta-harness check --baseline A.py --candidate B.py --data fixed.jsonl --metric M ...`

- Evaluates both harnesses on the same fixed set with the same model.
- Reports the paired McNemar test (via `stats.compare`), the mean score delta, and the cost
  and context-token deltas.
- Exits non-zero when quality drops significantly, when quality drops by more than
  `--max-drop`, or when cost rises by more than `--max-cost-increase`. Exits 0 otherwise.
- Writes `check.json` and a markdown summary for PR comments.
- Ships a GitHub Actions example at `examples/ci/meta-harness-check.yml`.

### P4. Online rollout (`meta_harness/rollout.py`, `commands/rollout.py`)

- **Router:** `rollout.Router(control, treatment, percent, holdout_salt)`.
  - Assignment is a deterministic hash of the request or user id.
  - `route(key)` returns the arm.
  - `record(key, arm, outcome, cost, latency)` appends redacted JSONL receipts.
- **Report:** `meta-harness rollout report --log PATH` gives per-arm success rate, cost and
  latency, using a two-proportion z-test and a Fisher exact test (stdlib), plus a verdict
  (`promote` / `hold` / `rollback`) against thresholds.
- This revives the receipts-with-holdout pattern for production traffic.

### P5. Proof run (blocked on credits)

- A fair metaharness vs best-of-n comparison on Symptom2Disease with 100/100 tasks and
  2 seeds.
- A TerminalBench search on about 20 tasks.
- Runs when a provider with credits is available. The README states the result either way.

## Constraints

- Same as the standalone spec §9.
- Each component's tests run offline with the scripted provider.
- Commits dated as the owner asks.
- No AI attribution.
