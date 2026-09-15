# Meta-Harness End-to-End Spec

Implements the system described in `paper.pdf` ("Meta-Harness: End-to-End Optimization of
Model Harnesses") on top of the existing `meta_harness` package, using **Unikey**
(getunikey.ai) as the model gateway.

## 1. Goal

A runnable outer loop that:

1. Seeds a population of baseline harnesses.
2. Runs a coding-agent proposer that reads the full experience filesystem and writes new
   candidate harness `.py` files.
3. Evaluates candidates on a **search split** only.
4. Reports a Pareto frontier over (accuracy, context cost) and evaluates that frontier once
   on a **held-out test split** the proposer never saw.

## 2. Model provider: Unikey

Verified from https://docs.getunikey.ai/docs/api-reference/ and https://docs.getunikey.ai/docs/intro/
on 2026-09-15:

- Base URL: `https://www.getunikey.ai`
- OpenAI-compatible: `POST /v1/chat/completions`, `GET /v1/models`, `POST /v1/embeddings`,
  `POST /v1/completions` (legacy).
- Auth: `Authorization: Bearer $UNIKEY_API_KEY`.
- Anthropic-compatible: `POST /v1/messages` with `x-api-key` — this is the endpoint Claude
  Code itself can be pointed at (`ANTHROPIC_BASE_URL=https://www.getunikey.ai`,
  `ANTHROPIC_AUTH_TOKEN=$UNIKEY_API_KEY`).
- Model ids are provider-qualified (`gpt-5.2`, `claude-sonnet-4-6`, `deepseek/deepseek-v4-flash`,
  `google/gemini-3.1-flash-lite`). `GET /v1/models` is authoritative; do not hard-code a list.
- Responses follow OpenAI shape including `usage.prompt_tokens` / `usage.completion_tokens`.
- Errors: `{"error": {"message", "type", "code"}}`.

The existing `OpenAICompatibleModel` already speaks this protocol. Unikey is a thin subclass
that changes the default base URL and environment variable.

## 3. Requirements

### R1 — Coding-agent proposer (paper §3 "Practical implementation")
The proposer is a coding agent with unrestricted read access to the experience filesystem.
It decides what to read. No parent-selection rule, no templated mutation operators. Its
guidance is a single minimal skill file describing the directory layout, the required harness
interface, where to write output, and what it must not touch.

### R2 — Search/test separation (paper §3)
"The proposer never sees test-set results." Search scores live inside the experience root.
Test scores are written outside it, once, at the end of the run.

### R3 — Seeded population (paper §4.1, §4.2)
Search is initialized from real baseline harnesses: zero-shot, few-shot(N), ACE, MCE for
classification; zero-shot and BM25 retrieval for math.

### R4 — Proposer reasoning is part of experience (paper §3, Fig. 2)
Each candidate directory stores the reasoning trace that produced it, next to its code,
scores, and execution trace.

### R5 — Candidate isolation
Candidate code is LLM-written. Interface validation runs in a subprocess under
`validation_timeout`. Evaluation enforces a per-candidate wall-clock deadline.

### R6 — Honest context cost (paper Table 2, Fig. 3)
Context cost is measured in input tokens, taken from provider `usage.prompt_tokens` when
available, otherwise estimated from characters. The current `trace.count` event-count
fallback is removed.

### R7 — Repeated evaluation
`--repeats N` averages N independent evaluations per candidate (paper reports pass@1 averaged
over three samples).

### R8 — Parallel evaluation
Candidates within an iteration evaluate concurrently. Harnesses are stateful across tasks
within one evaluation, so parallelism is across candidates and repeats, never across tasks
inside a single evaluation.

### R9 — Task-appropriate metrics
Classification uses exact match over the declared label set. Math uses answer equivalence:
`\boxed{}` extraction, LaTeX normalization, and numeric comparison with tolerance.

### R10 — Proposer-view ablation (paper Table 3)
`--proposer-view {scores,summary,full}` controls what the proposer can read:
- `scores`: harness source + scores only.
- `summary`: harness source + scores + an LLM-written summary per candidate, no raw traces.
- `full`: the entire experience root.

### R11 — Caching
Model calls are cached on disk keyed by (model id, request kwargs, prompt), so repeats and
re-runs do not re-bill.

### R12 — Agentic coding domain (paper §4.3)
A `TerminalHarness` wraps the existing bounded `TerminalAgent` so the same search loop
optimizes terminal-agent harnesses. Commands run inside a Docker container by default;
running them on the host requires an explicit opt-in flag.

## 4. Global constraints

- Python >= 3.10, standard library only. No new runtime dependencies (`pyproject.toml`
  declares none, and that stays true). `pytest` is dev-only.
- Windows-compatible: no symlinks, no POSIX-only calls, no `signal.alarm`.
- `.env` and `venv/` stay git-ignored. API keys are read from the environment only and must
  never be written into the experience filesystem, traces, or logs.
- Commits and PRs carry no AI/assistant co-author or attribution lines.

## 5. Out of scope

- Distributed / multi-machine evaluation.
- Resume-from-checkpoint of a partially completed run.
- Reproducing the paper's exact benchmark numbers (needs their datasets and budget).
