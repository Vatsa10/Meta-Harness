# Standalone Meta-Harness Engine Design

Status: approved for planning, 2026-10-11 (owner: "end to end do this all").

Pivot: this repository stops being a Claude Code plugin and becomes a standalone, plug-and-play
implementation of *Meta-Harness: End-to-End Optimization of Model Harnesses* (`paper.pdf`,
arXiv 2603.28052). Gap analysis behind this spec: `.superpowers/pivot/paper-gap.md`.

---

## 1. Goal and success

A user with an LLM pipeline and an eval set runs:

```bash
pip install -e .
meta-harness search --harness my_harness.py --data evals.jsonl --metric exact \
    --provider anthropic --model claude-haiku-4-5-20251001 --iterations 10
```

and gets a Pareto frontier of harnesses over (score, context tokens), each scored once on a
held-out split the proposer never saw, plus an auditable experience directory explaining every
candidate. No Claude Code, no `claude` CLI, no plugin.

Success:
1. The engine CLI imports and runs with only the Python stdlib and an API key.
2. The proposer is a tool-using agent over the experience filesystem (paper p.2, p.4: it
   *chooses what to read*), working with any supported provider.
3. The six evaluation bugs in the gap analysis are fixed and pinned by tests.
4. One paper domain — online text classification — runs end to end on a real public dataset with
   the paper's protocol, and a scripted-model smoke run proves the full pipeline offline.
5. The Table 3 ablation (scores / summary / full traces) and an equal-budget Best-of-N baseline
   are runnable from the CLI.
6. README explains what it is, how to use it, and how it differs.

## 2. Removed (the Claude Code layer)

Deleted in one commit (recoverable from git history):
`hooks/`, `commands/`, `.claude-plugin/`, `agents/`, `skills/` (the proposer skill moves into the
package, §4), `meta_harness/cc_harness.py`, `cc_history.py`, `claude_cli.py`, `learn.py`,
`learn_prompt.md`, `waste.py`, `drift.py`, `receipts.py`, `replay.py`, `temporal.py`,
`harness_store.py`, `baselines/cc_*.py`, `tools/tb_run.py`, `tools/tune_drift.py`,
`tools/prose_vs_rule.py`, `docs/plugin.md`, `docs/using-with-coding-agents.md`,
`docs/baseline-2026-09-18.json`, `docs/discovered-harness.md`, `tsconfig.json`, and every test
that only covers those. `__main__.py` keeps only engine commands. `keel/` is untouched and never
committed.

## 3. Plug-and-play interface

- **Harness file:** a Python file defining `class Harness` with `run(self, task, model) -> dict`
  (returns at least `{"output": ...}`), optional `learn(self, task, result)` for online domains.
  Existing `harnesses.py` shape is kept; the doc is the contract.
- **Data:** JSONL, one task per line, any fields; split into search/test by `--test-fraction`
  (default 0.3, seeded) or by explicit `--search`/`--test` files.
- **Metric:** a built-in name (`exact`, `contains`, `numeric`) or `module:function` taking
  `(task, result) -> float`.
- **Provider:** `--provider anthropic|openai|compatible` + `--model`; keys from the usual env vars
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`). A `scripted` provider replays a
  JSONL of canned responses for offline tests and demos.
- Separate `--proposer-provider/--proposer-model` (paper: strong proposer, cheaper base model).

## 4. Tool-using proposer (the core of the paper)

`meta_harness/proposer_agent.py`: a model-agnostic agent loop. The proposer gets the task
description and a tool set confined to the run directory:

- `list(path)`, `read(path, offset, limit)`, `grep(pattern, path)` — read-only over the whole
  experience filesystem (all prior candidates' source, scores, traces, reasoning).
- `write_candidate(source, rationale)` — the only write; ends the turn.

Max tool calls per proposal is bounded (default 40). Every tool call is logged to the candidate's
`proposer_log.jsonl` (paper Table 8: what the proposer read). Provider tool-calling is implemented
for Anthropic Messages and OpenAI chat-completions in `providers.py`. The skill/instructions text
lives in `meta_harness/proposer_prompt.md` with domain sections. Paths outside the run directory
and the test split are refused by the tool layer (fixes "isolation by convention").

Table 3 views become a `--view scores|summary|full` flag: `scores` exposes only `scores.json`,
`summary` adds an LLM summary per candidate, `full` exposes everything.

## 5. Evaluation fixes (each pinned by a test)

1. Context cost counted once, in tokens, by `MeteredModel` only; harnesses stop adding `len(prompt)`.
2. Repeats are real resamples: the cache is keyed by `(prompt, sample_index)`; `--repeats N` uses
   distinct indices.
3. Online classification test protocol: memory is learned from the train stream, then frozen
   copies are evaluated on test (paper §4.1).
4. Proposer prompt placeholders are filled by the engine, not left for a shell.
5. Validation probe per domain: the probe is the first search task, not a fake classification row.
6. Terminal metric fix is moot (terminal harness removed with the Claude Code layer); the
   terminal domain is out of scope (§8).

## 6. Domain: online text classification

`meta_harness/datasets.py` gains a loader for one public dataset from the paper usable without
login (Symptom2Disease via the Hugging Face datasets-server HTTP API, stdlib `urllib`), cached
under `data/` in the repo. Seeds: zero-shot and few-shot baselines. The paper's protocol:
labelled train stream feeds `learn`, held-out test evaluated once on the frontier.

## 7. Comparisons and ablations

- `meta-harness ablate --views scores,summary,full --runs N` — runs the search per view and
  reports median/best (Table 3 shape).
- `meta-harness search --optimizer best-of-n` — the same proposal budget with no experience
  filesystem access (proposer sees only the seed), as the equal-budget baseline.
- Evaluation budget (harness evaluations, model calls, tokens) is recorded in `run.json`.

## 8. Full paper coverage (owner, 2026-10-11: "the full gaps and system")

Every gap in `.superpowers/pivot/paper-gap.md` is in scope:

- **All classification datasets** from the paper (LawBench, Symptom2Disease, USPTO-50k) and the
  9 out-of-distribution sets, each via a stdlib loader with a local test fixture; all eight
  discovered classification harnesses and the dense / random few-shot baselines; ACE and MCE
  per their reference behaviour.
- **Retrieval-augmented math:** corpus build (stdlib BM25 index), decontamination against the
  test sets, IMO-style test set loader, and the full router (separate hard-geometry index,
  adaptive K, a real technique-statement check).
- **Agentic coding (TerminalBench-2):** a searchable Python harness with Terminus-2 and
  Terminus-KIRA style seeds, run in Docker when available (skipped with a stated reason when
  not), with the container recorded on the task so scoring runs inside it.
- **Comparison optimizers** under an equal evaluation budget: Best-of-N, an OpenEvolve-style
  evolutionary loop, a TTT-Discover-style loop, and a GEPA-style reflective prompt optimizer.
- **Multi-model transfer** (Table 6): evaluate a fixed harness across N models.
- **Proposer read statistics** (Table 8) and an `experience` query CLI (top-k, diffs, reuse
  across runs).
- **Sandboxing:** subprocess isolation with resource limits everywhere; Docker when available.

Expensive full-scale runs are documented commands; the README reports only numbers actually
produced, with their scale stated.

## 9. Constraints

Stdlib only (`dependencies = []`); explicit utf-8 I/O and `encoding="utf-8", errors="replace"`
on every subprocess; Windows-safe paths; all work inside the repository folder; no AI
attribution in commits, authors or docs; tests never call a real API (scripted provider);
pushes go to `main`.

## 10. Risks

- No API key in the build environment: the real-dataset run is a documented command the owner
  runs; the README reports only numbers actually produced.
- Provider tool-calling formats drift: both are tested against recorded response fixtures.
- Proposer cost: bounded tool calls and iterations; `run.json` reports tokens spent.

## 11. Designs borrowed from reference harnesses

Read from `refs/` (git-ignored clones of soketlabs/loop, rivet-dev/rivet, truefoundry/trueforge;
Apache-2.0 / Apache-2.0 / MIT). Designs only, re-written in stdlib Python; notes with source paths
in `.superpowers/pivot/refs-*.md`.

- **Proposer loop** (loop, trueforge): `write_candidate` terminates the turn; a before-tool hook
  enforces path confinement and test-split denial; iteration cap; orphaned tool calls closed with
  an error result; unknown tools and tool errors returned as tool results; `length` stop is an
  explicit failure.
- **Tool output bounds** (loop, trueforge): results capped (~6000 tokens / 2000 lines), head for
  files, tail for commands, UTF-8 safe, with a preview and a pointer to narrow via offset/limit or
  grep; full output spilled to a file inside the run directory.
- **Provider layer** (loop, trueforge): one internal message type converted per provider; usage
  record `{input, output, cache_read, cache_write, reasoning}` with an explicit flag for whether
  input includes cache; billed cost reported separately from the context-token Pareto axis; a
  scripted provider that replays text, tool-call and error responses.
- **Resumable, parallel runs** (rivet, trueforge): every step's result is a file, `--resume`
  skips finished steps, `run.json` checkpointed each iteration; per-task failures recorded as
  scores, per-evaluation timeouts; retries only on 429/5xx, never on harness errors or timeouts.
- **Sandbox** (rivet, loop): `Sandbox` protocol (`exec`, `put`, `get`) with a temp-dir subprocess
  implementation and a Docker CLI implementation; a preflight check with install hints.
- **Benchmark runner** (trueforge): resumable `{arm × task × trial}` matrix; failed trials kept
  and scored 0; optional blind LLM-judge metric ending in `VERDICT: PASS|FAIL`.
- **Terminus-style seeds** (loop, trueforge): exact-then-fuzzy edit, compaction at a reserve
  threshold with a "don't redo work" continuation.
