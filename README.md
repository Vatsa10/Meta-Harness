# Meta-Harness

A standalone implementation, with no runtime dependencies (stdlib only), of *Meta-Harness: End-to-End Optimization of Model
Harnesses* (arXiv 2603.28052).

## What it is

A **harness** is the code around a model: how the prompt is built, what is retrieved, what is
remembered between tasks, how the reply is parsed. Meta-Harness **searches over harness code**,
not over a prompt string. Each iteration, a tool-using **proposer** model browses an experience
directory holding the full source, scores and execution traces of every prior candidate, chooses
what to read, and writes a new harness; the engine evaluates it and adds it to the directory. The
paper's key result is that this full-history access is what makes the search work: removing the
raw traces (scores only, or scores plus LLM summaries) loses most of the gain (Table 3 below). The
output is a Pareto frontier over (score, context tokens), each frontier member scored once on a
held-out split the proposer never sees, plus an auditable record of every candidate.

## How it differs

Numbers from paper Table 1:

| | Prompt optimizers (OPRO, TextGrad, GEPA) | Score-guided evolvers (AlphaEvolve, TTT-Discover) | Meta-Harness |
|---|---|---|---|
| What is optimized | prompt text / instructions | programs | the whole harness program (retrieval, memory, parsing, prompts) |
| What the optimizer sees | last artifact + textual feedback or summaries | a window of programs or solution fragments + scores | **all** prior code, scores and raw traces, on a filesystem it navigates with tools |
| History per iteration | 0.002-0.015 MTok | 0.022-0.026 MTok | up to 10 MTok |

Paper Table 3 (online text classification, accuracy %; > ZS = runs above zero-shot):

| Proposer sees | Scores | Code | Summary | Traces | Median | Best | > ZS |
|---|---|---|---|---|---|---|---|
| Scores only | yes | yes | no | no | 34.6 | 41.3 | 26 |
| Scores + summary | yes | yes | yes | no | 34.9 | 38.7 | 23 |
| Meta-Harness (full) | yes | yes | - | yes | 50.0 | 56.7 | 39 |

`--view scores` still shows candidate source, as in the paper's ablation. These are the paper's
numbers, not ours. Ours are under [Results](#results).

Related work: DSPy (prompt-program optimization) and OpenEvolve (open AlphaEvolve-style
evolution) are close relatives; `--optimizer openevolve` is a style re-implementation.

Differences from the paper's own setup:

- The proposer is **any tool-calling model** behind a small provider layer (Anthropic Messages,
  OpenAI-compatible chat completions), not a coding-agent product. Its tools are `list`, `read`,
  `grep` and `write_candidate`, confined to the run directory.
- Context cost is **provider-reported tokens** metered by the engine (harnesses never count it).
- Stdlib only: `dependencies = []`. Requires Python 3.11+ (`tomllib`).

## Quick start

```bash
pip install -e .
```

Fully offline (a scripted provider replays canned replies; no key, no network):

```bash
meta-harness search --harness examples/sentiment/harness.py --data examples/sentiment/data.jsonl \
    --metric examples/sentiment/metric.py:score \
    --provider scripted --model examples/sentiment/base_script.jsonl \
    --proposer-provider scripted --proposer-model examples/sentiment/proposer_script.jsonl \
    --iterations 2 --root .meta-harness-example --no-cache
```

The seed scores 0 (the model is chatty), the scripted proposer reads the traces and writes a
harness that maps replies onto labels, which scores 1 on search and test. This checks the
pipeline, not model quality. See [examples/README.md](examples/README.md).

With a real model, put keys in the environment or a git-ignored `.env` at the repo root
(`--env-file` picks another file; the environment always wins):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...        # OPENAI_BASE_URL for --provider compatible
UNIKEY_API_KEY=...        # optional UNIKEY_BASE_URL
GROQ_API_KEY=...          # optional GROQ_BASE_URL
```

```bash
meta-harness search --harness my_harness.py --data evals.jsonl --metric exact \
    --provider anthropic --model claude-haiku-4-5-20251001 --iterations 10
meta-harness search --harness my_harness.py --data evals.jsonl --metric exact \
    --provider unikey --model gpt-5-mini --iterations 10
```

| Provider | `--provider` | Key / base URL |
|---|---|---|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL` |
| OpenAI | `openai` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| Any OpenAI-compatible endpoint | `compatible` | `OPENAI_API_KEY`, `OPENAI_BASE_URL` or `--base-url` (required) |
| Unikey gateway | `unikey` | `UNIKEY_API_KEY`, `UNIKEY_BASE_URL` |
| Groq | `groq` | `GROQ_API_KEY`, `GROQ_BASE_URL` |
| Offline replay | `scripted` | `--model` is the script path |

Keys are read from the environment only: never printed, logged, written to `run.json` or passed
on a command line.

## Contracts

**Harness** - a Python file defining `class Harness` (or a `build_harness()` factory):

```python
class Harness:
    def run(self, task, model, trace):     # `trace` is optional: run(self, task, model) also works
        reply = model("Classify: " + task["input"])   # model(prompt) -> str; tokens are metered
        return {"output": reply.strip()}              # a dict with "output", or a bare value

    def learn(self, task, result):          # optional; online domains stream train tasks into it
        ...
```

**Data** - JSONL (or JSON / CSV), one task per line, any fields. The target is the first of
`expected`, `label`, `answer`, `target`; it is hidden from the harness on held-out evaluation.
Split by `--test-fraction` (default 0.3, seeded by `--seed`) or explicit `--search` / `--test`
(online domains also `--train`).

**Metric** - `exact`, `contains`, `numeric`, a domain metric, or `path/to/file.py:function` /
`module:function` with signature `(task, result) -> float`:

```python
def score(task, result):
    return float(str(result.get("output", "")).strip().lower() == task["label"].lower())
```

Key `search` flags:

| Flag | Default | Meaning |
|---|---|---|
| `--harness` | required | seed harness file, repeatable |
| `--data` / `--search` `--test` / `--train` | - | eval set and splits |
| `--metric` | domain's, else `exact` | see above |
| `--provider` `--model` | required | base model the harness calls |
| `--proposer-provider` `--proposer-model` | same as base | proposer (paper: strong proposer, cheaper base) |
| `--iterations` / `--candidates` | 10 / 1 | proposer iterations / proposals per iteration |
| `--optimizer` | `metaharness` | `metaharness`, `best-of-n`, `openevolve`, `ttt_discover`, `gepa` |
| `--max-evaluations` | none | equal-budget cap on search-split harness evaluations |
| `--protocol` | domain's | `plain` or `online` (train stream into `learn`, frozen scoring) |
| `--view` | `full` | `scores`, `summary` or `full` (Table 3 ablation) |
| `--max-tool-calls` | 40 | proposer tool calls per proposal |
| `--domain` | `generic` | `classification`, `math`, `terminal`: prompt section, protocol, metric |
| `--repeats` | 1 | resamples per task (distinct cache sample index) |
| `--isolation` | `process` | `process`, `docker` or `none` (see Safety) |
| `--root` | `.meta-harness` | experience directory; held-out results go to `<root>-test` |
| `--resume` | off | continue a run; finished steps are skipped |
| `--price` | - | `PROVIDER:MODEL=IN,OUT[,CR,CW]` USD per million tokens for billed cost |
| `--proposer-command` | - | escape hatch: an external program writes `.py` files to `META_HARNESS_OUTPUT` |

## The experience directory

```
<root>/
  run.json                      config + budget: evaluations, model calls, tokens, proposals, cost
  frontier.json                 Pareto frontier over (score, context tokens)
  candidates/<id>/
    harness.py                  candidate source
    scores.json                 score, context cost, stop_reason histogram, per-task metrics
    traces.jsonl                every model call, prompt and response, per task
    proposer_log.jsonl          every proposer tool call (what it read)
    proposer_reasoning.md       the proposer's rationale
    receipt.json                verdict (accepted / rejected_* / failed_eval / ...), parents read
<root>-test/test_results.json   held-out scores of the frontier, computed once
```

The proposer's tools refuse every path outside `<root>`, and `<root>-test` is on the forbidden
list; `--test-root` inside `--root` is rejected.

## Commands

| Command | What it does | Key flags |
|---|---|---|
| `search` | harness search (above) | see table |
| `ablate` | N searches per view, median / best, McNemar + BH significance (Table 3 shape) | `--views scores,summary,full --runs N --max-tasks 20 -- <search args>` |
| `transfer` | fixed harnesses x several models (Table 6 shape) | `--harness --data --metric --models p:m,p:m [--protocol online --train-data]` |
| `ood` | harnesses on the nine out-of-distribution sets (Table 5) | `--harness --datasets all --provider --model --limit 20` |
| `bench` | resumable arm x task x trial matrix | `--arm name=harness.py@provider:model --data --trials --out` |
| `fetch` | download a paper dataset into `data/cache/<name>` | `<loader> [--full]` |
| `math-corpus` | build the deduplicated, decontaminated math retrieval corpus | `build [--limit-per-source 200 \| --full]` |
| `experience` | query runs: `top`, `diff`, `diff-runs`, `import`, `reads` (Table 8), `report`, `export` | `top <root> -k 5` |
| `sweep` | evaluate one harness across values of an env var, no proposer | `--harness --env KEY=v1,v2 --tasks --provider --model` |
| `models` | list models of an OpenAI-compatible provider | `--provider unikey\|groq\|openai\|compatible` |
| `inspect` | print stored results and the frontier | `<root>` |
| `demo` | deterministic offline end-to-end run | `--iterations 3` |

Offline checks of the example run:

```bash
meta-harness inspect .meta-harness-example
meta-harness experience reads .meta-harness-example
meta-harness ablate --views scores,full --runs 1 --root .meta-harness-ablate -- \
    --harness examples/sentiment/harness.py --data examples/sentiment/data.jsonl \
    --metric examples/sentiment/metric.py:score --provider scripted \
    --model examples/sentiment/base_script.jsonl --proposer-provider scripted \
    --proposer-model examples/sentiment/proposer_script.jsonl --iterations 1 --no-cache
```

Compare optimizers under an equal budget: run `search` once per `--optimizer` with the same
data, seeds, models and `--max-evaluations`, then compare the runs:

```bash
meta-harness search --optimizer best-of-n --max-evaluations 3 --root .mh-bon --iterations 2 \
    --harness examples/sentiment/harness.py --data examples/sentiment/data.jsonl \
    --metric examples/sentiment/metric.py:score --provider scripted \
    --model examples/sentiment/base_script.jsonl --proposer-provider scripted \
    --proposer-model examples/sentiment/proposer_script.jsonl --no-cache
meta-harness experience diff-runs .meta-harness-example .mh-bon
```

`best-of-n` gives the proposer only the seed (no experience filesystem); `openevolve`,
`ttt_discover` and `gepa` are style re-implementations of those methods, not their original code.

## Domains

| Domain | Datasets (loader) | Protocol | Deviations from the paper |
|---|---|---|---|
| `classification` | `symptom2disease` (HF `gretelai/symptom_to_diagnosis`), `lawbench` (open-compass LawBench task 3-3), `uspto50k` (HF `pingzhili/uspto-50k`); OOD: scicite, finer139, amazon5, fpb, goemotions, banking77, agnews, scitail, tweet_hate | online: train stream feeds `learn`, frozen copies scored on search/test | **USPTO-50k is 10-way** (the public copy has no 180-way label); **LawBench has 275 labels** (closest public copy, not the paper's 215) |
| `math` | `math_corpus` (8 sources, Table 10), `math_search` (OlympiadBench + Omni-MATH hard), `imo` (IMO-AnswerBench, IMO-ProofBench, ArXivMath) | plain, BM25 retrieval over the decontaminated corpus | **Omni-MATH** keeps all rows (about 4,428), not the paper's 933; **ProofBench**: only the 27 rows with a short answer are graded (proof grading needs a judge), so the full test set has fewer than 200 gradable items |
| `terminal` | `terminalbench` from a local TerminalBench-2 checkout (`META_HARNESS_TB_DIR`) | search = test (as in the paper); each task in its own Docker container, tests copied in only at scoring time | the container is restarted before scoring and scorer binaries are checked with `docker diff` (fixed list of scorer binaries); modifications outside the checked set remain a **residual reward-forging risk**; TB containers keep resource limits and `no-new-privileges` but not `--cap-drop ALL` |

Seeds per domain are in [baselines/](baselines/README.md).

## Safety

- **Held-out split**: scored once per frontier member after search; the proposer cannot read it,
  and resume never re-scores it.
- **Proposer confinement**: tools are read-only except `write_candidate`, bounded in output size,
  and refuse paths outside the run directory and the test directory.
- **Redaction**: anything matching `api_key=...`, `bearer ...`, `sk-...` or a loaded `*_API_KEY`
  value becomes `[redacted]` in logs, traces and receipts.
- **Secrets on disk**: seeds and candidates run from their copy inside the run directory, and
  the process-isolation open guard always refuses `.env*` files and `~/.ssh`.
- **Isolation** (`--isolation`) of candidate code, which the proposer wrote and you did not:

| Mode | What you get | Honest limit |
|---|---|---|
| `process` (default) | child process, allowlisted env (no keys), temp cwd, memory/CPU limits (rlimits on POSIX, Job Object on Windows), model calls served by the parent | socket and file-open guards are **best-effort and bypassable** (ctypes, `os.open`, subprocess); the child runs as your OS user |
| `docker` | `--network none --cap-drop ALL --read-only --user 65534`, memory/CPU limits, package and candidate mounted read-only | **the real security boundary**; needs Docker |
| `none` | in-process | **no isolation at all**. Never run untrusted or proposer-written candidates with `none` |

## Reproducing the paper

Small scale (the defaults: capped samples, at most about 20 tasks per split):

```bash
meta-harness fetch symptom2disease
meta-harness search --domain classification \
    --harness baselines/classification/zero_shot.py --harness baselines/classification/few_shot.py \
    --search data/cache/symptom2disease/val.jsonl --test data/cache/symptom2disease/test.jsonl \
    --provider unikey --model gpt-5-mini --iterations 3 --candidates 2
meta-harness ood --harness baselines/classification/zero_shot.py --datasets all \
    --provider unikey --model gpt-5-mini --limit 20
meta-harness math-corpus build
```

The classification domain streams `train.jsonl` (written by `fetch` next to `val.jsonl`) into
`learn`, then scores frozen copies.

Full scale (documented commands, not run here; cost scales with tokens):

```bash
meta-harness fetch symptom2disease --full
meta-harness math-corpus build --full
meta-harness ood --harness <frozen harness> --datasets all --provider <p> --model <m> --limit 1000
meta-harness ablate --views scores,summary,full --runs 3 --max-tasks 1000 -- <search args>
```

## Results

First live check, 2026-10-11 (full write-up: [docs/results/2026-10-11-live.md](docs/results/2026-10-11-live.md)).
It is a smoke run, not a benchmark.

- Providers and models: openai, `gpt-4o-mini` as the base model and `gpt-4.1-mini` as the proposer.
- Scale: Symptom2Disease, seed-0 subsample of 40 train, 20 search and 20 test examples; 3
  iterations of 2 candidates, `--max-evaluations 8`.
- Headline: best search accuracy 0.75 to 0.80; held-out test accuracy 0.55, unchanged. Billed
  cost about $0.336.

Caveats:

- One example is worth 5 accuracy points, so the search gain is a single example and the test
  result did not move.
- These numbers can't be compared with the paper's Table 2.
- The Best-of-N comparison is inconclusive: it produced 0 candidates under the 20-tool-call cap.

## Roadmap and limits

Not implemented:

- Proof grading for IMO-ProofBench (needs a judge).
- The paper's 180-way USPTO and 215-way LawBench label sets (no public copy found).
- A process-isolation boundary that a hostile candidate cannot bypass (use `docker`).
- Native Gemini provider (use `compatible`).

## Credits

- The method: Meta-Harness, arXiv 2603.28052.
- Designs re-implemented in stdlib Python (designs only, no code copied): [soketlabs/loop](https://github.com/soketlabs/loop) and
  [truefoundry/trueforge](https://github.com/truefoundry/trueforge) (proposer loop, tool output bounds, provider layer, benchmark runner,
  Terminus-style seeds), [rivet-dev/rivet](https://github.com/rivet-dev/rivet) (resumable runs, sandbox protocol), keel (MIT; candidate
  receipts, crash repair, retries, redaction, read window), and *An Empirical Study of Harness
  Design for Coding Agents* (arXiv:2609.20804: paired McNemar + Benjamini-Hochberg, stuck
  detection, staged compaction, stop-reason histograms, phase labels).
