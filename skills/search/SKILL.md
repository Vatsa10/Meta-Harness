---
name: search
description: Run a Meta-Harness search over harness code for a dataset, using Claude Haiku 4.5 as the harness model. Use when the user wants to optimize a prompt/retrieval/memory strategy against a labelled dataset, improve classification or math accuracy, cut context tokens, or asks to "search for a better harness".
---

# Run a harness search

Search over harness code for the dataset in `$ARGUMENTS` (a path to a `.jsonl`, `.json`, or
`.csv` file; if empty, ask which dataset, or offer `data/ticket_intents.jsonl` as a demo).

The plugin root is `${CLAUDE_PLUGIN_ROOT}`. Run everything from there.

## 1. Preflight

```bash
cd "${CLAUDE_PLUGIN_ROOT}"
python -c "import meta_harness; print('ok')" || pip install -e .
python -m meta_harness models --provider unikey | head -20
```

`UNIKEY_API_KEY` must be set (or present in a `.env` beside the plugin). If `models` fails with
an auth error, stop and tell the user to set it — do not guess a key.

## 2. Check the dataset shape

Read the first two lines of the dataset and pick the task type:

| Fields present | `--task-type` |
|---|---|
| `input` + `label` | `classification` |
| `problem` (+ `answer`) | `math` |
| `instruction` (+ `test_command`, `image`) | `terminal` — use the `optimize-agent` skill instead |

If the rows lack a `labels` field for classification, say so: the harness then has to infer the
label set from its own memory, which is harder and leaves the search more room.

## 3. Run the search

Default to Haiku 4.5 as the harness model. The paper's agentic-coding result (§4.3) is on
Haiku 4.5, where searched harnesses beat every hand-engineered agent on TerminalBench-2, so a
cheap executor plus a strong proposer is the intended shape — not a compromise.

```bash
cd "${CLAUDE_PLUGIN_ROOT}"
python -m meta_harness run \
  --tasks <dataset> \
  --task-type <type> \
  --provider unikey --model claude-haiku-4-5-20251001 \
  --proposer-model claude-opus-4-8 \
  --iterations 10 --candidates 1 --max-workers 4 \
  --root .meta-harness
```

Notes before you launch it:

- A run costs real money and takes minutes. Tell the user the model pair, iteration count and
  dataset size, then start it — do not ask for confirmation twice.
- If the `claude` CLI is not on PATH, or the user wants no nested agent, append
  `--proposer-command "python tools/llm_proposer.py"` and set
  `META_HARNESS_PROPOSER_MODEL=claude-opus-4-8`. That proposer needs only the API key.
- Add `--repeats 3` when scores look noisy (the paper averages three samples per problem).
- Add `--search-fraction 0.7` explicitly for datasets under ~30 rows so the held-out split
  stays meaningful.
- Long runs: launch in the background and report the frontier when it finishes.

## 4. Report

Print a table of every candidate — score, context cost in tokens, valid/invalid — then the
held-out test scores from `.meta-harness-test/test_results.json`. Name the winner and quote one
or two lines from its `proposer_reasoning.md` so the user sees *why* it won.

Two honest outcomes to call out rather than paper over:

- **Baselines already at ceiling.** If seeds score 1.0, accuracy cannot improve and the search
  optimizes context cost instead. That is the Pareto frontier working; say so plainly and
  suggest a harder dataset if accuracy was the goal.
- **Test below search.** A gap between the search split and held-out split is the honest
  number. Report it; do not report the search score as the result.

## 5. Offer next steps

- `/meta-harness:inspect` to dig into traces for a specific candidate.
- `/meta-harness:apply` to lift the winning harness into the user's own code.
