---
name: apply
description: Lift a discovered harness out of a finished run and into the user's own codebase, or turn it into a reusable skill. Use when the user asks to ship, adopt, port or productionise a winning harness, or wants the search result wired into their application.
---

# Ship a discovered harness

A search run produces readable Python, not weights. That is the point: the strategy transfers.
The paper's §4.2 result is a single discovered retrieval harness improving five models it was
never searched against, so a harness found with a cheap executor is usually worth keeping when
the user later upgrades models.

`$ARGUMENTS` may name a candidate id or a run root. Default to the top of
`.meta-harness/frontier.json`.

## 1. Pick the harness honestly

```bash
cd "${CLAUDE_PLUGIN_ROOT}"
cat .meta-harness/frontier.json
cat .meta-harness-test/test_results.json
```

Select on the **held-out** score, not the search score. If the frontier has several entries, the
choice is a trade the user makes: highest accuracy, or near-equal accuracy at a fraction of the
context cost. Present both with their numbers and let them choose.

## 2. Read it before you move it

```bash
cat .meta-harness/candidates/<id>/harness.py
cat .meta-harness/candidates/<id>/proposer_reasoning.md
```

Audit it first — this is LLM-written code you are about to put in production:

- hard-coded labels, answers, or strings lifted from the dataset → reject it, the search overfit
- unbounded memory growth across a long task stream
- silent `except:` swallowing real failures
- prompt assembly that scales with memory size and will blow up on a longer run

## 3. Port the strategy, not the file

The transferable part is the *policy*: what gets retrieved, how many examples, how the prompt is
ordered, when memory is written, whether there is a verification pass. Rewrite that into the
user's own code and conventions rather than dropping in a foreign file. Ask where it should live
before writing anything.

If the harness uses `meta_harness.retrieval` (`TfidfIndex`, `BM25Index`,
`reciprocal_rank_fusion`), either add the dependency or port those few functions — they are
standard-library only.

Remember the interface it was written against: `model(prompt) -> str`, memory on `self` across
the task stream, a new instance per evaluation. If the user's call site differs — streaming,
tools, a chat history — adapt the policy to it explicitly and say what you changed.

## 4. Alternative: keep it as a skill

If the user wants it reusable across projects rather than inlined, write the strategy as a skill
in their own `.claude/skills/<name>/SKILL.md`: what to retrieve, how to order the prompt, what
to remember. Cite the run it came from and its held-out score so the claim stays checkable.

## 5. Leave the evidence attached

Whatever form it takes, record in a comment or the skill body: the run root, candidate id,
held-out score, and dataset it was searched on. A harness with no provenance is a magic prompt
nobody will dare change later.
