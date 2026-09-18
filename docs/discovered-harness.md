# Discovered harness

The output of a search run, in the form you can actually use. Paste the doctrine into your
`CLAUDE.md`, a skill body, or a subagent's system prompt.

## Provenance

| | |
|---|---|
| Run | `.meta-harness-fm` |
| Candidate | `iteration-0002-00` |
| Search split | 6 tasks, score **1.000**, context **96,601** tokens/task |
| Held-out split | 2 unseen tasks, score **1.000**, context **97,028** tokens/task |
| Baseline | stock Claude Code: 1.000 at **130,692** — the winner is **26% cheaper** |
| Model | Claude Haiku 4.5 via the local CLI |
| Tasks | `data/cc_failure_modes.jsonl` |

Accuracy was saturated at 1.000 across every candidate, so this is a **context-cost** result, not
an accuracy one. Said plainly: it does the same work for a quarter less context. Accuracy headroom
needs Terminal-Bench (see `tools/tb_run.py`), where Haiku fails tasks outright.

## The doctrine

```
You are completing one task alone. No one will answer questions.

The Edit tool refuses any file you have not Read in this session. Read first, always - a
rejected Edit costs a turn and teaches you nothing.

When the task touches several files, Read all of them in a single message with one Read call
per file, then make all the edits. Do not walk the files one turn at a time.

Every clause of the requirement is tested, including clauses stated mid-sentence and
constraints on what must NOT change. When a requirement spans several files, change all of
them; a file left stale is a failure even when the file you changed is correct.

Before you stop, check each clause against the edit that satisfies it. Never leave a stub, a
TODO, or a placeholder, and never weaken a requirement to make it easier to satisfy.
```

## Why each line is there

Nothing here was written from intuition. Every clause traces to a measurement:

| Clause | Evidence |
|---|---|
| Read before Edit | A literal error string in `iteration-0001-00`'s trace: the Edit tool rejecting files never read, each rejection costing a turn |
| Batch reads into one message | Cost is `turns x fixed preamble`; the session preamble is re-sent every turn, ~30k tokens regardless of transcript size |
| Every clause is tested, mid-sentence included | 49 `correction` episodes in mined history — humans saying "that's wrong / revert" after tool use |
| Change every file a requirement spans | Cross-file drift, the failure mode the mined read:write ratio of 0.40 predicts |
| Don't repeat a failing call | 130 `thrash` episodes: one tool hammered inside a six-turn window |

## The order the numbers came in

| Harness | Context/task | vs stock |
|---|---|---|
| `cc_default` — stock Claude Code | 130,692 | — |
| `cc_guided` — doctrine written from intuition | 115,723 | −11% |
| `cc_evidence` — doctrine written from mined numbers | 96,772 | −26% |
| `iteration-0001-00` — proposer, first attempt | 111,068 | −15% (a regression) |
| `iteration-0002-00` — proposer, after reading the regression | **96,601** | **−26%** |

Two things worth keeping from that table. Doctrine derived from measurements beat doctrine
written from intuition by 16% before the search ran at all. And the proposer's first attempt made
things *worse* — it recovered only by reading its own predecessor's trace, which is the whole
argument for keeping every candidate instead of just the winner.

## Applying it

1. Paste the doctrine into `CLAUDE.md`, or into a skill body if you want it loaded conditionally.
2. Record where it came from — run root, candidate id, held-out score — so the claim stays
   checkable when someone wants to change it later.
3. After a couple of weeks of real work:

```bash
python -m meta_harness compare --split <the date you applied it>
```

That diffs your history either side of the change. It is observational, not an experiment: the
two windows cover different work, so treat a moved number as a lead rather than a result.
