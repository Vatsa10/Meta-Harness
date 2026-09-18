---
name: reading-execution-traces
description: Use when a prompt or agent variant scored worse than expected and the reason is unclear, when two variants differ in score but not obviously in behavior, or when a failure needs a cause rather than a count
---

# Reading execution traces

## Overview

A score says a case failed. A trace says what the model was shown and what it said back. Only the
second one tells you what to change.

**Core principle:** Diagnose from the exact prompt and the exact reply, on one failing case, before
proposing anything.

## When to Use

- A variant regressed and you do not know which change did it
- Two candidates score differently and you cannot explain why
- "The model keeps confusing X and Y" — before theorizing about why

**Skip when:** the failure is a stack trace or a non-200. That is a bug, not a harness problem.

## What a trace must contain

Per case: the **full prompt string sent**, the **raw reply**, the parsed prediction, the expected
value, and the reward. Anything less and you are guessing.

```json
{"event": "model_call", "prompt": "Labels: billing, bug...\nInput: card charged twice\nLabel:", "response": "duplicate_charge", "tokens": 41}
{"event": "task_end", "index": 3, "prediction": "duplicate_charge", "reward": 0.0}
```

Reward 0 with a response that reads correct is the single most common finding, and it is never a
model problem.

## The method

**1. Pick one failing case.** Not the aggregate. One.

**2. Read the prompt that was actually sent.** Not the template — the rendered string. Check in
this order, because these are the frequent causes:

| Check | Failure it catches |
|---|---|
| Is the expected label/answer present in the prompt at all? | Label set built from memory that was empty on the first cases |
| What did retrieval actually surface? | Nearest examples all from the wrong class, pushing the model away |
| Is the instruction still there? | Long context pushed it past where the model attends |
| Did a truncation cut mid-example? | Silent slicing in prompt assembly |

**3. Read the raw reply.** Correct content in a shape the parser dropped — a rationale before the
label, a quoted string, a trailing period — is a parsing bug wearing an accuracy costume.

**4. Diff against a candidate that passed the same case.** Same case, two traces, side by side. The
difference between them is the mechanism. This is the highest-value single move in harness
debugging and the one most often skipped in favour of re-reading the code.

**5. Only now propose a change** — and one change, aimed at the mechanism you just identified.

## Comparing across candidates

```bash
# every failing case for a candidate
grep -h '"event": "task_end"' candidates/<id>/traces.jsonl | grep '"reward": 0'

# the first prompt it actually sent
grep -h '"event": "model_call"' candidates/<id>/traces.jsonl | head -1
```

When a candidate regressed after several changes, the traces tell you which one moved behaviour:
look for the first call where the two candidates' prompts diverge in something other than the
edit you intended.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Reading aggregate accuracy and theorizing | Open one failing trace |
| Reading the prompt template instead of the sent string | Templates hide what memory and retrieval put in |
| Concluding "the model is confused" | The model saw something specific. Find out what |
| Skipping the winner/loser diff on the same case | It is the fastest path to the mechanism |
| Treating a format mismatch as an accuracy problem | Fix the parser, do not rewrite the prompt |

## Real-World Impact

Systems given scores plus code but no raw traces reached 34.6 median accuracy on a harness search;
adding LLM-written summaries of those traces reached 34.9. Raw trace access reached 50.0 — and its
*median* candidate beat the *best* candidate of either summarized condition. Summaries do not
recover the signal, because the diagnostic detail is exactly what summarizing removes.
