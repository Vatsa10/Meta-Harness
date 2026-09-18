---
name: optimizing-harnesses
description: Use when changing a prompt, agent scaffold, retrieval strategy, memory rule, or skill that something depends on - especially when failures were reported, when under deadline to "just fix the prompt", or when about to rewrite an artifact that already has users
---

# Optimizing harnesses

## Overview

A **harness** is the code around a fixed model: what it stores, retrieves, and shows the model at
each step. Prompts, few-shot selection, retrieval policy, memory rules, agent scaffolds, skills —
all harness.

**Core principle:** A harness edit you cannot attribute is not an improvement, it is a guess that
happened to score well.

**Violating the letter of the rules is violating the spirit of the rules.**

## When to Use

- Reported failures on an LLM feature, and you are about to edit the prompt
- Choosing between two retrieval or memory strategies
- An agent scaffold that works sometimes and you want it to work more
- Deadline pressure to "just fix it today"

**Skip when:** the artifact is a throwaway, has no users, or a human reads every output anyway.

## The Three Laws

```
ONE CHANGE PER CANDIDATE
EVERY CANDIDATE KEPT
NO SCORE WITHOUT A TRACE
```

### 1. One change per candidate

The strong instinct under deadline is to bundle: add label definitions AND few-shot examples AND
a tiebreak rule AND a rationale field, then measure once. If the bundle wins you do not know which
part won; if it loses you do not know which part lost, and you will throw away three good changes
with one bad one.

Each candidate is one hypothesis. Four ideas is four candidates. They are cheap — the eval run is
the expensive part, and it costs the same either way.

**The exception that is not one:** "these changes belong together" almost always means "I have not
tested them apart". Bundle only when one change is literally inert without the other (a field the
prompt adds and the parser reads), and say so in the candidate's notes.

### 2. Every candidate kept

Do not overwrite the prompt. Each attempt gets its own file, its score, and a note saying what it
was testing and what happened.

```
candidates/
  00-baseline.py          score 0.72   ctx 780 tok
  01-label-definitions.py score 0.84   ctx 1240 tok   "defined the 4 confusable labels"
  02-few-shot-8.py        score 0.80   ctx 2100 tok   "8 nearest examples by tf-idf"
  03-defs-plus-tiebreak.py score 0.88  ctx 1310 tok   "01 + priority rule for bundled tickets"
```

The archive is the substrate for the next improvement, not paperwork. Attempt 7 is written by
reading why 3 beat 2. Discard the losers and every future attempt starts from zero — which is why
hand-tuning plateaus after a handful of tries.

### 3. No score without a trace

For every case, store the exact prompt sent and the exact reply received. Per-label accuracy tells
you *that* the model confuses two labels. Only the trace tells you *why* — the label was missing
from the prompt, retrieval surfaced three examples of the wrong class, the reply was correct but
in a format the parser dropped.

This is the measured difference between diagnosing and guessing, not a style preference. See
`reading-execution-traces`.

## Two objectives, always

Score is not the only axis. Every candidate has a context cost — input tokens per case — and
prompt bloat is the default failure mode of hand-tuning: definitions, then examples, then rules,
then a rationale field, each defensible, the sum enormous and slow.

Track both. Keep the Pareto frontier: a candidate is worth keeping if nothing beats it on score
*and* cost together. A variant matching the incumbent's accuracy at half the tokens is a real win
and belongs on the frontier.

## Never tune against the held-out split

Split the eval set before you start. Tune on the search split. Look at the held-out split once,
at the end, to report a number.

The moment you see a held-out score and edit in response, it stops being held out and your reported
number is the number you tuned against. If you burn it, say so and cut a fresh split.

Report the held-out number as the result. The tuning-split number is not the result.

## Stop rule

Stop when the frontier stops moving, not when a candidate "looks better". Before declaring done:

- Held-out score measured once, reported as the result
- The originally reported failures re-checked case by case
- No class regressed relative to baseline (overall accuracy hides one label collapsing)
- Winner read for hard-coded case handling (see Red Flags)

## Rationalization Table

| Excuse | Reality |
|---|---|
| "Ship Friday, I'll bundle the changes" | Bundling does not save an eval run — it costs you attribution when it regresses |
| "These four changes go together" | Then you have not tested them apart. Split them |
| "Tokens are cheap, accuracy is what matters" | Bloat is the default failure of hand-tuning, and it compounds silently across every call |
| "Per-label accuracy shows me the problem" | That is the scores-only mode. It shows you *where*, never *why* |
| "I'll keep the winner and delete the rest" | The losers are what the next candidate is written from |
| "Just a quick peek at the test split" | That is the entire failure mode. One peek and the number is spent |
| "No time for an eval set" | Then you have no time to know whether you fixed it or shuffled which cases fail |
| "It obviously helps, the six tickets pass now" | Six tickets is the set you overfit to. Check the ones that used to pass |

## Red Flags - STOP

- A single edit that changes four things at once
- The previous prompt exists only in git history
- A score with no stored prompt or reply behind it
- The reported number came from the split you tuned on
- The winning candidate contains `if "refund" in text` or a label list copied from your eval data
- "Seems better" appearing anywhere in the decision

Code-space overfitting is visible in a way weight-space overfitting is not — read the winner
before you ship it. Brittle if-chains and memorized answers are a failed search regardless of
score.

## Scaling up

Three or four candidates is a good afternoon by hand. Past that, hand-running the loop is the
bottleneck and the archive gets sloppy. Escalate to `running-harness-search`, which runs the
propose-evaluate-log loop automatically and enforces all three laws by construction.

**REQUIRED SUB-SKILL:** `building-eval-sets` if there is no scored eval set yet. Without one, none
of this applies — you are back to "seems better".
