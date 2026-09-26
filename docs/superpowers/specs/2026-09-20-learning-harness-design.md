# Learning Harness Design

> **Corrected by [2026-09-26-drift-and-waste-design.md](2026-09-26-drift-and-waste-design.md).**
> §4 below labels the artifact layer ordering `rule > injection > skill > doctrine` as a claim of
> `paper.pdf`. It is not: the ordering is this project's own design position and appears nowhere
> in the paper. Where this document and the newer one disagree, the newer one wins.

A Claude Code harness that improves at coding from its own failures, where every change is
scored against the failure that produced it before it is adopted, and adopted changes are
enforced by mechanism rather than suggested in prose.

> Supersedes the *Coding Wiki* design previously in this file. That design stored facts and
> retained them by a usage counter the model had to self-report. Two things killed it: the
> evidence on this machine says **behaviour** changed outcomes and recalled facts did not (a
> doctrine derived from measurements beat one written from intuition by 16% before any search
> ran), and Claude Code's function-hook surface makes the counter unnecessary — a hook that
> injects context knows what it injected. What survives from it: provenance on every artifact,
> archive-never-delete, tombstones, and the staging gate on new claims.

## 1. Goal

Turn the failures Claude Code already produces into harness changes that are measured before
adoption, and enforced after it.

The paper's claim, applied recursively: the harness is code, so search it. Claude Code is one of
the paper's own Table 7 baselines (58.0% on Opus 4.6, 27.5% on Haiku 4.5) and searched harnesses
beat it by 18.4 and 10.1 points **on the same frozen model**. That gap is scaffolding.

## 2. Evidence this is built on

Mined from 37 real sessions on this machine, 31,369 tool calls:

| Signal | Value |
|---|---|
| tool errors | 1,136 (3.6% of calls), 737 of them Bash |
| thrash episodes | 130 — one tool hammered inside a six-turn window |
| corrections | 49 — a human said "revert / that's wrong" right after tool use |
| read:write | 0.40 — edits outnumber reads 2.5 to 1 |

Measured in the harness search, and binding on every design decision below: **a session re-sends
its preamble every turn at roughly 30k tokens.** Anything always-present is paid per turn,
forever. Cost is `turns x preamble`, not prose length.

Already shipped from this evidence: a doctrine at 96,601 tokens/task against stock Claude Code's
130,692 at identical pass rate, held out at 97,028. That is the incumbent this system must beat.

## 3. Architecture

Three layers, each with one job.

```
OBSERVE            LEARN                        ENFORCE
tool.call     ->   meta-harness learn      ->   tool.check   (deny)
turn.complete      mine -> propose ->           prompt.section (inject)
                   replay-score -> stage        skills/ , CLAUDE.md
```

**Observe** — function hooks see failures live. `tool.call` receives every call and its result,
so `is_error` results and repeated identical calls are detected in-process, with no transcript
parsing. Transcript mining (`cc_history`) remains, but only for the existing 37 sessions of
history; new evidence arrives through the hook.

**Learn** — `meta-harness learn` turns an observed failure into a candidate artifact, scores it
by replay, and stages the winners. This is the existing search loop with a different unit of
work.

**Enforce** — an adopted artifact acts through the strongest mechanism that can express it.

## 4. Artifact types, strongest first

| Artifact | Mechanism | Standing token cost | Can the model ignore it? |
|---|---|---|---|
| `rule` | `tool.check` returns `{decision: "deny", reason}` | zero | **No** |
| `injection` | `prompt.section` returns text for this turn only | per-turn, only when relevant | No — it is placed, not requested |
| `skill` | `SKILL.md` loaded by description | zero until loaded | Yes |
| `doctrine` | a paragraph in `CLAUDE.md` | every turn, forever | Yes |

**The ordering is a rule, not a preference: express a fix at the strongest layer that can carry
it.** The discovered doctrine's top clause — *"Read before you edit; the Edit tool rejects unread
files"* — is prose asking for compliance. As a `tool.check` on `Edit` it is arithmetic. Same
intent, zero tokens, no compliance question.

Doctrine remains for things no mechanism can express ("never weaken a requirement to make it
easier"), and it stays under the monthly scored cycle.

## 5. The learn cycle

```bash
meta-harness learn                  # observe -> propose -> replay-score -> stage
meta-harness learn --accept <id>    # install the artifact, with provenance
meta-harness learn --reject <id>    # tombstone, so it is never re-proposed
meta-harness learn --status         # staged, installed, and rejected-with-the-number
```

### 5.1 Select a failure

Take the highest-frequency failure class not already covered by an installed artifact or a
tombstone. Classes are ranked by count, so the system works on what actually costs time.

### 5.2 Build the replay

Each failure becomes a **replay**: the smallest task that reproduces it, with a verification.

| Class | Count | Verification | Human needed |
|---|---|---|---|
| `tool_error` | 1,136 | Does the same error recur? | No |
| `thrash` | 130 | Is the same tool still hammered inside the window? | No |
| `correction` | 49 | Did the requirement actually get met? | Yes |

**~1,266 of 1,315 episodes verify mechanically.** That is what makes this buildable without a
hand-written benchmark: the failure carries its own test. Semantic episodes are reached only
when the user chooses to write one.

Replays are seeded from `file-history` pre-edit contents where available, exactly as the existing
draft-task pipeline does, with hidden tests written at scoring time.

### 5.3 Propose

The agent proposer reads the failure, its trace, and the existing artifacts, then writes **one**
candidate at the strongest applicable layer. One change per candidate — a bundled artifact cannot
be attributed, which is the first of the three laws.

### 5.4 Score by replay

A candidate is retained only if **both** hold:

1. **It fixes its origin replay** — the failure it was born from no longer occurs.
2. **It does not regress the existing task set** — score no worse; context cost is the tiebreak.

This is the rule that makes the system work on a saturated benchmark. The bar is not "raise the
mean" — which is unmeasurable when everything already scores 1.000 — but "fix this specific
thing, break nothing." Every retained artifact ships with the replay as its regression test.

### 5.5 Stage, never auto-install

A winning score justifies **proposing** an artifact, never adopting it unseen. An installed
artifact changes every future session in every repository; the blast radius is not symmetric with
the evidence. Accepting is one command.

Rejected candidates are archived with the number that killed them and tombstoned, so the same
idea is not re-proposed next cycle.

## 6. Storage

```
~/.claude/harness/
  artifacts/<id>/
    artifact.json     type, mechanism, provenance, scores
    payload.(ts|md)   the tool.check rule, prompt.section text, or SKILL.md body
    replay/           the task that verifies it: files, test_files, test_command
  staged/<id>/        scored, awaiting acceptance
  archive/<id>/       rejected or superseded, with the number
  tombstones.json     ids never to re-propose
  installed.json      what is live, and where each artifact came from
```

Every artifact carries `sources` (session id and turn) and its scores. A claim with no provenance
cannot be checked later, and an artifact nobody can check is one nobody can safely change.

## 7. The hook layer

A function-hook module (`hooks/harness.ts`), gated behind
`CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1` (Claude Code 2.1.274+):

| Hook | Job |
|---|---|
| `tool.call` | Record failures live: error results, repeated identical calls. Append-only to a local log. |
| `tool.check` | Apply installed `rule` artifacts. Deny with the reason and the artifact id. |
| `prompt.section` | Inject installed `injection` artifacts relevant to this turn; return `null` otherwise. |
| `turn.complete` | Count turns since the last cycle; nothing expensive runs inline. |

The hooks never call a model and never block on the network. Observation appends to a file;
enforcement is a lookup. Anything expensive is the CLI's job, run deliberately.

**Failure policy: every hook fails open.** A hook that throws must let the turn proceed. A
learning system that can break a session is worse than no learning system.

## 8. Cycles

| Cycle | Runs | Does |
|---|---|---|
| Continuous | `tool.call` hook | Observe failures |
| Weekly | `meta-harness learn` | One candidate, scored, staged |
| Monthly | `meta-harness compare --split <date>` plus a scored search | Check the live harness against the frozen baseline; re-tune doctrine |

Doctrine changes only through a scored search. Editing `CLAUDE.md` on a hunch is precisely the
unmeasured hand-tuning `optimizing-harnesses` exists to stop.

## 9. Measurement

Frozen baseline, 2026-09-18, 30 sessions: **37.2 tool errors, 4.33 thrash, 1.63 corrections per
session; read:write 0.39.**

| Metric | Source | Direction |
|---|---|---|
| origin-replay pass rate | `learn` | The only direct evidence an artifact works |
| tool errors / session | `compare` | Down |
| thrash / session | `compare` | Down |
| corrections / session | `compare` | Down |
| read:write | `compare` | Up |
| context / task | search | Down at equal score |

`compare` is observational — the two windows cover different work — so a moved number is a lead,
not proof. The replay score is the causal claim; `compare` only says whether it survived contact
with real work.

## 10. The experiment worth running

Take the doctrine already discovered and installed. Express it twice: once as prose in
`CLAUDE.md`, once as `tool.check` rules. Score both on the same task set.

That asks the paper's question at the layer it matters: **is the harness the words, or the
mechanism?** Nobody has published this comparison, and the apparatus to run it is the apparatus
described above.

## 11. Constraints

- Engine: Python 3.10+, standard library only. Hooks: TypeScript, no runtime dependencies beyond
  the Claude Code declarations.
- The hook shells to `python -m meta_harness` via `$.process.run` rather than duplicating logic.
- Windows-compatible; every subprocess declares an encoding (already enforced by an AST test).
- Everything stays on this machine. Observation inherits the existing redaction default.
- Artifacts are plain files, readable and removable by hand without the tool.

## 12. Out of scope

- Embeddings or a vector index.
- Syncing between machines.
- Auto-installing any artifact.
- Learning from anything but this machine's own sessions.
- Terminal-Bench in the weekly loop. It is the credibility check (~$35/candidate, ~9h), run
  deliberately when a public number is worth the spend — not part of learning.

## 13. Risks

| Risk | Mitigation |
|---|---|
| A `rule` denies a legitimate call | Rules ship with their replay; `learn --status` lists every live rule; removal is deleting a directory |
| A bad hook degrades every session | Fail open by construction; hooks do no model or network work |
| An artifact overfits its replay | It must also not regress the task set; winners are read before acceptance, as the doctrine was |
| Function hooks are early access and may change | Declarations are pinned and regenerated on upgrade; the engine works without hooks, only less directly |
| The cycle stops being run | `learn --status` reports staleness; a single command does the whole cycle |
| Saturated task set hides regressions | The replay is the primary signal precisely because the mean is uninformative |
