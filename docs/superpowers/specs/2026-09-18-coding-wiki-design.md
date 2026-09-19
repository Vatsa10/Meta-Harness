# Coding Wiki Design

A knowledge base that grows from Claude Code's own session history, is curated on a weekly
cycle, and evicts entries on measured use rather than on age.

## 1. Goal

Stop re-deriving things already learned. Every session on this machine already writes a
transcript; the knowledge in those transcripts currently dies with the session. This turns it
into durable, retrievable, self-pruning entries.

The paper's design applied to knowledge: a filesystem the agent queries selectively, never a
blob packed into context.

## 2. Why this shape (evidence)

Measured from 37 real sessions on this machine (`.meta-harness/history/report.json`):

| Signal | Value | What it implies |
|---|---|---|
| tool errors | 1,136 (3.6% of calls) | The same failures recur; their fixes are re-derived |
| thrash episodes | 130 | One tool hammered in a six-turn window - a known-gotcha shape |
| corrections | 49 | A human re-explaining something already explained |
| read:write | 0.40 | Context is re-read rather than remembered |

And the hard constraint, measured in the harness search: a session re-sends its preamble every
turn at roughly 30k tokens. **Anything always-present is paid per turn, forever.** That single
number rules out "put the knowledge in CLAUDE.md" as a scaling strategy and dictates the
index/body split below.

## 3. Content

Four entry types, all four in scope:

| Type | Holds | Example |
|---|---|---|
| `gotcha` | A failure and its fix | Windows subprocess decoding needs an explicit encoding |
| `fact` | How a project works | This repo's test command; where baselines live |
| `decision` | Why something is the way it is | Why Haiku is the default harness model |
| `solution` | Symptom to remedy | `UnicodeDecodeError: charmap` to the encoding fix |

## 4. Storage

```
~/.claude/wiki/
  INDEX.md              one line per live entry - the only always-on cost
  entries/<id>.md       frontmatter + body; opened only when relevant
  inbox/<id>.md         mined, not yet accepted
  archive/<id>.md       evicted, retained for audit
  tombstones.json       ids proven wrong - never re-mine
```

### Entry schema

```yaml
---
id: windows-subprocess-encoding      # kebab-case, stable, used in links
type: gotcha                         # gotcha | fact | decision | solution
scope: global                        # global | project:<claude-code-slug>
triggers: [cp1252, UnicodeDecodeError, subprocess, charmap]
confidence: confirmed                # confirmed | provisional | disputed
sources: ["201ad322#412"]            # session id # turn index
links: [pythonutf8-mode]             # ids of related entries
uses: 3
last_used: 2026-09-20
created: 2026-09-18
---
Windows defaults subprocess text decoding to cp1252, which raises on the first non-ASCII byte a
tool prints. Pass encoding="utf-8", errors="replace" to every subprocess.run that sets text=True.
```

**Constraints.** Body <= 150 words. `triggers` are the retrieval keys and must be terms that
appear in the situation, not in the answer - an error string, a filename, a command. `sources` is
required and makes every claim checkable, the same rule applied to the discovered doctrine.

### INDEX.md

One line per entry, `id | type | triggers`. At ~15 tokens per line, 150 entries is ~2.3k standing
tokens. The index is capped (§7); it is the only part of the wiki that is always in context.

## 5. Retrieval

1. `INDEX.md` is imported by `~/.claude/CLAUDE.md` and therefore always present.
2. When a situation matches an index line's triggers, Claude opens that one entry file.
3. Opening an entry records the hit: `uses += 1`, `last_used = today`.

The usage counter is the only honest evidence an entry earns its standing cost, so writing it
back is part of retrieval, not an optional extra.

## 6. Lifecycle

### Grow (weekly, gated)
`wiki mine` clusters history into candidate entries:
- repeated `tool_error` on the same tool with the same error text -> `gotcha`
- a `correction` episode whose tools and text recur -> `gotcha` or `fact`
- a symptom string appearing in two or more sessions -> `solution`

Candidates land in `inbox/`. **Nothing asserting a new fact enters `entries/` without review.** A
wrong entry teaches every future session something false; the inbox is the gate that prevents it.

### Learn (continuous)
Retrieval increments `uses` and stamps `last_used`. No other signal is trusted for eviction.

### Compress (weekly, automatic)
- Three or more live entries sharing two or more triggers -> merged into one, the originals
  archived with a pointer to the merged id.
- A body over 150 words -> rewritten shorter, meaning preserved.

Merging and archiving are reversible and evidence-driven, so they need no gate.

### Evict (weekly, automatic, three distinct reasons)
| Reason | Rule | Result |
|---|---|---|
| Unused | `uses == 0` and `created` older than 90 days | `archive/` |
| Superseded | A newer entry contradicts it | `archive/` with `superseded_by` |
| Wrong | The user says so | `tombstones.json`, never re-mined |

Age alone never evicts. An entry used twice in two years is doing its job.

## 7. Index cap

The index is capped at 150 lines: every `scope: project:<current>` entry first, then global
entries by `uses` descending. Entries outside the cap stay retrievable by grep - they only lose
their standing line. This keeps the always-on cost bounded no matter how large the wiki grows.

## 8. Cycles

**Weekly - wiki.** Mine to inbox, count, compress, evict, rebuild INDEX.md, write a short report
naming what changed and what awaits review.

**Monthly - doctrine.** Re-run `compare --split <last change date>` against the frozen baseline,
and run a scored harness search against the task set. Doctrine changes only via a scored search;
editing CLAUDE.md on a hunch is the unmeasured hand-tuning `optimizing-harnesses` exists to stop.

## 9. Interfaces

```bash
meta-harness wiki mine      [--limit N] [--project P]   # history -> inbox/
meta-harness wiki accept    <id> | --all                # inbox/ -> entries/
meta-harness wiki reject    <id> [--wrong]              # drop, or tombstone
meta-harness wiki curate    [--dry-run]                 # compress + evict + rebuild index
meta-harness wiki search    <query>                     # grep triggers and bodies
meta-harness wiki use       <id>                        # record a retrieval
meta-harness wiki status                                # counts, index size, pending review
meta-harness wiki week                                  # the weekly cycle, end to end
```

Plus a model-invoked skill, `consulting-the-wiki`, that tells Claude when to search it and
requires recording the hit.

## 10. Measurement

The wiki is itself a harness change, so it gets judged like one:

| Metric | Source | Meaning |
|---|---|---|
| hit rate | `uses` summed per week | Is it consulted at all? |
| dead fraction | entries with `uses == 0` past 30 days | Early warning, 60 days ahead of the 90-day eviction in section 6: a high value now predicts a large sweep later |
| index tokens | `INDEX.md` size | The bill being paid every turn |
| corrections/session | `compare` | Whether re-explaining drops |
| tool errors/session | `compare` | Whether known failures stop recurring |

Baseline to beat, frozen 2026-09-18: 37.2 tool errors, 4.33 thrash, 1.63 corrections per session;
read:write 0.39.

A wiki whose dead fraction exceeds half is failing and should be pruned hard, not grown.

## 11. Constraints

- Python 3.10+, standard library only. No new runtime dependencies.
- Windows-compatible; every subprocess gets an explicit encoding (enforced by an existing test).
- Everything stays on this machine. Bodies may contain code and project detail; nothing is sent
  anywhere, and mining inherits the existing redaction default.
- Entries are plain Markdown, readable and editable by hand without the tool.

## 12. Out of scope

- Embeddings or a vector index. Triggers plus grep is enough at this size, and adds no dependency.
- Syncing or sharing wikis between machines.
- Auto-editing CLAUDE.md from wiki content. Doctrine changes go through a scored search.
- Mining anything other than local Claude Code transcripts.

## 13. Risks

| Risk | Mitigation |
|---|---|
| Auto-mined entries are wrong | Inbox gate; `sources` on every claim; tombstones stop resurrection |
| Retrieval never records a hit, so eviction deletes live entries | `wiki use` is part of the retrieval skill; dead fraction is reported weekly and a suspiciously high one means instrumentation broke, not that the wiki is dead |
| Index grows until it taxes every turn | Hard cap at 150 lines, project-scoped first |
| The weekly cycle stops being run | It is one command, and `wiki status` shows staleness |
| Knowledge goes stale as the codebase moves | `confidence` downgrades on contradiction; superseded entries archive with a pointer |
