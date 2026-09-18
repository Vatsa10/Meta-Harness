# Meta-Harness as a Claude Code plugin

The plugin is not a wrapper around the CLI. It is four skills that change how Claude works on
prompts, agent scaffolds and retrieval strategies — plus one agent, and the search engine as the
escalation path when hand-tuning runs out.

The repository doubles as the plugin: `.claude-plugin/plugin.json` sits at the root next to the
Python package, so `running-harness-search` can call `python -m meta_harness` from
`${CLAUDE_PLUGIN_ROOT}` with nothing else to configure.

## Install

```bash
claude --plugin-dir /path/to/Meta-Harness            # try it
# or
/plugin marketplace add /path/to/Meta-Harness
/plugin install meta-harness@meta-harness-marketplace
```

`pip install -e .` only matters if you reach the automated search. The discipline skills need
nothing installed.

## What it adds

All four skills are **model-invoked**: Claude loads them when the situation matches, the way
superpowers loads TDD. You do not type a slash command.

| Skill | Loads when |
|---|---|
| `optimizing-harnesses` | About to change a prompt, agent scaffold, retrieval or memory rule that has users — especially under deadline |
| `building-eval-sets` | There is no way to score whether a change helped |
| `reading-execution-traces` | A variant regressed and the reason is unclear |
| `running-harness-search` | Hand-tuning stalled past several candidates |

One agent: `meta-harness:harness-analyst` — read-only, reports what a finished search found and
whether the win survives the noise.

## The discipline

`optimizing-harnesses` carries three laws, each aimed at a failure measured in baseline testing
(see Testing below):

```
ONE CHANGE PER CANDIDATE
EVERY CANDIDATE KEPT
NO SCORE WITHOUT A TRACE
```

Plus two objectives (accuracy *and* context tokens) and a held-out split that is read exactly
once. These are the paper's findings restated as working rules:

| Paper | Rule |
|---|---|
| §4.3 — the proposer had to isolate a confounded prompt edit after two regressions | One change per candidate |
| §3 — full history through a filesystem, not summaries | Every candidate kept |
| Table 3 — scores-only 34.6, +summaries 34.9, raw traces 50.0 | No score without a trace |
| Table 2 — beat ACE by 7.7 points on 4× fewer context tokens | Two objectives |
| §3 — "the proposer never sees test-set results" | Held-out split read once |

## Testing

Written against measured baselines, per `superpowers:writing-skills`.

**RED.** A fresh Sonnet agent, no skill, given a deadline-pressured prompt-fixing scenario
("just fix the prompt today", two days already sunk, six reported failures). It performed better
than expected: it triaged the failures first, insisted on a 40-60 case eval set with a held-out
split, called that non-negotiable under time pressure, and warned against reactive special-casing.

What it did *not* do, and what the skill exists for:

- bundled four prompt changes into one edit and measured once — no attribution
- kept one rewritten prompt; earlier attempts survived only in git history
- planned per-label accuracy and no traces — the scores-only condition from Table 3
- added definitions, few-shot examples and a rationale field with no mention of token cost

`building-eval-sets` is therefore the *weakest* of the four: baseline Claude already does most of
it when asked directly. It earns its place on the parts baseline skipped — mechanical metrics,
two cases per class minimum, and never labelling with the model under test.

**GREEN.** The same scenario, same model, with the skill loaded and the bundling instinct stated
explicitly in the prompt.

## Why no session hook

caveman and ponytail hook `SessionStart` and `UserPromptSubmit` because a persona must persist
against drift. superpowers hooks only to inject its index skill across a 14-skill library.

Meta-Harness is a small process library, so it relies on model invocation by description, the same
mechanism `superpowers:test-driven-development` uses. Nothing is injected into sessions that are
not doing harness work.

## Why Haiku 4.5 is the default harness model

When `running-harness-search` reaches the engine, the harness model defaults to Haiku 4.5 through
the local `claude` CLI — no API key, since Claude Code is already authenticated.

The paper's agentic-coding result (§4.3, Table 7) is on Haiku 4.5: searched harnesses reached
**37.6%** on TerminalBench-2, ahead of every hand-engineered agent reported on that model
(Goose 35.5, Terminus-KIRA 33.7, Mini-SWE-Agent 29.8, Terminus 2 28.3, Claude Code 27.5,
OpenHands 13.9). The gain is larger than on Opus 4.6 (76.4 vs 74.7) — a weaker executor leaves
more headroom in its scaffolding. Cheap executor, strong proposer is the intended shape.

## Development

```bash
claude plugin validate .        # manifest check
claude --plugin-dir .           # load without installing
/reload-plugins                 # pick up edits mid-session
```

Skills live in `skills/<name>/SKILL.md`, the agent in `agents/harness-analyst.md`, both at the
plugin root — never inside `.claude-plugin/`, which holds only manifests.

The proposer's own contract — the file handed to the coding agent *inside* the search loop — is a
different thing at `meta_harness/skill/SKILL.md`. Editing that changes how the search proposes
candidates; editing `skills/` changes how Claude approaches harness work.
