# Meta-Harness as a Claude Code plugin

The repository is itself the plugin: `.claude-plugin/plugin.json` sits at the root next to the
Python package, so the skills can call `python -m meta_harness` from `${CLAUDE_PLUGIN_ROOT}`
with nothing else to install.

## Install

Try it without installing:

```bash
claude --plugin-dir /path/to/Meta-Harness
```

Install it properly from the bundled marketplace:

```bash
/plugin marketplace add /path/to/Meta-Harness
/plugin install meta-harness@meta-harness-marketplace
```

Then, once per machine:

```bash
cd /path/to/Meta-Harness
pip install -e .
export UNIKEY_API_KEY=sk-...        # or put it in .env beside the plugin
```

## What it adds

| Skill | Does |
|---|---|
| `/meta-harness:search <dataset>` | Runs a harness search on a labelled dataset. Haiku 4.5 executor, Opus proposer. Reports the frontier and the held-out score. |
| `/meta-harness:inspect [root]` | Reads a finished run: leaderboard, traces, failed tasks, proposer reasoning, overfitting audit. |
| `/meta-harness:optimize-agent <tasks>` | Searches over a terminal agent's own harness (system prompt, step budget, history packing), scored by test-command exit code, in Docker. |
| `/meta-harness:apply [id]` | Lifts a winning harness into your own codebase or turns it into a skill, with an audit first. |

One agent: `meta-harness:harness-analyst` — read-only, explains what a finished run found and
whether the win is real. Invoke it by name or let Claude pick it up after a run.

## Why Haiku 4.5 is the default harness model

The paper's agentic-coding result (§4.3, Table 7) is on Claude Haiku 4.5: searched harnesses
reached **37.6%** on TerminalBench-2, ahead of every hand-engineered agent reported on that
model.

| Harness (Haiku 4.5) | Pass |
|---|---|
| OpenHands | 13.9 |
| Claude Code | 27.5 |
| Terminus 2 | 28.3 |
| Mini-SWE-Agent | 29.8 |
| Terminus-KIRA | 33.7 |
| Goose | 35.5 |
| **Meta-Harness** | **37.6** |

The gain is larger on Haiku than on Opus 4.6 (76.4 vs Terminus-KIRA's 74.7) — a weaker executor
leaves more headroom in its scaffolding for the search to find. So the default pairing is a
cheap executor with a strong proposer. That is the intended shape, not a budget compromise, and
it is also what makes a 10-20 iteration run affordable.

Override either side:

```bash
python -m meta_harness run ... --model claude-opus-4-8 --proposer-model claude-opus-4-8
```

## Cost shape

The proposer runs once per iteration; the executor runs once per task per candidate per repeat.
With 25 tasks, 10 iterations and 1 repeat that is ~275 executor calls and 10 proposer calls, so
the executor model choice dominates the bill. Every call is cached on disk by
`(model, prompt, kwargs)`, so re-running an unchanged candidate is free.

## Development

```bash
claude plugin validate .        # manifest check
claude --plugin-dir .           # load without installing
/reload-plugins                 # pick up edits mid-session
```

Skills live in `skills/<name>/SKILL.md`, the agent in `agents/harness-analyst.md`. Both are at
the plugin root — never inside `.claude-plugin/`, which holds only the manifests.

The proposer's own contract (the file handed to the coding agent inside the search loop) is a
different thing and lives at `meta_harness/skill/SKILL.md`. Editing that changes how the search
proposes candidates; editing `skills/` changes how you drive the search from Claude Code.
