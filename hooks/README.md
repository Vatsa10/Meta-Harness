# Meta-Harness function hooks

Observe failures and enforce learned artifacts, in-process. The plugin listens on exactly three
events: `tool.call`, `tool.check` and `prompt.submit`. Nothing listens on `prompt.section`:
that event's return REPLACES a cached system-prompt section, and it never carries the user's
words. Anything the model should see is attached as `context`, either on the user's turn
(`prompt.submit`) or on a tool result (`tool.call`, via `ToolCallResult.context`).

| Event | Job |
|---|---|
| `tool.call` | Append failures (error results, repeated identical calls) to the observation log; remember calls the user rejected; count calls for the drift note |
| `tool.check` | Deny a call the user already rejected this session (rejection memory), a call matching a session rule, or a call matching an installed `rule` artifact - each with the reason and its source. Read-before-edit is not one of these: Claude Code's own engine already denies an Edit/Write of a file not read this session, before the plugin ever sees the call, so the plugin does not duplicate that check |
| `prompt.submit` | Attach installed `injection` artifacts relevant to this turn; parse a human's "stop doing X" into a session rule; attach the one-time first-run line; reset the drift count when the user speaks |

Only `rule` and `injection` artifacts are enforced here; `skill` and `doctrine` artifacts are
scored and recorded, but no hook installs them into a session.

## What each feature does

- **Rejection memory.** A call the user rejected is denied if it is attempted again, identically,
  in the same session. It clears when the user's later prompt mentions it, and at session end.
- **Session rules.** A human's own prompt of the form "stop running pytest" / "don't use git push
  --force" becomes a rule that denies matching calls IMMEDIATELY, for the rest of the session.
  Only a human's prompt counts: a plugin, peer, scheduled-trigger or notification prompt creates
  no rule (same origin rule the drift note uses, `userSpoke` in `drift.ts`). The rules are
  written to `<harness_home>/pending-session-rules.json` as a record only; nothing reads that
  file back, so no rule outlives its session. The end-of-session "keep it?" prompt in the design
  is not implemented.
- **First-run line.** On the first turn after install, if `<harness_home>/waste.json` exists and
  covers at least one session, one hedged line is attached: an estimated count of calls that may
  have gone to work the user later corrected, pointing at `/meta-harness:harness waste`. That file is written
  only by running `meta-harness waste` (or `/meta-harness:harness waste`); nothing runs it in the background,
  so without a prior run there is no line. Shown once, gated by `<harness_home>/bootstrap.json`.
- **Drift note - built, tested, ships DISABLED.** Counts tool calls since the user last spoke and,
  on the call that reaches `min_calls`, appends one note to that tool result. The ship gate
  (`tools/tune_drift.py`) refused every judge it was given - word-overlap precision was about
  0.01, the kNN judge never fired, and the model judge was never evaluated - so nothing enables it.
  It turns on only if `<harness_home>/drift.json` says `{"enabled": true, "min_calls": N}`, and
  then "enabled" means a plain call-count threshold: the `judge` field is validated but no judge
  runs live.

## Failing open

No handler may break a turn or submit it twice, but they get there differently:

Every `on(...)` passes a function literal written in the call, because Claude Code's loader
rejects the whole module otherwise ("the hook is not a function literal or the name of one"):
`on('tool.call', afterCall(...))` loads in the unit fakes but not in Claude Code.
`tests/test_hook_literals.py` checks this against the source. Each literal calls a guard:

- `guardBefore` runs handlers that call `next` themselves (`tool.check`, and the `prompt.submit`
  handlers for injection, session rules and the drift reset). A throw BEFORE `next` falls
  through to `next(event)`; `next` is never called a second time.
- `guardAfter` runs observers that work after the tool (`tool.call` observation and rejection
  tracking). `next` is called once, up front; a throw costs the observation, never a re-run of
  the tool.
- `guardAfterMap` is `guardAfter` for a handler that returns a replacement outcome (the drift note
  appended to a copy of the tool result). On a throw, core's outcome is returned untouched.
- `safely`, `afterCall` and `afterCallMap` are the same guards as hook-returning wrappers, kept
  for the unit tests; never pass them to `on`.
- The first-run handler does its own fail-open handling, because it does fallible work (the
  marker write) after `next`: `next` is called exactly once, and a failed write is swallowed.

Hooks never call a model and never block on the network; anything expensive belongs in
`python -m meta_harness`.

## Not built

The design also sketches a `turn.complete` counter; it is not implemented, and nothing here
depends on it.

No environment-snapshot hook ships here. The paper's winning TerminalBench-2 discovery - an
environment snapshot injected before the first model call - was measured not to transfer to a
developer's own repository, where the environment is already known (50 orientation calls across
344 sessions on this machine). See
[docs/superpowers/specs/2026-09-26-drift-and-waste-design.md](../docs/superpowers/specs/2026-09-26-drift-and-waste-design.md).

Function hooks are early access. Enable them before loading the plugin:

```sh
export CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1
claude --plugin-dir .
```
