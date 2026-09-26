# Meta-Harness function hooks

Observe failures and enforce learned artifacts, in-process.

| Hook | Job |
|---|---|
| `tool.call` | Append failures (error results, repeated identical calls) to the observation log |
| `tool.check` | Apply installed `rule` artifacts; deny with the reason and the artifact id |
| `prompt.section` | Inject installed `injection` artifacts relevant to this turn |

Only `rule` and `injection` artifacts are enforced here; `skill` and `doctrine` artifacts are
scored and recorded, but no hook installs them into a session.

The design also sketches a `turn.complete` counter; it is not implemented, and nothing here
depends on it.

No environment-snapshot hook ships here. The paper's winning TerminalBench-2 discovery — an
environment snapshot injected before the first model call — was measured not to transfer to a
developer's own repository, where the environment is already known (50 orientation calls across
344 sessions on this machine). See
[docs/superpowers/specs/2026-09-26-drift-and-waste-design.md](../docs/superpowers/specs/2026-09-26-drift-and-waste-design.md).

Function hooks are early access. Enable them before loading the plugin:

```sh
export CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1
claude --plugin-dir .
```

Every handler is wrapped in `safely`, so a hook that throws logs and lets the turn proceed.
Hooks never call a model and never block on the network; anything expensive belongs in
`python -m meta_harness`.
