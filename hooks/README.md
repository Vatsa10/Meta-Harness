# Meta-Harness function hooks

Observe failures and enforce learned artifacts, in-process.

| Hook | Job |
|---|---|
| `tool.call` | Append failures (error results, repeated identical calls) to the observation log |
| `tool.check` | Apply installed `rule` artifacts; deny with the reason and the artifact id |
| `prompt.section` | Inject installed `injection` artifacts relevant to this turn |
| `turn.complete` | Count turns since the last learn cycle |

Function hooks are early access. Enable them before loading the plugin:

```sh
export CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1
claude --plugin-dir .
```

Every handler is wrapped in `safely`, so a hook that throws logs and lets the turn proceed.
Hooks never call a model and never block on the network; anything expensive belongs in
`python -m meta_harness`.
