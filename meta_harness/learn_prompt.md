You are proposing one harness artifact that stops a failure Claude Code keeps making.

Express the fix at the STRONGEST layer that can carry it. The ordering is binding:

| type | mechanism | can the model ignore it? |
|---|---|---|
| `rule` | a tool.check that denies the call, with a reason | No |
| `injection` | text placed into the turn when relevant | No |
| `skill` | a SKILL.md loaded by description | Yes |
| `doctrine` | a paragraph in CLAUDE.md, paid every turn forever | Yes |

Choose `rule` when the failure is decidable from the call and the session state alone.
Choose `injection` when the agent needs a fact at a particular moment.
Choose `skill` or `doctrine` only when no mechanism can express the fix.

Write ONE artifact, targeting only this failure. Do not bundle several changes: a bundled
artifact cannot be attributed when it regresses.

Answer in exactly this form:

TYPE: <rule|injection|skill|doctrine>
PAYLOAD:
```
<the artifact body>
```
