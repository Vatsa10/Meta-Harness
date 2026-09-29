# Decision Receipts Design

Status: approved for planning, 2026-09-29.

Extends `2026-09-26-drift-and-waste-design.md`. The receipt pattern is borrowed from keel
(`keel/docs/decision-architecture.md` §5, MIT): every decision leaves a bounded record at the
moment it is made, with the observed outcome attached afterwards.

---

## 1. Why

`meta-harness learn` tries to prove an artifact helps by replaying a past failure. Run live on
this machine's history, **0 of 6 replays reproduced the failure** — a replay of an old
conversational request almost never recreates the exact mistake. So `learn` can refuse unproven
artifacts, but it can never prove a good one.

The fix is to stop measuring backwards and measure forwards. Every time the plugin acts, it
writes a receipt. What happens next in the same session is the outcome. A small random
**holdout** — the plugin records that it would have acted, and does not — makes the comparison
causal: acted cases versus held-out cases, same artifact, same kind of moment.

Success: `meta-harness receipts report` says, per artifact, whether acting measurably reduced the
failure it targets, and says "not enough data" until it can. `learn` keeps or proposes retiring
artifacts on that evidence.

## 2. What gets a receipt

| Intervention | Receipt | Eligible for holdout |
|---|---|---|
| Learned rule denies a call (`tool.check`) | yes | **yes** |
| Learned injection attaches context (`prompt.submit`) | yes | **yes** |
| Drift note on a tool result (`tool.call`) | yes | **yes** |
| Session rule denies ("stop doing X") | yes | **no** — the user's direct order |
| Rejection memory denies | yes | **no** — the user already said no |

Only the plugin's own learned judgement is ever withheld. Anything that exists because the user
said so is always applied; its receipt is still written, marked `acted`, for reporting.

## 3. Holdout

- Default rate **10%**, configurable in `<harness_home>/receipts.json` as
  `{"holdout_rate": 0.1}`. `0` disables holdout entirely. Missing, corrupt or out-of-range config
  means the default; a rate above 0.5 is clamped to 0.5 so a misconfiguration can never disable
  most of what the user accepted.
- The draw is per intervention, independent, uniform.
- A held intervention does exactly what the plugin would do if the artifact were not installed:
  the call proceeds, no context is attached, no note appears.

## 4. Receipt record

Appended to `<harness_home>/receipts-<sessionId>.jsonl`, one JSON object per line:

```json
{"ts": "...", "session": "...", "call": 37, "event": "tool.check",
 "source": "learned-rule", "artifact": "<id or null>", "signature": "tool_error:Bash:...",
 "tool": "Bash", "decision": "acted" | "held"}
```

- `call` is the session's tool-call sequence number (§5).
- `source` ∈ `learned-rule`, `learned-injection`, `drift-note`, `session-rule`,
  `rejection-memory`.
- No message text and no tool input — signatures and counters only.
- Writing is fail-open: a failed write is logged and never blocks, changes or retries the
  decision.

## 5. Call sequence

The observer logs only failures, so "within the next K calls" needs a shared counter:

- A per-session `call` counter, incremented once per `tool.call`, lives in hook state.
- Every `observed-<session>.jsonl` record gains the same `call` field.
- Receipts and observations are then joined on `(session, call)` ranges without timestamps.

## 6. Outcome attribution (offline, Python)

`meta_harness/receipts.py` joins receipts to observations:

- **Recurred**: an observed `tool_error` with the receipt's `signature` in the same session at
  `call` in `(receipt.call, receipt.call + K]`, K = 10.
- Secondary, reported but not used for verdicts: calls until the stretch ended, and whether the
  stretch ended in a correction (from `meta_harness.waste`).

Per artifact: acted count, held count, recurrence rate in each group, and the difference.

## 7. Verdicts — the same honesty rule as the drift gate

- Fewer than `MIN_PER_ARM = 5` acted or held receipts: **not enough data**. No verdict.
- Otherwise: **helps** when acted recurrence is lower than held by at least `MIN_EFFECT = 0.2`
  and a two-proportion z-test clears 1.96; **no measurable effect** otherwise.
- The thresholds are module constants, not parameters.
- Session rules and rejection memory have no held arm, so they report counts and recurrence
  only, never a verdict — stated in the output.

## 8. Surfaces

- `meta-harness receipts report [--json]`: the table of §6-7 per artifact and per source.
- `meta-harness receipts export [--out PATH]`: one JSONL case per receipt with its outcome
  attached, keel-style.
- `meta-harness learn --status`: each installed artifact carries its receipt verdict.
- Retirement: an installed artifact with a **no measurable effect** verdict on sufficient data is
  proposed for retirement, alongside the existing quiet-signature rule. Proposed, never performed.

## 9. Constraints

Same as the prior specs: stdlib only, `dependencies = []`; `subprocess.run(text=True)` passes
`encoding="utf-8", errors="replace"`; Windows-safe paths and explicit utf-8 I/O; every hook
registration is a function literal, one `on()` per event, `$` never passed out of the hook,
`$.env.get` with literal names (the loader rules); `claude plugin validate` must pass; tests use
the real event shapes (top-level args at `tool.call`, `input` at `tool.check`); no AI
attribution anywhere.

Randomness in hooks goes through one module-level draw function that tests can replace, so
holdout behaviour is testable deterministically.

## 10. Out of scope

- Changing any decision based on receipts automatically. Receipts inform; the user decides.
- Holding out session rules or rejection memory.
- Any UI beyond the CLI and `/meta-harness:harness`.

## 11. Risks

- **Slow evidence.** An artifact that fires twice a week needs weeks to reach `MIN_PER_ARM`. The
  report says "not enough data" rather than guessing.
- **Holdout cost.** 10% of learned interventions are withheld by design. Configurable to 0.
- **Confounding.** Held and acted cases are randomised per intervention, which controls for the
  moment; the sample is still small and the report says so.
