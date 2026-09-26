---
description: Inspect the harness itself - wasted work, staged/pending changes awaiting your approval, and why the last denial happened
argument-hint: waste | pending | why
---

# /harness

One command, three modes. Pick the mode from `$ARGUMENTS`; if none is given, ask which of
`waste`, `pending`, or `why` is wanted.

## `/harness waste`

Run:

```
python -m meta_harness waste --this-project --json
```

Summarize the report for the user: sessions read, corrections found, calls burned before they
spoke up, and repeated identical failures. Then state the report's own caveat verbatim (its
`caveat` field) - do not round it up or drop the qualifier. The measured number is that in a
hand-labelled audit, roughly half of flagged corrections (11 of 21, 52.4%) turned out to be
genuine; say exactly that, not "most corrections are real" or "this measures waste precisely."
This also refreshes `<harness_home>/waste.json` as a side effect, which other tooling reads.

If the user wants the wider history (not just this project), drop `--this-project` and consider
`--since` or `--limit`.

## `/harness pending`

Everything here is a proposal. Nothing listed is installed or applied - list what exists and how
to accept or discard each one:

1. Staged and installed artifacts, plus retirement candidates:

   ```
   python -m meta_harness learn --status
   ```

   The JSON has `staged` (not yet installed - accept with `python -m meta_harness learn --accept
   <ID>`, or discard with `python -m meta_harness learn --reject <ID>`), `installed`, and
   `retirement_candidates` (artifacts the harness proposes retiring because their failure
   signature no longer shows up live - this is a proposal only, nothing is retired
   automatically; reviewing and rejecting/accepting is on the user).

2. Pending session rules, staged at `<harness_home>/pending-session-rules.json`. These come from
   "stop doing X" style corrections and are never installed automatically - read that file and
   list each proposed rule. Accepting one means moving it into the harness's active session-rule
   config yourself; until then it has no effect on behavior.

Report all three lists together (staged artifacts, retirement candidates, pending session
rules), even when one is empty, and say explicitly for each what accepting or discarding it
would take.

## `/harness why`

Explain the most recent denial or drift note Claude saw in this session. Denials come from one
of four sources:

- a learned rule (an installed artifact matching the current call),
- the read-before-edit check (an edit blocked because the file wasn't read first this session),
- rejection memory (the same call was already rejected earlier this session),
- a session rule (a standing "stop doing X" instruction in effect for this session).

Quote the denial reason text exactly as it was shown - that reason string is the only evidence
available; do not invent a fuller justification than what the hook actually said. If nothing was
denied in this session, say so rather than fabricating one. Then explain, in the user's terms,
how to undo or override it: rejection-memory and session-rule denials clear at session end; a
learned rule can be reviewed and rejected with `python -m meta_harness learn --reject <ID>`
(add `--wrong` if it should also be tombstoned so it's never proposed again).
