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
spoke up, and repeated identical failures. Then relay the report's `caveat` field verbatim -
quote it as given, do not paraphrase, round, restate its figures from memory, or drop a
qualifier. The caveat is the only statement of how reliable these numbers are.
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

2. Session rules, recorded at `<harness_home>/pending-session-rules.json`. These come from a
   human's own "stop doing X" style prompt (never from a plugin, peer, scheduled or notification
   prompt). A parsed rule takes effect IMMEDIATELY: it denies matching calls for the rest of the
   current session, with no acceptance step. The file is only a record of the rules created;
   nothing reads it back, so a rule never outlives its session and there is no way to make one
   permanent yet. (The end-of-session "keep it?" prompt the design describes is not
   implemented.) Read that file and list each rule, and say plainly that it is already in effect
   for this session and ends with it.

Report all three lists together (staged artifacts, retirement candidates, session rules), even
when one is empty, and say explicitly for each what accepting or discarding it would take (for a
session rule: nothing - it ends with the session).

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
