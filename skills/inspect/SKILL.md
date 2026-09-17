---
name: inspect
description: Read a finished Meta-Harness run - leaderboard, Pareto frontier, execution traces, failed tasks, and the proposer's reasoning chain across iterations. Use when the user asks why a harness won or lost, what the search actually changed, or wants to debug a candidate's failures.
---

# Inspect a finished run

Analyse the experience filesystem at `$ARGUMENTS` (default `.meta-harness` under
`${CLAUDE_PLUGIN_ROOT}`). Everything here is plain files — query them, do not read them whole.

## Leaderboard

```bash
cd "${CLAUDE_PLUGIN_ROOT}"
python -m meta_harness inspect .meta-harness
```

Or directly, sorted by score then context cost:

```bash
for f in .meta-harness/candidates/*/scores.json; do
  python -c "import json,sys;d=json.load(open(sys.argv[1]));print(f\"{d['score']:>6.3f} {d['context_cost']:>7.1f}tok {d['candidate_id']:<20} {d.get('error') or ''}\")" "$f"
done | sort -rn
```

Held-out numbers live outside the search root, in `<root>-test/test_results.json`. The search
score is not the result; the test score is.

## What a candidate actually did

```bash
C=.meta-harness/candidates/<id>

cat $C/harness.py                                     # the code
cat $C/proposer_reasoning.md                          # why it was written
grep -h '"event": "model_call"' $C/traces.jsonl | head -3   # prompts it really sent
grep -h '"event": "task_end"' $C/traces.jsonl | grep '"reward": 0'   # what it got wrong
```

`traces-1.jsonl`, `traces-2.jsonl` … are additional `--repeats` of the same candidate. A
candidate whose score moves between repeats is unstable, not better.

## The questions worth answering

- **Why did the winner win?** Diff its `harness.py` against the strongest baseline, and compare
  the first `model_call` prompt of each. The difference is usually one concrete thing: more
  examples retrieved, a tighter label primer, a verification pass, a shorter prompt.
- **What did the proposer believe?** Read `proposer_reasoning.md` in iteration order. The paper's
  §4.3 qualitative claim is that the proposer forms causal hypotheses — isolating a confounded
  edit after two regressions, then pivoting to a safer additive change. Check whether that
  happened here, and say so if it did not.
- **Did anything break?** `valid: false` candidates carry the exception in `error`. A cluster of
  identical provider errors means the run hit the gateway, not a bad harness.
- **Is the win real?** Compare score against `score_std` and the held-out number. A 0.04 gain on
  25 tasks with one repeat is noise.

## Overfitting audit

Code-space overfitting is visible in a way weight-space overfitting is not. Grep the winner for
hard-coded dataset content:

```bash
grep -nE "'(billing|bug|account)'|if .*input.*==|== *\"" .meta-harness/candidates/<id>/harness.py
```

Brittle if-chains and label lists copied from the data are a failed search, regardless of score.
Report them.

Report findings as a short written answer, not a file dump.
