# CI gate

`meta-harness check` evaluates a baseline and a candidate harness on the same fixed tasks with
the same model, pairs the outcomes by task id, and fails the build on a regression.

```
meta-harness check --baseline A.py --candidate B.py --data fixed.jsonl --metric exact \
  --provider P --model X [--repeats N] [--max-drop 0.02] [--max-cost-increase 0.10] \
  [--alpha 0.05] [--price IN,OUT] [--out check.json] [--markdown summary.md] [--isolation process]
```

Exit codes:

- `0`: no regression.
- `1`: quality dropped significantly (paired exact McNemar at `--alpha`, more losses than wins),
  or the mean score dropped by more than `--max-drop`, or cost rose by more than
  `--max-cost-increase` (relative).
- `2`: setup error (missing file, bad metric, unbuildable model, invalid harness).

Cost needs a price: pass `--price INPUT,OUTPUT` (USD per million tokens) or set
`META_HARNESS_PRICES`. Without one the cost gate is skipped and shown as `n/a`. Context-token
deltas are always reported.

`meta-harness-check.yml` is a GitHub Actions workflow for pull requests. It reads the API key
from a repository secret and appends the markdown table to the job summary. Use a task set large
enough for McNemar to have power; with few tasks only `--max-drop` will ever trip.
