# Offline example: sentiment

Runs the whole search loop with no API key and no network, using the `scripted` provider.

| File | Role |
|---|---|
| `sentiment/harness.py` | seed harness: passes the model's reply through unparsed |
| `sentiment/data.jsonl` | 30 hand-written reviews, labels `positive` / `negative` / `neutral` |
| `sentiment/metric.py` | custom metric `score(task, result) -> float` |
| `sentiment/base_script.jsonl` | replayed base model: chatty replies chosen by regex on the prompt |
| `sentiment/proposer_script.jsonl` | replayed proposer: `list`, `read`, `grep`, then `write_candidate` |

```bash
meta-harness search --harness examples/sentiment/harness.py --data examples/sentiment/data.jsonl \
    --metric examples/sentiment/metric.py:score \
    --provider scripted --model examples/sentiment/base_script.jsonl \
    --proposer-provider scripted --proposer-model examples/sentiment/proposer_script.jsonl \
    --iterations 2 --root .meta-harness-example --no-cache
meta-harness inspect .meta-harness-example
```

Expected: `initial-0000` scores 0 (replies like "Sounds Positive to me." never equal a label);
`iteration-0001-00` maps replies onto the declared labels and scores 1 on search and on the
held-out split in `.meta-harness-example-test/test_results.json`. The script holds four proposals,
so up to four iterations replay. This proves the pipeline, not model quality.

To use your own model, swap `--provider scripted --model <script>` for a real provider and model
(see the top-level README) and drop the proposer script flags.
