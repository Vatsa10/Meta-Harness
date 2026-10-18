# Real-world LangGraph use cases

Three multi-agent LangGraph apps in `examples/usecases/` for testing Meta-Harness's LangGraph
integration. Each one has:

- prompts split between inline node code and a `settings.py` module;
- real tools that read and write a local backend, reset per task in a sandbox;
- `run(task) -> dict` and `build_graph(llm, tools)`;
- a seeded train/search/test split;
- recordings-ready traces and labels;
- a deterministic grader that checks final state, with no LLM judging.

| Use case | Source | License | Tasks (train/search/test) | Docs / rows |
|---|---|---|---|---|
| `tau_retail` | tau-bench retail (Sierra Research) | MIT | 50 (20/15/15) | 21 users, 69 orders, 50 products |
| `sql_analyst` | Chinook DB (Luis Rocha) + 40 authored questions | MIT | 40 (15/10/15) | 11 tables |
| `research_ops` | U.S. Constitution + Bill of Rights (Project Gutenberg #5, #2) + 36 authored requests | public domain | 36 (11/10/15) | 34 docs |

Raw downloads are cached in the git-ignored `data/cache/` (`build_data.py --fetch`). The committed
subsets carry their license files in each `data/` directory.

## Graders

| Use case | Score is 1 only if |
|---|---|
| `tau_retail` | the users+orders state hash equals the gold state (tau-bench's gold actions replayed through the same tools) and every required output was said |
| `sql_analyst` | `scratch.answer` equals the gold result set (multiset; column order ignored; numbers rounded to 2 dp) |
| `research_ops` | the answer matches an accepted spelling and the ticket store holds exactly the gold ticket (queue, doc_id, priority), or no ticket when none is due |

The oracle policy scores 1.0 on every task. The broken policies score 0 on all 20 tau train tasks and on 14 of 15 sql train tasks.
For research_ops the broken policy scores 3 of 11, since it is only wrong when the gold ticket differs.
Graders and the SQL scratch guard were mutation-checked: 14 mutants, all killed by `tests/test_usecases.py`.

## Live baseline (test split, no search, temperature 0)

| Use case | deepseek-flash | gpt-4o-mini |
|---|---|---|
| `tau_retail` | 0.867 (13/15) | 0.533 (8/15) |
| `sql_analyst` | 1.000 (15/15) | 0.733 (11/15) |
| `research_ops` | 0.933 (14/15) | 0.667 (10/15) |

Final-run cost: $0.18. The gpt-4o-mini cost uses its list price ($0.15 / $0.60 per M tokens). The
deepseek-flash cost is an estimate at $0.07 / $0.28 per M tokens; set the real price in
`examples/usecases/_common.py:LIVE_MODELS`. All live runs, including the runs made while tuning
difficulty, cost about $0.36, under the $2 budget. Per-task results are in
`examples/usecases/<name>/baseline_<model>.json`.

Difficulty was raised twice after the first baselines.

- **research_ops:** the first set scored 1.00 / 0.87. Requests now need date arithmetic, combine two
  provisions, include distractors and set priority from a 14-day deadline rule.
- **sql_analyst:** first 1.00 / 0.87. The 15 test slots now use business definitions, window
  functions and multi-step logic. The question years were also corrected to the DB's 2021–2025 range.
- **tau_retail:** first 0.93 / 0.33. The scripted customer now names the payment method in words,
  never by id.

gpt-4o-mini is inside the 20–80% target on all three. deepseek-flash still saturates sql_analyst and
sits above 80% on the other two, so its search headroom is small. Search gains should be measured on
gpt-4o-mini, or the tasks made harder still.

## Reproduce

```bash
python examples/usecases/<name>/build_data.py --fetch   # once, network
python examples/usecases/make_artifacts.py             # offline traces + labels
python -m pytest -q tests/test_usecases.py             # offline
python examples/usecases/live_baseline.py --env .env --limit 15
```
