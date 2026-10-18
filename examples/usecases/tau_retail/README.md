# tau_retail - retail customer service (tau-bench style)

Source: tau-bench retail environment, Sierra Research, https://github.com/sierra-research/tau-bench
(MIT License; copy in `data/LICENSE-tau-bench`). Used: `orders.json`, `users.json`, `products.json`,
`wiki.md` (store policy, `data/policy.md`) and `tasks_test.py` (parsed with `ast`, never executed).

Subset: the first 50 test tasks whose gold actions use only the tools implemented here and write at
least once (2 upstream tasks skipped because a gold write itself errors). Kept: the 21 users those tasks
touch, their 69 orders, all 50 products. Split 20/15/15 (seed 20261018).

Graph: `triage` (LLM, inline prompt) -> `agent` (LLM + tools; prompt = settings + policy) <-> `customer`
(scripted: confirms actions, gives payment method / cancel reason in words, never ids) -> END.
Tools: reads `find_user`, `get_user`, `get_order`, `list_products`; writes `cancel_order`,
`modify_order`, `change_address`, `return_items`, `exchange_items` (tau-bench semantics and policy checks).
Backend: JSON files copied into a fresh sandbox per task.

Grader: final users+orders state hash equals the gold state (gold actions replayed through the same
tools) AND every required output appears in the agent's messages (tau-bench's reward).

## Files
| File | Role |
|---|---|
| `app.py` | the LangGraph graph: `build_graph(llm, tools)` seam and `run(task, llm=None) -> dict` entry |
| `settings.py` | the owner's settings module (some prompts + knobs); other prompts are inline in `app.py` |
| `tools.py` | stateful backend + LangChain tools (reads and writes), `TOOL_KINDS` read/write map |
| `oracle.py` | scripted fake-model policies: `oracle_model(task)` (gold) and `broken_model(task)` |
| `grader.py` | deterministic grader `grade(task, result)` / `metric(task, result)` (final-state checks, no LLM) |
| `build_data.py` | `--fetch` downloads the public source into git-ignored `data/cache/`; then builds `data/` + `tasks.jsonl` |
| `tasks.jsonl` | tasks with a seeded `split` (train/search/test) and gold |
| `traces.jsonl` | recordings-ready runs (`tool_calls: [{name, args, output, write}]`), oracle on every task + broken on train |
| `labels.jsonl` | `meta-harness label` rows (good/bad) derived from the grader |
| `baseline_<model>.json` | live baseline on the test split (no search) |
