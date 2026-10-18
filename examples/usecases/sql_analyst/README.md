# sql_analyst - planner / SQL writer / executor / verifier on Chinook

Source: Chinook database, Luis Rocha, https://github.com/lerocha/chinook-database (MIT; copy in
`data/LICENSE-chinook.md`). `data/chinook.sqlite` is `Chinook_Sqlite.sqlite` (invoice dates 2021-2025).

40 authored questions (`questions.py`) with gold SQL; gold result sets are computed at build time and
top-N questions are checked for ties at the cut-off. 15 of them (the test slots) use business
definitions, window functions and multi-step logic. Split 15/10/15 (seed 20261018).

Graph: `planner` (LLM, settings prompt) -> `writer` (LLM + `list_tables`/`describe_table`/`run_sql`
reads and `save_result`/`insert_rows` scratch writes) -> `executor` (saves the final query as
`scratch.answer` via `save_result`) -> `verifier` (LLM, inline prompt; REVISE loops back once).
Writes can only touch the attached `scratch` schema: a SQLite authorizer denies every write to `main`
and ATTACH/PRAGMA. Both files live in a fresh sandbox per task.

Grader: multiset equality of `scratch.answer` rows with the gold rows (column order/names ignored,
column count enforced, numbers rounded to 2 decimals).

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
