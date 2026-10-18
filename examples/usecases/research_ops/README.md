# research_ops - civic research desk: retrieve, answer, file tickets

Source: the U.S. Constitution (Project Gutenberg eBook #5) and the Bill of Rights (eBook #2) - public
domain in the U.S.; the Project Gutenberg header/footer and trademark are stripped. Split into 34 docs
(24 article sections + 10 amendments) in `data/corpus.jsonl`.

36 authored requests (`questions.py`): eligibility checks needing date arithmetic, multi-hop reference
numbers, rights complaints and compliance cases, some with distractors. Each has accepted answer
spellings and a gold ticket (queue, doc_id, priority) or none. Split 11/10/15 (seed 20261018).

Graph: `retriever` (BM25 via `meta_harness.retrieval.BM25Index`, `search_docs` tool) -> `researcher`
(LLM + `search_docs`/`read_doc`, ANSWER/SOURCE lines) -> `operations` (LLM + `list_tickets` /
`file_ticket` write; ticket policy in settings, instructions inline). Tickets are a JSON file in a
fresh sandbox per task.

Grader: the ANSWER matches an accepted spelling (word-bounded, so 250 != 25) AND the ticket store holds
exactly the gold ticket - or no ticket when none is due.

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
