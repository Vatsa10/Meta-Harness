# Real-world LangGraph use cases

Three multi-agent LangGraph apps with tools that read and write a local stateful backend (reset per
task), public data, seeded train/search/test splits and deterministic graders. See
`docs/usecases.md` for sources, licenses, task counts and live baselines.

| Use case | Backend | Graph |
|---|---|---|
| `tau_retail/` | JSON order DB (tau-bench retail subset) | triage -> agent <-> scripted customer |
| `sql_analyst/` | Chinook SQLite + scratch schema | planner -> writer -> executor -> verifier |
| `research_ops/` | BM25 corpus + ticket JSON | retriever -> researcher -> operations |

```bash
pip install -e .[langgraph] langchain-openai
python examples/usecases/make_artifacts.py            # offline: traces.jsonl + labels.jsonl
python -m pytest -q tests/test_usecases.py            # offline smoke tests
python examples/usecases/live_baseline.py --env .env  # live baseline on the test split
```
`_common.py` holds the scripted chat model (each node tags its call `node:<name>`), tool recording,
the in-node tool loop, sandboxing and the seeded split.
