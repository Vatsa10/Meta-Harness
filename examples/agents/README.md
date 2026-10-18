# Offline examples: production-agent configs

Two plain-Python stand-ins for production agents. Neither imports a framework. Each has the
adapter shape `entrypoint(task, config, model) -> dict`. Real wiring is in `docs/adapters.md`.

| App | Stand-in for | Knobs in `config.json` |
|---|---|---|
| `router_graph/graph_app.py:run_graph` | a LangGraph graph: router -> math or chat node | `router.math_threshold`, node system prompts, `few_shot`, `retrieval_k`, `memory.policy` |
| `support_agent/agent_app.py:run_agent` | an OpenAI Agents SDK agent with two function tools | `instructions`, tool descriptions, `max_turns`, `tool_choice` |

Each folder has these files:

| File | Role |
|---|---|
| `config.json`, `schema.json` | the knobs, plus their bounds and frozen keys |
| `seed/` | output of `adapt init` (harness.py, config.json, schema.json) |
| `data.jsonl` | a small eval set (`input`, `label`) |
| `base_script.jsonl` | replayed base model: replies chosen by regex on the prompt |
| `proposer_script.jsonl` | replayed proposer: a code edit, then an out-of-bounds config, then a good config edit |

Regenerate a seed:

```bash
meta-harness adapt init --entrypoint graph_app:run_graph --config examples/agents/router_graph/config.json \
    --schema examples/agents/router_graph/schema.json --path examples/agents/router_graph \
    --out examples/agents/router_graph/seed
```

Search (offline, scripted):

```bash
meta-harness search --domain agent_config --harness examples/agents/router_graph/seed/harness.py \
    --search examples/agents/router_graph/data.jsonl --metric exact \
    --provider scripted --model examples/agents/router_graph/base_script.jsonl \
    --proposer-provider scripted --proposer-model examples/agents/router_graph/proposer_script.jsonl \
    --iterations 3 --candidates 1 --root .meta-harness-agents --no-cache
```

Expected verdicts are `accepted: 2` and `rejected_invalid_source: 2`. The candidates go like this:

| Candidate | Router graph | Support agent |
|---|---|---|
| `iteration-0001-00` | rejected: added `import os` (a code edit) | rejected: added `import os` (a code edit) |
| `iteration-0002-00` | rejected: `math_threshold` 1.5 is above the maximum | rejected: `max_turns` 50 is above the maximum |
| `iteration-0003-00` | accepted, scores 1.0 | accepted, scores 1.0 |
| Seed score | 2/6, because math questions fall through to the chat node | 1/4, because `lookup_order` is described as "internal helper" |
| Fix in `iteration-0003-00` | threshold set to 0.5 | a description that says it returns the shipping status |

Each refusal reason is in `candidates/<id>/receipt.json`. These runs prove the pipeline, not model
quality.
