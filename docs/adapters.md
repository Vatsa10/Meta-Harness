# Agent adapters: searching a production agent's config

Production agents live in LangGraph, the OpenAI Agents SDK, the Claude Agent SDK or a custom
service. Meta-Harness does not import those frameworks. Instead the harness becomes a config, and
the search edits that config.

| Piece | What it is |
|---|---|
| config | a JSON file with the knobs: system prompts, tool descriptions, few-shot examples, routing thresholds, retrieval k, memory policy |
| schema | types, enums, numeric bounds, max string lengths, required keys, and `frozen` keys the proposer may not change |
| entrypoint | your `module:function(task, config, model) -> dict`, which runs your real agent with `config` |
| harness | a generated `harness.py` that loads its sibling `config.json` and calls the entrypoint |

## Quick start

```bash
meta-harness adapt init --entrypoint my_agent:run --config agent_config.json \
    --schema agent_schema.json --path src/ --out seeds/my_agent
meta-harness search --domain agent_config --harness seeds/my_agent/harness.py \
    --data evals.jsonl --metric exact --provider openai --model gpt-4.1-mini --iterations 10
```

`adapt init` writes `seed_dir/harness.py`, `config.json` and `schema.json`, refuses a config that
already breaks its schema, and imports the entrypoint once to check it resolves. `--path` is the
directory your module is imported from. If it is inside this repository it is stored relative to
the package, so the generated file is the same on every checkout. Otherwise it is stored as an
absolute path.

A search copies only `harness.py` into each candidate directory. So the generated file also
carries config.json verbatim, between `# BEGIN CONFIG` and `# END CONFIG`. For a candidate, that
block is config.json. When a sibling `config.json` exists, the harness loads it, and the gate
requires it to equal the block. After you edit `seed_dir/config.json` or `schema.json` by hand,
run `meta-harness adapt sync seed_dir` to re-render harness.py.

## What the gate enforces (`--domain agent_config`)

The `agent_config` domain registers a VALIDATE hook. The search runs it on every candidate before
evaluation, and a refused candidate is recorded as `rejected_invalid_source` with the reason in its
`receipt.json`. A candidate is refused when:

- any byte of harness.py outside the config block differs from its seed. That covers new imports,
  edited functions, a changed `ENTRYPOINT`, and a loosened `SCHEMA_JSON`;
- the config block is not valid JSON, or differs from a sibling config.json;
- the config violates the seed's schema. That covers a wrong type, a value outside an `enum`, a
  number outside `minimum`/`maximum`, a string over `maxLength` (or under `minLength`), too many
  items (`maxItems`), a missing `required` key, an unknown key, or a changed `frozen` path.

The domain's PROMPT tells the proposer to copy the seed and edit only the config block. The same
check is available without a search, as `meta-harness adapt check --candidate DIR --seed DIR`, or
as `meta_harness.adapters.validate_candidate(candidate_dir, seed_dir, schema=None)`.

### Schema format

The schema is a stdlib-only subset of JSON Schema:

```json
{
  "type": "object",
  "required": ["router", "tools"],
  "frozen": ["tools.0.name", "nodes.math.system_prompt"],
  "properties": {
    "router": {"type": "object", "properties": {
      "math_threshold": {"type": "number", "minimum": 0, "maximum": 1}}},
    "tools": {"type": "array", "maxItems": 4, "items": {"type": "object", "properties": {
      "name": {"type": "string"}, "description": {"type": "string", "maxLength": 200}}}},
    "memory": {"type": "string", "enum": ["none", "last_turn", "summary"]}
  }
}
```

An object that declares `properties` refuses keys it does not list unless
`"additionalProperties": true`. `frozen` holds dotted paths. A list index is written as a number,
and a path may name a whole subtree. Freeze what your code depends on, such as tool names or
output-format instructions that a parser reads.

## The model you are handed

`model` is Meta-Harness's metered model: `model(prompt) -> str`. Every call through it is traced,
cached and costed, and it is the same model for the seed and for every candidate. Under the default
`--isolation process`, the harness runs in a child process with no API keys in its environment and
socket connects blocked. So an agent that creates its own LLM client cannot reach its provider
there. Route your agent's LLM calls through `model`, as in the patterns below. If the agent must
call its own services, run with `--isolation none`. Only do that for code you trust. Docker
isolation mounts only the package and the candidate directory, so an entrypoint outside the
package is not importable there.

## Wiring: LangGraph

Build the graph from the config inside the entrypoint, and wrap `model` as the LLM:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict, total=False):
    question: str
    answer: str

def run(task, config, model):
    def route(state):
        score = math_score(state["question"])            # your scoring
        return "math" if score >= config["router"]["math_threshold"] else "chat"

    def node(name):
        def call(state):
            prompt = f"{config['nodes'][name]['system_prompt']}\n\nQuestion: {state['question']}"
            return {"answer": model(prompt).strip()}
        return call

    graph = StateGraph(State)
    graph.add_node("math", node("math"))
    graph.add_node("chat", node("chat"))
    graph.add_conditional_edges(START, route, {"math": "math", "chat": "chat"})
    graph.add_edge("math", END)
    graph.add_edge("chat", END)
    final = graph.compile().invoke({"question": task["input"]})
    return {"output": final["answer"]}
```

If your nodes use LangChain chat models, put a thin adapter in front of `model`. For example,
use a `langchain_core.language_models.llms.LLM` subclass whose `_call` returns `model(prompt)`.
The point is that the graph never builds its own provider client.

## Wiring: OpenAI Agents SDK

Tool descriptions and instructions come from the config. `function_tool` accepts
`name_override` and `description_override`:

```python
from agents import Agent, Runner, function_tool

def lookup_order(order_id: str) -> str:
    return ORDERS.get(order_id, "unknown order")

def run(task, config, model):
    described = {t["name"]: t["description"] for t in config["tools"]}
    tools = [function_tool(lookup_order, name_override="lookup_order",
                           description_override=described["lookup_order"])]
    agent = Agent(name="support", instructions=config["instructions"], tools=tools,
                  model=config.get("model", "gpt-4.1-mini"))
    result = Runner.run_sync(agent, task["input"], max_turns=config["max_turns"])
    return {"output": str(result.final_output)}
```

The SDK makes its own OpenAI calls. Either run this search with `--isolation none` (the
model the SDK uses is then billed outside Meta-Harness's meter), or give the agent a custom
`agents.models.interface.Model` implementation that forwards each request to `model`.
`examples/agents/support_agent/` is a plain-Python stand-in with the same shape: instructions,
two tools with descriptions, `max_turns` and `tool_choice`.

## Wiring: Claude Agent SDK

```python
import asyncio
from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, create_sdk_mcp_server, query, tool

def run(task, config, model):
    lookup = tool("lookup_order", config["tools"]["lookup_order"], {"order_id": str})(_lookup)
    server = create_sdk_mcp_server(name="store", version="1.0.0", tools=[lookup])
    options = ClaudeAgentOptions(system_prompt=config["system_prompt"], max_turns=config["max_turns"],
                                 mcp_servers={"store": server},
                                 allowed_tools=["mcp__store__lookup_order"])

    async def go():
        answer = ""
        async for message in query(prompt=task["input"], options=options):
            if isinstance(message, ResultMessage):
                answer = message.result or ""
        return answer

    return {"output": asyncio.run(go())}
```

Here `_lookup` is your async tool function. The Claude Agent SDK drives its own model and
subprocess, so it needs `--isolation none` and credentials in the environment. Score it on a fixed
eval set, and keep the search model and the deployment model the same.

## Examples (offline, no framework installed)

`examples/agents/` holds two plain-Python stand-ins, each with an entrypoint, `config.json`,
`schema.json`, a generated `seed/`, a small eval JSONL and scripted replays for the base model and
the proposer. See `examples/agents/README.md` for the commands and the expected verdicts.
