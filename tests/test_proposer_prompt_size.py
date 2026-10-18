import json
import re
from dataclasses import asdict, is_dataclass

import pytest

from meta_harness.core import CandidateEvaluator, SearchConfig, SearchRunner
from meta_harness.metrics import estimate_tokens
from meta_harness.proposer_agent import LAYOUTS, PROMPT_PATH, AgentProposer, render_prompt
from meta_harness.providers import model_from_environment

TOKEN_CAP = 1_100  # the pre-mission prompt was ~1,500 estimated tokens per first request
SEED = "class Harness:\n    def run(self, task, model, trace):\n        return model(task['input'])\n"
NEW = "class Harness:\n    def run(self, task, model, trace):\n        return 'fruit'\n"
WRITE = {"name": "write_candidate", "arguments": {"source": NEW, "rationale": "always fruit"}}


def _plain(tool):
    return asdict(tool) if is_dataclass(tool) else tool


def first_request(tmp_path, view):
    seen = []

    class Spy:
        def __init__(self, inner):
            self.inner = inner

        def chat(self, system, messages, tools, **kwargs):
            if not seen:
                seen.append((system, json.loads(json.dumps(messages, default=str)),
                             [_plain(t) for t in tools]))
            return self.inner.chat(system, messages, tools, **kwargs)

    rows = tmp_path / "p.jsonl"
    rows.write_text(json.dumps({"tool_calls": [WRITE]}), encoding="utf-8")
    seed = tmp_path / "seed.py"
    seed.write_text(SEED, encoding="utf-8")
    proposer = AgentProposer(Spy(model_from_environment("scripted", str(rows))), view=view,
                             max_tool_calls=5, summarizer=lambda text: "short summary")
    config = SearchConfig(root=tmp_path / "store", iterations=1)
    SearchRunner(config, CandidateEvaluator(lambda p, **_: "vehicle"), proposer).run(
        [seed], [{"input": "apple", "label": "fruit"}, {"input": "car", "label": "vehicle"}],
        [{"input": "pear", "label": "fruit"}])
    assert (config.root / "candidates" / "iteration-0001-00" / "harness.py").is_file()
    return seen[0]


def request_tokens(system, messages, tools):
    return (estimate_tokens(system) + estimate_tokens(json.dumps(messages, default=str))
            + estimate_tokens(json.dumps(tools, default=str)))


@pytest.mark.parametrize("view", ["full", "scores", "summary", "seed"])
def test_first_request_is_a_short_mission_for_every_view(tmp_path, view):
    system, messages, tools = first_request(tmp_path, view)
    prompt = messages[0]["content"]
    assert request_tokens(system, messages, tools) <= TOKEN_CAP
    assert LAYOUTS[view] in prompt  # the evidence layout of the active view
    assert "Read the files before you write" in prompt and "write_candidate" in prompt
    assert "at most 5 tool calls" in prompt.lower()
    assert not re.search(r"\{[a-z_]+\}", prompt)
    # nothing from the run is inlined: no scores table, no trace lines
    assert '"score"' not in prompt and '"event"' not in prompt and "apple" not in prompt


def test_view_layouts_list_only_files_that_view_holds():
    assert "traces.jsonl" in LAYOUTS["full"]
    for view in ("scores", "summary", "seed"):
        assert "traces.jsonl" not in LAYOUTS[view] and "proposer_reasoning.md" not in LAYOUTS[view]
    assert "summary.md" in LAYOUTS["summary"] and "summary.md" not in LAYOUTS["scores"]


def test_every_placeholder_is_filled():
    text = render_prompt(PROMPT_PATH.read_text(encoding="utf-8"), iteration=2, count=3, index=1,
                         domain="math", max_tool_calls=9, objective="maximize score",
                         layout=LAYOUTS["scores"])
    assert not re.search(r"\{[a-z_]+\}", text)
    assert "iteration 2" in text and "candidate 1 of 3" in text and "math" in text
