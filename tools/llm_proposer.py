"""A proposer you can run without a coding-agent CLI.

Reads the experience filesystem the same way the agent proposer's skill describes, packs a
bounded slice of it into one prompt, and asks a gateway model for new candidate harnesses.
Weaker than a real coding agent (it cannot decide what to read), but it needs no extra tool
and makes a full search runnable anywhere.

    meta-harness run ... --proposer-command "python tools/llm_proposer.py"

Environment: META_HARNESS_ROOT, META_HARNESS_OUTPUT, META_HARNESS_ITERATION,
META_HARNESS_COUNT (set by the runner). By default it calls the local `claude` CLI
(no API key); META_HARNESS_PROPOSER_PROVIDER and META_HARNESS_PROPOSER_MODEL
override that.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meta_harness.providers import model_from_environment  # noqa: E402

MAX_SOURCE_CHARS = 4000
MAX_TRACE_LINES = 12
TOP_CANDIDATES = 4

INSTRUCTIONS = """You are the proposer in a harness-search loop. A harness is a single Python
file that wraps a frozen language model and decides what context it sees.

Required interface:

    class Harness:
        def run(self, task, model, trace):
            # task: classification {"input","label"?,"labels"?} | math {"problem","answer"?}
            #     | agent {"instruction","files","test_command"}
            # model(prompt) -> str        (every call is priced in input tokens)
            # trace.event(name, payload)  (written to traces.jsonl for the next proposer)
            ...

The instance persists across the tasks of one evaluation, so self is your memory. The label is
available only AFTER you predict - use it to update memory, never to produce the prediction.
Standard library only, plus meta_harness.retrieval (TfidfIndex, BM25Index) and, for agent
tasks, meta_harness.cc_harness.

AGENT TASKS: the harness under search is a Claude Code invocation. Subclass ClaudeCodeHarness
and return an AgentConfig from config(task):

    from meta_harness.cc_harness import AgentConfig, ClaudeCodeHarness

    class Harness(ClaudeCodeHarness):
        def config(self, task):
            return AgentConfig(append_system_prompt="...", prompt_template="{instruction}",
                               allowed_tools=("Read","Write","Edit","Glob","Grep"),
                               max_turns=30, model="haiku")

You control the doctrine, the instruction framing, the tool set, the turn budget and the model.
You do not control scoring: the task's own test_command runs afterwards, and the tests are
hidden from the agent. Never try to report your own score. Read agent_step trace events to see
what the agent did and where it stopped.

Two objectives: higher score, lower context cost (input tokens). Never hard-code dataset
strings or answers. Never crash on empty labels, empty memory, or an odd model reply.

Diagnose from the evidence below, then output each candidate as:

REASONING: <one paragraph: the failure you found, the change you made, what would refute it>
```python
<complete harness file>
```
"""


def read_scores(root: Path) -> list[dict]:
    results = []
    for path in sorted(root.glob("candidates/*/scores.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return results


def failures(directory: Path, limit: int = 4) -> list[str]:
    trace = directory / "traces.jsonl"
    if not trace.is_file():
        return []
    out = []
    with trace.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"event": "task_end"' not in line:
                continue
            try:
                payload = json.loads(line).get("payload") or {}
            except ValueError:
                continue
            if payload.get("reward") == 0:
                out.append(f"  predicted={payload.get('prediction')!r} on task #{payload.get('index')}")
            if len(out) >= limit:
                break
    return out


def first_prompt(directory: Path) -> str:
    trace = directory / "traces.jsonl"
    if not trace.is_file():
        return ""
    with trace.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if '"event": "model_call"' in line:
                try:
                    return str((json.loads(line).get("payload") or {}).get("prompt", ""))[:1200]
                except ValueError:
                    return ""
    return ""


def history_digest(root: Path, limit: int = 12) -> str:
    """Mined Claude Code history, if this run enabled it. Real failures beat candidate traces."""
    report_path = root / "history" / "report.json"
    if not report_path.is_file():
        return ""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    lines = ["\n=== mined Claude Code history (real sessions on this machine) ===",
             f"sessions={report.get('sessions')} tool_calls={report.get('tool_calls')} "
             f"error_rate={report.get('error_rate')} read_to_write={report.get('read_to_write_ratio')}",
             f"subagent_calls={report.get('subagent_calls')} workflow_calls={report.get('workflow_calls')} "
             f"skill_calls={report.get('skill_calls')}",
             f"episodes={report.get('episode_kinds')}",
             f"top tools={list((report.get('tool_histogram') or {}).items())[:8]}",
             f"top errors={list((report.get('tool_errors') or {}).items())[:6]}"]
    episodes_path = root / "history" / "episodes.jsonl"
    if episodes_path.is_file():
        shown = 0
        with episodes_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                try:
                    episode = json.loads(line)
                except ValueError:
                    continue
                if episode.get("kind") != "correction":
                    continue
                lines.append(f"  correction: {episode.get('detail')} after tools "
                             f"{episode.get('tools')} [{episode.get('matched')}]")
                shown += 1
                if shown >= limit:
                    break
    return "\n".join(lines)


def build_prompt(root: Path, iteration: int, count: int) -> str:
    scored = [r for r in read_scores(root) if r.get("valid")]
    broken = [r for r in read_scores(root) if not r.get("valid")]
    scored.sort(key=lambda r: (-float(r.get("score", 0)), float(r.get("context_cost", 0))))

    parts = [INSTRUCTIONS, f"\n=== iteration {iteration}: leaderboard ===" ]
    for record in scored:
        parts.append(f"{record['score']:.3f}  ctx={record['context_cost']:.0f} tok  {record['candidate_id']}")
    for record in broken:
        parts.append(f"INVALID  {record['candidate_id']}: {str(record.get('error'))[:200]}")

    for record in scored[:TOP_CANDIDATES]:
        directory = root / "candidates" / record["candidate_id"]
        source = (directory / "harness.py")
        parts.append(f"\n=== {record['candidate_id']} (score {record['score']:.3f}, "
                     f"{record['context_cost']:.0f} tok) ===")
        reasoning = directory / "proposer_reasoning.md"
        if reasoning.is_file():
            parts.append("previous proposer said: " + reasoning.read_text(encoding="utf-8", errors="replace")[:600])
        if source.is_file():
            parts.append("```python\n" + source.read_text(encoding="utf-8", errors="replace")[:MAX_SOURCE_CHARS] + "\n```")
        prompt = first_prompt(directory)
        if prompt:
            parts.append("first prompt it actually sent:\n" + prompt)
        wrong = failures(directory)
        if wrong:
            parts.append("failed tasks:\n" + "\n".join(wrong))

    run_file = root / "run.json"
    if run_file.is_file():
        try:
            manifest = json.loads(run_file.read_text(encoding="utf-8"))
            sample = manifest.get("tasks", [])[:2]
            parts.append(f"\n=== search set: {manifest.get('task_count')} tasks, e.g. ===\n"
                         + json.dumps(sample, indent=2)[:800])
        except ValueError:
            pass

    parts.append(f"\nPropose {count} new candidate harness file(s) now.")
    return "\n".join(parts)


def extract(text: str) -> list[tuple[str, str]]:
    """Return (reasoning, code) pairs."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    reasons = re.findall(r"REASONING:\s*(.+?)(?:\n```|\Z)", text, flags=re.DOTALL | re.IGNORECASE)
    pairs = []
    for index, code in enumerate(blocks):
        reason = reasons[index].strip() if index < len(reasons) else "(no reasoning given)"
        pairs.append((reason, code.strip() + "\n"))
    return pairs


def main() -> int:
    root = Path(os.environ.get("META_HARNESS_ROOT", "."))
    output = Path(os.environ["META_HARNESS_OUTPUT"])
    iteration = int(os.environ.get("META_HARNESS_ITERATION", "1"))
    count = int(os.environ.get("META_HARNESS_COUNT", "1"))
    output.mkdir(parents=True, exist_ok=True)

    model = model_from_environment(os.environ.get("META_HARNESS_PROPOSER_PROVIDER", "claude-cli"),
                                   os.environ.get("META_HARNESS_PROPOSER_MODEL", "opus"))
    prompt = build_prompt(root, iteration, count)
    print(f"[proposer] prompt {len(prompt)} chars, asking for {count} candidate(s)")
    # No max_tokens: the gateway's per-model default is more portable than guessing which
    # of max_tokens / max_completion_tokens a given upstream accepts.
    reply = model(prompt)

    pairs = extract(reply)
    if not pairs:
        print("[proposer] no python block in reply:\n" + reply[:1000], file=sys.stderr)
        return 1
    for index, (reason, code) in enumerate(pairs[:count]):
        (output / f"candidate-{index:02d}.py").write_text(code, encoding="utf-8")
        (output / f"candidate-{index:02d}.reasoning.md").write_text(reason, encoding="utf-8")
        print(f"[proposer] wrote candidate-{index:02d}.py ({len(code)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
